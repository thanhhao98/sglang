// Step-5 verify indexer in one launch: DeepSelect top-512 of each score row, then the region pack
// of the selected ids in the same CTA (an integer port of selection_triton._pack_regions).
//
// Bitwise contract with deep_select_topk + _pack_regions. topk_ids is written by DeepSelect's own
// device function under DeepSelectTopKFp32Kernel's Config, so it equals deep_select_topk's output.
// The pack depends only on the selected set (ids outside [0, current) masked to INT32_MAX, then
// sorted ascending), the row's position and request, and req_to_token, so any exact sort of the
// same ids reproduces it.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "cuda_kernels/v3_fp32/topk_select.cuh"
#include "deep_select_topk.cuh"

namespace sglang {

namespace step5_region_pack {

// _pack_regions with K = STEP5_HISTORY_REGIONS.
inline constexpr int32_t kHistoryRegions = 512;
inline constexpr int64_t kRegionIdsWidth = kHistoryRegions + 1;  // 512 history slots, then the current region
inline constexpr int32_t kMaskedId = 2147483647;                 // INT32_MAX, _pack_regions' mask fill
inline constexpr int32_t kMaxPackedTail = 4;                     // packed slots 512..515 (widths 513/514/516)

struct Params {
  const void* positions;     // [R] int32 or int64
  const void* row_to_req;    // [R] int32 or int64; request slot 0 is a padding row
  const void* req_to_token;  // [request slots, map_stride] int32 or int64, unit inner stride
  const int32_t* topk_ids;   // [R, 512], written by the devfunc of this launch
  int32_t* region_ids;       // [R, 513]
  int32_t* counts;           // [R]
  int64_t* packed;           // [R, packed_width]
  int32_t* causal_lengths;   // [R]
  int64_t map_stride;        // req_to_token.stride(0)
  int32_t packed_width;      // 513, 514 or 516
  bool positions_int64;
  bool row_to_req_int64;
  bool req_to_token_int64;
};

SGL_DEVICE int64_t load_integer(const void* ptr, bool is_int64, int64_t offset) {
  return is_int64 ? static_cast<const int64_t*>(ptr)[offset]
                  : static_cast<int64_t>(static_cast<const int32_t*>(ptr)[offset]);
}

// ((physical // 8) << 32) | logical as _pack_regions computes it on int64: Triton's integer //
// truncates toward zero like C's /, and the shift and the OR act on the two's-complement words.
SGL_DEVICE int64_t pack_region(int64_t physical, int64_t logical) {
  return static_cast<int64_t>((static_cast<uint64_t>(physical / 8) << 32) | static_cast<uint64_t>(logical));
}

// True when the CTA's keys ascend in thread order. Every thread of the CTA must call it.
template <uint32_t kThreads>
SGL_DEVICE bool keys_ascending(int32_t key, int32_t* exchange) {
  const uint32_t tid = threadIdx.x;
  exchange[tid] = key;
  __syncthreads();
  const bool ordered = tid + 1 == kThreads || key <= exchange[tid + 1];
  return __syncthreads_and(ordered) != 0;
}

// Bitonic sort of one key per thread: thread t returns the t-th smallest key of the CTA. Partners
// closer than a warp swap through shuffles; the 10 stages with partner distance >= 32 go through
// `exchange` (2 * kThreads int32), alternating halves so each stage needs one barrier: a half is
// rewritten two such stages later, after a barrier that every thread passes only once it has read
// that half. Every thread of the CTA must call it.
template <uint32_t kThreads>
SGL_DEVICE int32_t bitonic_sort_ascending(int32_t key, int32_t* exchange) {
  static_assert(kThreads >= 2 * device::kWarpThreads && (kThreads & (kThreads - 1)) == 0);
  const uint32_t tid = threadIdx.x;
  uint32_t half = 0;
#pragma unroll
  for (uint32_t size = 2; size <= kThreads; size <<= 1) {
#pragma unroll
    for (uint32_t stride = size >> 1; stride > 0; stride >>= 1) {
      int32_t other;
      if (stride >= device::kWarpThreads) {
        int32_t* slots = exchange + half * kThreads;
        slots[tid] = key;
        __syncthreads();
        other = slots[tid ^ stride];
        half ^= 1u;
      } else {
        other = __shfl_xor_sync(0xffffffffu, key, static_cast<int>(stride));
      }
      const bool ascending = (tid & size) == 0;
      const bool keep_min = ((tid & stride) == 0) == ascending;
      key = keep_min ? (other < key ? other : key) : (other > key ? other : key);
    }
  }
  return key;
}

// _pack_regions for row blockIdx.x, one history slot per thread. `exchange` holds 3 * kThreads
// int32 of shared memory. Every thread of the CTA must call it.
template <uint32_t kThreads>
SGL_DEVICE void pack_row(const Params& p, int32_t* exchange) {
  static_assert(static_cast<int32_t>(kThreads) == kHistoryRegions, "one selected region id per thread");
  const int64_t row = blockIdx.x;
  const int32_t lane = static_cast<int32_t>(threadIdx.x);

  // position = int32(Positions[row]); current = position // 8; count = min(current, 512)
  const int32_t position = static_cast<int32_t>(load_integer(p.positions, p.positions_int64, row));
  const int32_t current = position / 8;
  const int64_t request = load_integer(p.row_to_req, p.row_to_req_int64, row);
  const bool live = request > 0;
  const int32_t count = current < kHistoryRegions ? current : kHistoryRegions;

  // ids = sort(where(0 <= ids < current, ids, INT32_MAX)). No output reads them unless the row is
  // live with count > 0; that test reads only the row's own data, so it is CTA-uniform, like the
  // barriers inside the sort it guards.
  int32_t id = kMaskedId;
  if (live && count > 0) {
    // Stored by this CTA's devfunc epilogue and made visible to the CTA by the __syncthreads()
    // after the devfunc. Read through L2 (ld.global.cg), never through the read-only data path.
    const int32_t selected = __ldcg(p.topk_ids + row * kHistoryRegions + lane);
    id = selected >= 0 && selected < current ? selected : kMaskedId;
    // Rows whose end fits DeepSelect's init window (end <= 8192, its is_already_sorted_in_index)
    // leave the unsorted Config in index order and skip the network; the check makes that a
    // shortcut, not an assumption.
    if (!keys_ascending<kThreads>(id, exchange)) {
      id = bitonic_sort_ascending<kThreads>(id, exchange + kThreads);
    }
  }
  const int32_t result = lane < count ? id : (lane == count ? current : -1);

  int32_t* region_row = p.region_ids + row * kRegionIdsWidth;
  region_row[lane] = live ? result : -1;

  // The address math is int64, as in _pack_regions: request * map_stride + 8 * region.
  const int64_t mapping_row = request * p.map_stride;
  int64_t* packed_row = p.packed + row * p.packed_width;
  int64_t value = -1;
  if (live && lane <= count) {
    const int64_t logical = static_cast<int64_t>(result) * 8;
    value = pack_region(load_integer(p.req_to_token, p.req_to_token_int64, mapping_row + logical), logical);
  }
  packed_row[lane] = value;
  // Slots 512 and up (ABI padding for widths 514 and 516): the current region sits in slot 512
  // when all 512 history slots are used, -1 everywhere else.
  if (lane < kMaxPackedTail && kHistoryRegions + lane < p.packed_width) {
    int64_t tail = -1;
    if (live && count == kHistoryRegions && lane == 0) {
      const int64_t logical = static_cast<int64_t>(current) * 8;
      tail = pack_region(load_integer(p.req_to_token, p.req_to_token_int64, mapping_row + logical), logical);
    }
    packed_row[kHistoryRegions + lane] = tail;
  }
  if (lane == 0) {
    region_row[kHistoryRegions] = live && count == kHistoryRegions ? current : -1;
    p.counts[row] = live ? count + 1 : 0;
    // position + 1 wraps like Triton's int32 add (signed overflow is undefined in C++).
    p.causal_lengths[row] = live ? static_cast<int32_t>(static_cast<uint32_t>(position) + 1u) : 0;
  }
}

template <typename Kernel>
__launch_bounds__(Kernel::NUM_THREADS, Kernel::TARGET_OCCUPANCY, 1) __global__ void deep_select_region_pack_kernel(
    __grid_constant__ const TopkSelectArgs args,
    __grid_constant__ const typename Kernel::TmaParams tma_params,
    __grid_constant__ const Params pack) {
  // Every global read below may be a predecessor's output: the ends (__ldg) and the TMA-loaded
  // scores inside the devfunc, then positions, requests and req_to_token in the pack. Under a PDL
  // launch this waits for the predecessor; otherwise it returns at once.
  device::PDLWaitPrimary<true>();

  Kernel::topk_select_kernel_devfunc(args, tma_params);

  // All Kernel::NUM_THREADS threads reach this barrier, since every exit of the devfunc is
  // CTA-uniform:
  //  - topk_select.cuh:84-92 returns after the shortcut epilogue when end_vocab_idx <= topk;
  //    end_vocab_idx is __ldg(end_ptr + blockIdx.x), one address for the whole CTA, and topk is a
  //    launch argument, so the branch depends only on the row;
  //  - topk_select.cuh:554-557 returns after take_action_when_have_nan, behind
  //    have_nan = __syncthreads_or(have_nan), which hands every thread the same value;
  //  - otherwise control falls off the end after stage_output_and_epilogue (topk_select.cuh:560).
  // The only other `return` (issue_tma_loads_for_round) leaves that helper, not the devfunc. With
  // this Config (unsorted, no values) and abort_when_nan_found = false, none of these epilogues
  // syncs after the branch, so no thread can be left inside a barrier the others skipped.
  //
  // The barrier also orders the devfunc's last shared-memory reads (its epilogue reads the staged
  // indices from surviving_topk_pairs) before the sort below reuses that region, and makes the
  // devfunc's topk_ids stores visible to every thread of the CTA. TMA only ever writes
  // incoming_topk_pairs and tma_load_buf, never surviving_topk_pairs, so no async-proxy write can
  // land in the reused region either.
  __syncthreads();

  extern __shared__ CUTE_ALIGNAS(1024) char region_pack_smem[];
  auto& smem = *reinterpret_cast<typename Kernel::SharedMemoryPlanFP32*>(region_pack_smem);
  static_assert(
      3 * Kernel::NUM_THREADS * sizeof(int32_t) <= sizeof(smem.surviving_topk_pairs),
      "the region pack's sort scratch must fit in the devfunc's survivor buffers");
  pack_row<Kernel::NUM_THREADS>(pack, reinterpret_cast<int32_t*>(smem.surviving_topk_pairs));
}

// topk_select_fp32::run_topk_select_kernel (topk_select.cuh:575-601) for the fused kernel: the same
// checks, dynamic shared memory, TMA descriptor and launcher, plus an opt-in PDL launch.
template <typename Config>
void launch(const TopkSelectArgs& args, const Params& pack, bool use_pdl) {
  KU_ASSERT(args.sorted_value == Config::sorted_value, "Dispatch failure");
  KU_ASSERT(args.sorted_index == Config::sorted_index, "Dispatch failure");
  KU_ASSERT(args.return_value == Config::return_value, "Dispatch failure");
  static_assert(std::is_same_v<typename Config::ValueT, float>);
  KU_ASSERT(args.vocab_size < MAX_VOCAB_SIZE, "`vocab_size` is too big");

  using Kernel = topk_select_fp32::TopkSelectKernelFP32<Config>;
  KU_ASSERT(args.topk <= Kernel::MAX_TOPK, "topk is too large");
  static_assert(INPUT_STRIDE_ALIGNMENT_REQUIREMENT % 16 == 0);

  const auto kernel = deep_select_region_pack_kernel<Kernel>;
  constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlanFP32);
  KU_ASSERT(smem_size * Kernel::TARGET_OCCUPANCY <= args.shared_memory_size_per_sm);
  KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  KU_ASSERT(args.stride_input_batch % 4 == 0, "stride_input_batch must be 16B-aligned");
  const typename Kernel::TmaParams tma_params = {Kernel::make_topk_tensor_map(args)};

  ku::KernelLaunchConfig config{dim3(args.batch_size), dim3(Kernel::NUM_THREADS), smem_size, args.stream};
  config.use_pdl = use_pdl;
  ku::launch_kernel(config, kernel, args, tma_params, pack);
}

}  // namespace step5_region_pack

template <int kTopK>
struct DeepSelectTopKRegionPackFp32Kernel {
  // DeepSelectTopKFp32Kernel's own Config: the same selection, so topk_ids is written identically.
  using Config = typename DeepSelectTopKFp32Kernel<kTopK>::Config;
  static_assert(kTopK == step5_region_pack::kHistoryRegions, "the region pack keeps 512 history regions");

  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView ends,
      const tvm::ffi::TensorView topk_ids,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView row_to_req,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::TensorView region_ids,
      const tvm::ffi::TensorView counts,
      const tvm::ffi::TensorView packed,
      const tvm::ffi::TensorView causal_lengths,
      bool use_pdl) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto N = SymbolicSize{"width"};
    auto S = SymbolicSize{"input_stride"};
    auto W = SymbolicSize{"packed_width"};
    auto Q = SymbolicSize{"request_slots"};
    auto C = SymbolicSize{"context"};
    auto M = SymbolicSize{"mapping_stride"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto position_dtype = SymbolicDType{};
    auto request_dtype = SymbolicDType{};
    auto mapping_dtype = SymbolicDType{};

    TensorMatcher({B, N}).with_strides({S, 1}).with_dtype<fp32_t>().with_device(device).verify(scores);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(ends);
    // Contiguous: the pack reads row * kTopK, as _pack_regions does.
    TensorMatcher({B, kTopK}).with_dtype<int32_t>().with_device(device).verify(topk_ids);
    TensorMatcher({B}).with_dtype<int32_t, int64_t>(position_dtype).with_device(device).verify(positions);
    TensorMatcher({B}).with_dtype<int32_t, int64_t>(request_dtype).with_device(device).verify(row_to_req);
    TensorMatcher({Q, C})
        .with_strides({M, 1})
        .with_dtype<int32_t, int64_t>(mapping_dtype)
        .with_device(device)
        .verify(req_to_token);
    TensorMatcher({B, kTopK + 1}).with_dtype<int32_t>().with_device(device).verify(region_ids);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(counts);
    TensorMatcher({B, W}).with_dtype<int64_t>().with_device(device).verify(packed);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(causal_lengths);

    const auto stride = static_cast<uint64_t>(S.unwrap());
    const auto width = static_cast<uint64_t>(N.unwrap());
    const auto packed_width = W.unwrap();
    RuntimeCheck(
        stride * sizeof(float) % INPUT_STRIDE_ALIGNMENT_REQUIREMENT == 0,
        "DeepSelect requires the score row stride to be a multiple of ",
        INPUT_STRIDE_ALIGNMENT_REQUIREMENT,
        " bytes, got ",
        stride);
    static_assert(kTopK * sizeof(int32_t) % OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT == 0);
    RuntimeCheck(width < MAX_VOCAB_SIZE, "DeepSelect width must be below ", MAX_VOCAB_SIZE, ", got ", width);
    RuntimeCheck(
        packed_width == 513 || packed_width == 514 || packed_width == 516,
        "Step5 packed regions require width 513, 514 or 516, got ",
        packed_width);

    const auto batch = static_cast<uint32_t>(B.unwrap());
    if (batch == 0) {
      return;
    }
    const auto dev = device.unwrap();
    int smem_optin = 0;
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev.device_id));

    TopkSelectArgs args{};
    args.batch_size = batch;
    args.vocab_size = static_cast<uint32_t>(width);
    args.topk = kTopK;
    args.input = scores.data_ptr();
    args.output_index = topk_ids.data_ptr();
    args.end_ptr = static_cast<int*>(ends.data_ptr());
    args.stride_input_batch = stride;
    args.stride_output_index_batch = kTopK;
    args.idx_oob_fill_value = -1;
    args.value_oob_fill_value = -INFINITY;
    args.abort_when_nan_found = false;
    args.shared_memory_size_per_sm = static_cast<uint64_t>(smem_optin);
    args.stream = LaunchKernel::resolve_device(dev);

    const step5_region_pack::Params pack{
        .positions = positions.data_ptr(),
        .row_to_req = row_to_req.data_ptr(),
        .req_to_token = req_to_token.data_ptr(),
        .topk_ids = static_cast<const int32_t*>(topk_ids.data_ptr()),
        .region_ids = static_cast<int32_t*>(region_ids.data_ptr()),
        .counts = static_cast<int32_t*>(counts.data_ptr()),
        .packed = static_cast<int64_t*>(packed.data_ptr()),
        .causal_lengths = static_cast<int32_t*>(causal_lengths.data_ptr()),
        .map_stride = M.unwrap(),
        .packed_width = static_cast<int32_t>(packed_width),
        .positions_int64 = position_dtype.is_type<int64_t>(),
        .row_to_req_int64 = request_dtype.is_type<int64_t>(),
        .req_to_token_int64 = mapping_dtype.is_type<int64_t>(),
    };
    step5_region_pack::launch<Config>(args, pack, use_pdl);
  }
};

}  // namespace sglang
