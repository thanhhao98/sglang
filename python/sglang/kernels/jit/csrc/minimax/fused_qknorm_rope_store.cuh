// MiniMax-M3.1's grouped Gemma q/k/index norm + RoPE with the FP8 q epilogue and its
// NVFP4 K/V/index-K page store, in one launch: fused_gemma_qknorm_rope_kernel's warps,
// unchanged, after which the warps of the main-K and index-K heads store the heads
// they normed (and V beside main K) with nvfp4_store_kv_index_kernel's quantizer.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "fused_gemma_qknorm_rope.cuh"
#include "nvfp4_store_kv_index.cuh"
#include <cstdint>

namespace sglang {

/// The two stock kernels' parameters, unchanged, and the norm jobs (a token's head
/// index in group order) whose heads the store reads back.
struct QKNormRopeStoreParams {
  FusedGemmaQKNormParams norm;
  Nvfp4StoreParams store;
  uint32_t main_k_job;   // the first main-K head's job; the K heads are consecutive jobs
  uint32_t index_k_job;  // the index-K head's job
};

template <typename PosT, int64_t kHeadDim, int64_t kRopeDim, bool kUsePDL>
struct QKNormRopeStoreTrait {
  using Norm = FusedTrait<PosT, kHeadDim, kRopeDim, kUsePDL>;
  using Store = Nvfp4StoreTrait<kHeadDim, kUsePDL>;
  // One lane per 16-element NVFP4 block: a warp stores its K head and the V head
  // beside it.
  static constexpr uint32_t kBlocksPerHead = Store::kWorkerSize;
  static_assert(2 * kBlocksPerHead <= Norm::kWorkerSize, "a warp stores a K and a V head");

  /// \brief One block of `Nvfp4StoreTrait::forward`, after its PDL wait: the same
  /// source, slot and buffer addressing and the same `quantize_block`.
  ///
  /// The block's 16 values are read with volatile loads: main K and index K were
  /// written by this kernel (other lanes of this warp), and the read-only data path
  /// does not see a kernel's own writes.
  template <typename T>
  SGL_DEVICE static void
  store_block(const Nvfp4StoreParams& params, uint32_t branch, uint32_t head_id, uint32_t token_id, uint32_t block_id) {
    using namespace device;
    const auto src = static_cast<const bf16_t*>(params.in_ptrs[branch]) + token_id * params.in_token_stride[branch] +
                     head_id * params.in_head_stride[branch] + block_id * kNvfp4BlockSize;
    const auto loc = static_cast<int64_t>(static_cast<const T*>(params.indices)[token_id]);
    const int64_t page = loc / params.page_size;
    const int64_t row = loc % params.page_size;

    const auto& pay_s = params.payload_strides[branch];
    const auto& sc_s = params.scale_strides[branch];
    const auto payload = pointer::offset(
        params.payload_ptrs[branch], page * pay_s[0], head_id * pay_s[1], row * pay_s[2], block_id * kNvfp4BlockBytes);
    const auto scale =
        pointer::offset(params.scale_ptrs[branch], page * sc_s[0], head_id * sc_s[1], row * sc_s[2], block_id);

    const auto bits = reinterpret_cast<const volatile uint16_t*>(src);
    bf16_t values[kNvfp4BlockSize];
#pragma unroll
    for (int i = 0; i < kNvfp4BlockSize; ++i) {
      values[i] = __ushort_as_bfloat16(bits[i]);
    }
    uint8_t scale_byte;
    const uint64_t packed = minimax_nvfp4::quantize_block(values, *params.global_scale, &scale_byte);
    *static_cast<uint64_t*>(payload) = packed;
    *static_cast<uint8_t*>(scale) = scale_byte;
  }

  template <typename T>
  SGL_DEVICE static void forward(const QKNormRopeStoreParams& params) {
    // The q/k norm kernel's warp, unchanged: its norm weight and position reads, PDL
    // wait, norm, PDL trigger, then the RoPE'd head written back in place (and the FP8
    // copy of a q group's head). So, like the norm kernel, this kernel triggers its
    // dependents before its writes, page stores included. Every read below follows
    // its wait.
    Norm::forward(params.norm);

    const auto& norm = params.norm;
    const uint32_t work_id = (blockIdx.x * blockDim.x + threadIdx.x) / Norm::kWorkerSize;
    if (work_id >= norm.num_tokens * norm.total_heads) return;
    const uint32_t token_id = work_id / norm.total_heads;
    const uint32_t job = work_id % norm.total_heads;
    const uint32_t kv_head = job - params.main_k_job;  // wraps for the jobs before main K
    const bool main_k = kv_head < params.store.num_kv_heads;
    if (!main_k && job != params.index_k_job) return;

    // Every lane's write of the normed head before any lane reads its block back.
    __syncwarp();
    const uint32_t lane_id = threadIdx.x % Norm::kWorkerSize;
    const uint32_t block_id = lane_id % kBlocksPerHead;
    if (main_k) {
      if (lane_id >= 2 * kBlocksPerHead) return;
      const uint32_t branch = lane_id < kBlocksPerHead ? kMainK : kMainV;
      store_block<T>(params.store, branch, kv_head, token_id, block_id);
    } else {
      if (lane_id >= kBlocksPerHead) return;
      store_block<T>(params.store, kIndexK, 0, token_id, block_id);
    }
  }
};

template <typename Trait, typename T>
__global__ void fused_gemma_qknorm_rope_store_kernel(const __grid_constant__ QKNormRopeStoreParams params) {
  Trait::template forward<T>(params);
}

/// \brief `fused_gemma_qknorm_rope_fp8` followed by `nvfp4_store_kv_index`, in one
/// launch, writing what the two write.
///
/// The first twenty arguments are `fused_gemma_qknorm_rope_fp8`'s and the last twelve
/// `nvfp4_store_kv_index`'s, with their checks. `k` and `idx_k` must be the heads of
/// groups `k_group` and `index_k_group` inside `qkv` (the warp that norms a head
/// stores it), and `v` must lie inside `qkv`'s rows outside every group.
///
/// \tparam PosT Position dtype (int32 or int64).
/// \tparam HEAD_DIM Head dimension of every group and of the caches (128).
/// \tparam ROTARY_DIM Rotary dimension (64).
/// \tparam kUsePDL Whether to launch with PDL (wait and trigger as the q/k norm kernel).
/// \param k_group Group whose heads are `k`; `index_k_group`: the group whose head is `idx_k`.
template <typename PosT, int64_t HEAD_DIM, int64_t ROTARY_DIM, bool kUsePDL>
void fused_gemma_qknorm_rope_store_fp8(
    tvm::ffi::TensorView qkv,
    tvm::ffi::TensorView w0,
    tvm::ffi::TensorView w1,
    tvm::ffi::TensorView w2,
    tvm::ffi::TensorView w3,
    tvm::ffi::TensorView cos_sin_cache,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView fp8_a,
    tvm::ffi::TensorView fp8_b,
    int64_t off0,
    int64_t cnt0,
    int64_t off1,
    int64_t cnt1,
    int64_t off2,
    int64_t cnt2,
    int64_t off3,
    int64_t cnt3,
    int64_t num_groups,
    int64_t group_a,
    int64_t group_b,
    double eps,
    int64_t k_group,
    int64_t index_k_group,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView idx_k,
    tvm::ffi::TensorView k_payload,
    tvm::ffi::TensorView k_scale,
    tvm::ffi::TensorView v_payload,
    tvm::ffi::TensorView v_scale,
    tvm::ffi::TensorView idx_k_payload,
    tvm::ffi::TensorView idx_k_scale,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView global_scale,
    int64_t page_size) {
  using namespace host;
  auto N = SymbolicSize{"num_tokens"};
  auto H = SymbolicSize{"kv_heads"};
  auto P = SymbolicSize{"pages"};
  auto R = SymbolicSize{"page_size"};
  auto indice_dtype = SymbolicDType{};
  auto device = SymbolicDevice{};
  constexpr auto D = HEAD_DIM;
  constexpr auto Rd = ROTARY_DIM;
  constexpr int64_t kPayloadBytes = HEAD_DIM / 2;
  constexpr int64_t kScaleBytes = HEAD_DIM / kNvfp4BlockSize;
  device.set_options<kDLCUDA>();

  // fused_gemma_qknorm_rope's checks
  TensorMatcher({N, -1}).with_dtype<bf16_t>().with_device(device).verify(qkv);
  TensorMatcher({D}).with_dtype<bf16_t>().with_device(device).verify(w0).verify(w1).verify(w2).verify(w3);
  TensorMatcher({-1, Rd}).with_dtype<fp32_t>().with_device(device).verify(cos_sin_cache);
  TensorMatcher({N}).with_dtype<PosT>().with_device(device).verify(positions);
  RuntimeCheck(num_groups >= 1 && num_groups <= kMaxGroups);

  // nvfp4_store_kv_index's checks, on the same rows and device
  TensorMatcher({N, H, D}).with_strides({-1, -1, 1}).with_dtype<bf16_t>().with_device(device).verify(k).verify(v);
  TensorMatcher({N, D}).with_strides({-1, 1}).with_dtype<bf16_t>().with_device(device).verify(idx_k);
  TensorMatcher({P, H, R, kPayloadBytes})  //
      .with_strides({-1, -1, kPayloadBytes, 1})
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(k_payload)
      .verify(v_payload);
  TensorMatcher({P, H, R, kScaleBytes})  //
      .with_strides({-1, -1, kScaleBytes, 1})
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(k_scale)
      .verify(v_scale);
  TensorMatcher({P, R, kPayloadBytes})  //
      .with_strides({-1, kPayloadBytes, 1})
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(idx_k_payload);
  TensorMatcher({P, R, kScaleBytes})  //
      .with_strides({-1, kScaleBytes, 1})
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(idx_k_scale);
  for (const auto& payload : {k_payload, v_payload}) {
    CHECK_HOST(payload.stride(0) % kNvfp4BlockBytes == 0 && payload.stride(1) % kNvfp4BlockBytes == 0)
        << "main payload page and head strides must be multiples of " << kNvfp4BlockBytes;
  }
  CHECK_HOST(idx_k_payload.stride(0) % kNvfp4BlockBytes == 0)
      << "index payload page stride must be a multiple of " << kNvfp4BlockBytes;
  for (const auto& payload : {k_payload, v_payload, idx_k_payload}) {
    CHECK_HOST(reinterpret_cast<uintptr_t>(payload.data_ptr()) % kNvfp4BlockBytes == 0)
        << "payload base must be " << kNvfp4BlockBytes << "-byte aligned";
  }
  TensorMatcher({N}).with_dtype<int32_t, int64_t>(indice_dtype).with_device(device).verify(indices);
  TensorMatcher({1}).with_dtype<float>().with_device(device).verify(global_scale);
  CHECK_HOST(R.unwrap() == page_size) << "page size " << page_size << " does not match the cache row count "
                                      << R.unwrap();

  const tvm::ffi::TensorView weights[kMaxGroups] = {w0, w1, w2, w3};
  const int64_t offsets[kMaxGroups] = {off0, off1, off2, off3};
  const int64_t counts[kMaxGroups] = {cnt0, cnt1, cnt2, cnt3};
  const int64_t num_kv_heads = H.unwrap();

  // The warp that norms a head stores it: the stored K and index-K heads must be those
  // groups' heads in place in `qkv`, and V must be outside every group (no warp writes it).
  CHECK_HOST(
      k_group >= 0 && k_group < num_groups && index_k_group >= 0 && index_k_group < num_groups &&
      k_group != index_k_group)
      << "the K group " << k_group << " and the index-K group " << index_k_group << " must be two of " << num_groups
      << " groups";
  for (int64_t i = 0; i < num_groups; ++i) {
    for (int64_t j = i + 1; j < num_groups; ++j) {
      CHECK_HOST(offsets[i] + counts[i] <= offsets[j] || offsets[j] + counts[j] <= offsets[i])
          << "groups " << i << " and " << j << " overlap";
    }
  }
  const auto byte_offset = [&qkv](const tvm::ffi::TensorView& view) {
    return static_cast<int64_t>(
        reinterpret_cast<uintptr_t>(view.data_ptr()) - reinterpret_cast<uintptr_t>(qkv.data_ptr()));
  };
  // A size-1 dimension's stride is never used: its only index is 0.
  const bool one_token = N.unwrap() == 1;
  const auto rows_of_qkv = [&](const tvm::ffi::TensorView& view) {
    return one_token || view.stride(0) == qkv.stride(0);
  };
  const int64_t head_bytes = D * static_cast<int64_t>(sizeof(bf16_t));
  const int64_t row_bytes = qkv.size(1) * static_cast<int64_t>(sizeof(bf16_t));
  CHECK_HOST(
      counts[k_group] == num_kv_heads && rows_of_qkv(k) && (num_kv_heads == 1 || k.stride(1) == D) &&
      byte_offset(k) == offsets[k_group] * head_bytes)
      << "k must be the heads of group " << k_group << " in qkv";
  CHECK_HOST(
      counts[index_k_group] == 1 && rows_of_qkv(idx_k) && byte_offset(idx_k) == offsets[index_k_group] * head_bytes)
      << "idx_k must be the head of group " << index_k_group << " in qkv";
  const int64_t v_bytes = byte_offset(v);
  CHECK_HOST(
      rows_of_qkv(v) && (num_kv_heads == 1 || v.stride(1) == D) && v_bytes >= 0 && v_bytes % head_bytes == 0 &&
      v_bytes + num_kv_heads * head_bytes <= row_bytes)
      << "v must be whole heads inside qkv's rows";
  const int64_t v_head = v_bytes / head_bytes;
  for (int64_t i = 0; i < num_groups; ++i) {
    CHECK_HOST(v_head + num_kv_heads <= offsets[i] || offsets[i] + counts[i] <= v_head)
        << "v overlaps group " << i << ", which the kernel writes";
  }

  auto params = QKNormRopeStoreParams{};

  // fused_gemma_qknorm_rope's parameters
  auto& norm = params.norm;
  norm.qkv = static_cast<bf16_t*>(qkv.data_ptr());
  norm.cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr());
  norm.positions = positions.data_ptr();
  norm.num_tokens = static_cast<uint32_t>(N.unwrap());
  norm.num_groups = static_cast<uint32_t>(num_groups);
  norm.token_stride = static_cast<int64_t>(qkv.stride(0));
  norm.eps = static_cast<float>(eps);
  uint32_t total_heads = 0;
  for (int64_t i = 0; i < kMaxGroups; ++i) {
    const auto cnt = (i < num_groups) ? counts[i] : 0;
    norm.weight[i] = static_cast<const bf16_t*>(weights[i].data_ptr());
    norm.group_offset[i] = static_cast<uint32_t>(i < num_groups ? offsets[i] : 0);
    norm.group_count[i] = static_cast<uint32_t>(cnt);
    if (i == k_group) params.main_k_job = total_heads;
    if (i == index_k_group) params.index_k_job = total_heads;
    total_heads += static_cast<uint32_t>(cnt);
  }
  norm.total_heads = total_heads;
  auto bind_fp8 = [&](const tvm::ffi::TensorView& out, int64_t group_id) {
    CHECK_HOST(group_id >= 0 && group_id < num_groups)
        << "fp8 output names group " << group_id << " but only " << num_groups << " are active";
    auto G = SymbolicSize{"fp8_heads"};
    TensorMatcher({N, G, D})  //
        .with_strides({-1, D, 1})
        .with_dtype<fp8_e4m3_t>()
        .with_device(device)
        .verify(out);
    CHECK_HOST(G.unwrap() == counts[group_id])
        << "fp8 output has " << G.unwrap() << " heads, group " << group_id << " has " << counts[group_id];
    norm.fp8_out[group_id] = static_cast<fp8_e4m3_t*>(out.data_ptr());
    norm.fp8_token_stride[group_id] = out.stride(0);
  };
  bind_fp8(fp8_a, group_a);
  bind_fp8(fp8_b, group_b);

  // nvfp4_store_kv_index's parameters
  auto& store = params.store;
  store.in_ptrs[kMainK] = k.data_ptr();
  store.in_ptrs[kMainV] = v.data_ptr();
  store.in_ptrs[kIndexK] = idx_k.data_ptr();
  store.in_token_stride[kMainK] = k.stride(0);
  store.in_token_stride[kMainV] = v.stride(0);
  store.in_token_stride[kIndexK] = idx_k.stride(0);
  store.in_head_stride[kMainK] = k.stride(1);
  store.in_head_stride[kMainV] = v.stride(1);
  store.in_head_stride[kIndexK] = 0;
  store.payload_ptrs[kMainK] = k_payload.data_ptr();
  store.payload_ptrs[kMainV] = v_payload.data_ptr();
  store.payload_ptrs[kIndexK] = idx_k_payload.data_ptr();
  store.scale_ptrs[kMainK] = k_scale.data_ptr();
  store.scale_ptrs[kMainV] = v_scale.data_ptr();
  store.scale_ptrs[kIndexK] = idx_k_scale.data_ptr();
  const tvm::ffi::TensorView main_payload[2] = {k_payload, v_payload};
  const tvm::ffi::TensorView main_scale[2] = {k_scale, v_scale};
  for (int b = 0; b < 2; ++b) {
    store.payload_strides[b][0] = main_payload[b].stride(0);
    store.payload_strides[b][1] = main_payload[b].stride(1);
    store.payload_strides[b][2] = main_payload[b].stride(2);
    store.scale_strides[b][0] = main_scale[b].stride(0);
    store.scale_strides[b][1] = main_scale[b].stride(1);
    store.scale_strides[b][2] = main_scale[b].stride(2);
  }
  store.payload_strides[kIndexK][0] = idx_k_payload.stride(0);
  store.payload_strides[kIndexK][1] = 0;
  store.payload_strides[kIndexK][2] = idx_k_payload.stride(1);
  store.scale_strides[kIndexK][0] = idx_k_scale.stride(0);
  store.scale_strides[kIndexK][1] = 0;
  store.scale_strides[kIndexK][2] = idx_k_scale.stride(1);
  store.indices = indices.data_ptr();
  store.global_scale = static_cast<const float*>(global_scale.data_ptr());
  store.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
  store.num_total_heads = static_cast<uint32_t>(2 * num_kv_heads + 1);
  store.page_size = static_cast<uint32_t>(page_size);
  store.total_jobs = static_cast<uint32_t>(N.unwrap() * (2 * num_kv_heads + 1));

  // fused_gemma_qknorm_rope's grid: one warp per (token, head) of the groups
  using Trait = QKNormRopeStoreTrait<PosT, HEAD_DIM, ROTARY_DIM, kUsePDL>;
  const auto needed_threads = static_cast<int64_t>(norm.num_tokens) * total_heads * device::kWarpThreads;
  RuntimeCheck(needed_threads < std::numeric_limits<uint32_t>::max());
  if (needed_threads == 0) return;
  const uint32_t block_size = 256u;
  const uint32_t num_blocks = div_ceil(static_cast<uint32_t>(needed_threads), block_size);
  const auto kernel = indice_dtype.is_type<int32_t>()  //
                          ? fused_gemma_qknorm_rope_store_kernel<Trait, int32_t>
                          : fused_gemma_qknorm_rope_store_kernel<Trait, int64_t>;
  LaunchKernel(num_blocks, block_size, device.unwrap())  //
      .enable_pdl(kUsePDL)(kernel, params);
}

}  // namespace sglang
