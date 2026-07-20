// K3 MNNVL fused all-reduce (bf16-only), two zero-copy algorithms:
//
//   - push (1shot): lamport-style push, but the data staging is a SINGLE
//     multicast store into slot `rank` of every peer's push workspace
//     (replacing the 7 unicast stores of the generic kernel), followed by a
//     local zero-marker polling reduce. Input is read in place and the
//     result is written back in place — no staging copies. Works for any
//     input tensor; reuses the CustomAllReduceV2 push workspace + counter.
//
//   - pull_mc (2shot): NVLS reduce-scatter + broadcast directly ON the input
//     tensor, which therefore MUST live in (multicast-bound) symmetric
//     memory: each rank `multimem.ld_reduce`s its shard from the input's
//     multicast address and `multimem.st`s the result back — in-place, zero
//     copies. Reuses the CustomAllReduceV2 pull semaphores for the
//     enter/exit barriers.
//
// Both kernels optionally fuse a residual add (`out = allreduce(x) + r`).
// Contract: the residual must be identical on every rank (it is a fully
// reduced tensor, e.g. the attn-res prefix sum) or absent.
//
// TODO: remove dependency on the custom_all_reduce, move out common utilities
#include "../../distributed/custom_all_reduce.cuh"

namespace {

namespace ptx {

// The push protocol needs 8-byte atoms: the "slot empty" marker is a fully
// zero 8-byte group, so producers store and consumers load in units whose
// atomicity matches the marker granularity (v2.b64 = 2 independent 8B atoms).

template <typename V>
SGL_DEVICE void ld_global_16B_atom8B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  addr = static_cast<const uint8_t*>(addr) + vec_offset * sizeof(V);
  ulong2 val;
  asm volatile("ld.global.v2.b64 {%0, %1}, [%2];" : "=l"(val.x), "=l"(val.y) : "l"(addr));
  x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
SGL_DEVICE void ld_relaxed_16B_atom8B(V& x, const void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  addr = static_cast<const uint8_t*>(addr) + vec_offset * sizeof(V);
  ulong2 val;
  asm volatile("ld.relaxed.sys.global.v2.b64 {%0, %1}, [%2];" : "=l"(val.x), "=l"(val.y) : "l"(addr) : "memory");
  x = *reinterpret_cast<const V*>(&val);
}

template <typename V>
SGL_DEVICE void st_global_16B_atom8B(const V& x, void* addr, int64_t vec_offset) {
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const auto val = *reinterpret_cast<const ulong2*>(&x);
  addr = static_cast<uint8_t*>(addr) + vec_offset * sizeof(V);
  asm volatile("st.global.v2.b64 [%2], {%0, %1};" : : "l"(val.x), "l"(val.y), "l"(addr));
}

template <typename V>
SGL_DEVICE void st_multimem_16B_atom8B(const V& x, void* mc_addr, int64_t vec_offset) {
#if SGL_ARCH_HOPPER_OR_GREATER
  static_assert(alignof(V) == 16 && sizeof(V) == 16);
  const auto val = *reinterpret_cast<const ulong2*>(&x);
  const auto addr = static_cast<uint8_t*>(mc_addr) + vec_offset * sizeof(V);
  // multimem.st rejects vector qualifiers on .b64; two scalar stores keep the
  // 8-byte atomicity the zero-marker protocol relies on.
  asm volatile("multimem.st.relaxed.sys.global.b64 [%1], %0;" : : "l"(val.x), "l"(addr) : "memory");
  asm volatile("multimem.st.relaxed.sys.global.b64 [%1], %0;" : : "l"(val.y), "l"(addr + 8) : "memory");
#else
  assert(false && "multimem store is only supported on Hopper or later architecture");
#endif
}

}  // namespace ptx

struct ArFusionParams {
  uint8_t* input;                             // push: tensor pointer (in place); pull_mc: multicast VA of the tensor
  const uint8_t* residual;                    // may be null (compile-time kHasResidual selects)
  uint8_t* push_ws_mc;                        // push only: multicast VA of the push workspace base
  uint8_t* push_ws_local;                     // push only: local push workspace base (poll side)
  Counter* push_counter;                      // push only: per-block phase counters (local memory)
  Semaphore* pull_semaphores[kMaxWorldSize];  // pull_mc only: per-rank semaphores
  int64_t push_buffer_stride;                 // per-buffer bytes (2 * world_size buffers)
  uint32_t rank;
  uint32_t num_vecs;  // 16B vectors
};

// ---------------------------------------------------------------------------
// push: multicast data staging + local zero-marker polling reduce (bf16)
// ---------------------------------------------------------------------------

template <uint32_t kWorldSize, bool kHasResidual, bool kUsePDL>
__global__ __launch_bounds__(1024, 1)  //
    void ar_fusion_push_kernel(const __grid_constant__ ArFusionParams params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;
  // one -0.0 bf16 in an otherwise +0.0 8-byte group: numerically a no-op for
  // the sum, but the group is no longer the all-zero "slot empty" marker
  constexpr uint64_t kNegZeroMark = 0x8000ull;

  const auto tx = threadIdx.x;
  const auto bx = blockIdx.x;
  const auto global_tid = bx * blockDim.x + tx;
  const auto num_threads = blockDim.x * gridDim.x;
  const auto num_vecs = params.num_vecs;

  // prologue: the previous phase flip (counter inc) must be visible
  device::PDLWaitPrimary<kUsePDL>();
  const auto phase = params.push_counter[bx].get() & 1;
  const auto r = params.rank;
  const auto stride_bytes = params.push_buffer_stride;
  const auto phase_stride_bytes = (phase * kWorldSize) * stride_bytes;
  // one multicast store lands this rank's data in slot r of EVERY peer
  const auto push_ptr = params.push_ws_mc + r * stride_bytes + phase_stride_bytes;
  const auto poll_ptr = params.push_ws_local + phase_stride_bytes;

  // stage 1: multicast-push local data, clearing all-zero 8B groups
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    ulong2 vec;
    ptx::ld_global_16B_atom8B(vec, params.input, vid);
    static_assert(fp_trait<bf16_t>::pos_zero == 0);
    if (vec.x == 0) vec.x = kNegZeroMark;
    if (vec.y == 0) vec.y = kNegZeroMark;
    ptx::st_multimem_16B_atom8B(vec, push_ptr, vid);
  }

  // stage 2: poll all slots, reduce (+ residual), write back in place,
  // re-establish the empty markers for the next same-phase round
  vec_t zero_vec;
  *reinterpret_cast<ulong2*>(&zero_vec) = ulong2{0, 0};
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec[kWorldSize + kHasResidual];
    if constexpr (kHasResidual) vec[kWorldSize].load(params.residual, vid);
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ptx::ld_relaxed_16B_atom8B(vec[i], poll_ptr + i * stride_bytes, vid);
        const auto bits = *reinterpret_cast<const ulong2*>(&vec[i]);
        has_zero |= bits.x == 0;
        has_zero |= bits.y == 0;
      }
      if (!has_zero) break;
    } while (true);
    const auto out_vec = reduce(vec);  // fp32 accumulation over 8(+1) inputs
    ptx::st_global_16B_atom8B(out_vec, params.input, vid);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      ptx::st_global_16B_atom8B(zero_vec, poll_ptr + i * stride_bytes, vid);
    }
  }

  // epilogue: flip this block's phase
  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  if (tx == 0) params.push_counter[bx].inc(1);  // u32 overflow is safe under mod 2
}

// ---------------------------------------------------------------------------
// pull_mc: NVLS reduce-scatter + broadcast, in place on the (symmetric
// memory) input tensor (bf16)
// ---------------------------------------------------------------------------

template <uint32_t kWorldSize, bool kHasResidual, bool kUsePDL>
__global__ __launch_bounds__(1024, 1)  //
    void ar_fusion_pull_kernel(const __grid_constant__ ArFusionParams params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;
  using SumOp = device::ReductionTrait<device::ReductionOp::SUM, bf16x2_t>;

  const auto tx = threadIdx.x;
  const auto bx = blockIdx.x;
  const auto global_tid = bx * blockDim.x + tx;
  const auto num_threads = blockDim.x * gridDim.x;
  const auto r = params.rank;

  // this rank's shard of the 16B-vector range
  const auto avg_vecs = params.num_vecs / kWorldSize;
  const auto rem_vecs = params.num_vecs % kWorldSize;
  const auto vec_bias = avg_vecs * r + min(r, rem_vecs);
  const auto num_vecs = avg_vecs + (r < rem_vecs ? 1 : 0);
  const auto mc_shard = params.input + int64_t(vec_bias) * 16;
  [[maybe_unused]] const auto res_shard = kHasResidual ? params.residual + int64_t(vec_bias) * 16 : nullptr;

  // enter barrier (relaxed): all ranks' producers have finished writing the
  // input (each rank signals only after its own PDL wait). Mirrors
  // AllReducePullImpl::sync_enter_pull<false>.
  uint32_t counter_val = 0;
  if (tx < kWorldSize) {
    device::PDLWaitPrimary<kUsePDL>();
    const auto semaphore = &params.pull_semaphores[tx][bx];
    const auto current = tx == r ? semaphore->counter_ptr()->inc(2 * kWorldSize) : 0;
    counter_val = current + kWorldSize;
    semaphore->put_relaxed();
    if (tx == r) {
      while (semaphore->get_relaxed() - current < kWorldSize)
        ;
    }
  }
  __syncthreads();

  // reduce-scatter via multimem load, fuse residual, broadcast via multimem
  // store — in place on the symmetric input
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    ld_multimem_16B(vec, mc_shard, vid);
    if constexpr (kHasResidual) {
      vec_t res_vec;
      res_vec.load(res_shard, vid);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        vec[j] = SumOp::reduce(vec[j], res_vec[j]);
      }
    }
    st_multimem_16B(vec, mc_shard, vid);
  }

  // exit barrier (release/acquire): every peer's broadcast into MY buffer is
  // visible before my next kernel reads it. Mirrors sync_exit_pull<true>.
  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  if (tx < kWorldSize) {
    const auto semaphore = &params.pull_semaphores[tx][bx];
    semaphore->put_release();
    if (tx == r) {
      while (semaphore->get_acquire() - counter_val < kWorldSize)
        ;
    }
  }
}

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------

template <uint32_t kWorldSize, bool kUsePDL>
struct ArFusionKernel {
 private:
  using TensorView = tvm::ffi::TensorView;

  static ArFusionParams
  make_params(const host::distributed::CommunicatorObj& data, TensorView input, std::optional<TensorView> residual) {
    using namespace host;
    SymbolicSize N = {"num_elements"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    if (residual.has_value()) {
      TensorMatcher({N}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(input).verify(residual.value());
    } else {
      TensorMatcher({N}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(input);
    }
    const auto num_elems = N.unwrap();
    CHECK_HOST(data.world_size == kWorldSize) << "world size mismatch: " << data.world_size << " != " << kWorldSize;
    CHECK_HOST(num_elems > 0 && num_elems % 8 == 0)
        << "num_elements must be a positive multiple of 8 (16B vectors), got " << num_elems;
    ArFusionParams params{};
    params.input = static_cast<uint8_t*>(input.data_ptr());
    params.residual = residual.has_value() ? static_cast<const uint8_t*>(residual.value().data_ptr()) : nullptr;
    params.push_ws_mc = nullptr;
    params.push_ws_local = data.push_workspaces[data.rank];
    params.push_counter = data.push_counter;
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      params.pull_semaphores[i] = data.pull_semaphores[i];
    }
    params.push_buffer_stride = data.push_bytes;
    params.rank = data.rank;
    params.num_vecs = static_cast<uint32_t>(num_elems / 8);
    return params;
  }

  template <bool kHasResidual, typename Stream>
  static void launch_push(const ArFusionParams& params, uint32_t num_blocks, const Stream& stream) {
    host::LaunchKernel(num_blocks, choose_block_size(params.num_vecs), stream)
        .enable_pdl(kUsePDL)(ar_fusion_push_kernel<kWorldSize, kHasResidual, kUsePDL>, params);
  }

  template <bool kHasResidual, typename Stream>
  static void
  launch_pull(const ArFusionParams& params, uint32_t num_blocks, uint32_t block_size, const Stream& stream) {
    host::LaunchKernel(num_blocks, block_size, stream)
        .enable_pdl(kUsePDL)(ar_fusion_pull_kernel<kWorldSize, kHasResidual, kUsePDL>, params);
  }

 public:
  /// In-place fused all-reduce (+residual) via multicast push. `ws_mc_base`
  /// is the multicast VA of the workspace slab base (the push region starts
  /// at offset 0). Reuses the CustomAllReduceV2 push workspace: the input
  /// may be any contiguous bf16 CUDA tensor.
  static void push(CommunicatorRef ref, TensorView input, std::optional<TensorView> residual, int64_t ws_mc_base) {
    const auto& data = *ref.get();
    auto params = make_params(data, input, residual);
    CHECK_HOST(ws_mc_base != 0) << "push requires a multicast-capable workspace";
    const int64_t nbytes = int64_t(params.num_vecs) * 16;
    CHECK_HOST(nbytes <= data.push_bytes)
        << "input size " << nbytes << " exceeds push workspace size " << data.push_bytes;
    params.push_ws_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(ws_mc_base));
    const auto stream = host::LaunchKernel::resolve_device(input.device());
    // the grid is bound to the counter array and must stay constant
    if (residual.has_value()) {
      launch_push<true>(params, data.num_push_blocks, stream);
    } else {
      launch_push<false>(params, data.num_push_blocks, stream);
    }
  }

  /// In-place fused all-reduce (+residual) via NVLS on the input itself.
  /// `input_mc_ptr` is the multicast VA of the input tensor, which MUST be
  /// allocated from (multicast-bound) symmetric memory; the residual must be
  /// identical on every rank. `num_blocks == 0` picks the tuned default.
  static void pull_mc(
      CommunicatorRef ref,
      TensorView input,
      std::optional<TensorView> residual,
      int64_t input_mc_ptr,
      int64_t num_blocks) {
    const auto& data = *ref.get();
    auto params = make_params(data, input, residual);
    CHECK_HOST(input_mc_ptr != 0) << "pull_mc requires the input's multicast address";
    params.input = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(input_mc_ptr));
    // tuned on B200x8 (bs=64..8192 x 7168 bf16): 32 CTAs is optimal or
    // within noise at every size (16 loses ~10% at the large end)
    constexpr uint32_t kDefaultBlocks = 32;
    constexpr uint32_t kBlockSize = 512;
    const auto grid = std::min<uint32_t>(
        num_blocks > 0 ? static_cast<uint32_t>(num_blocks) : kDefaultBlocks,
        data.num_pull_blocks);  // bound by the semaphore array
    const auto stream = host::LaunchKernel::resolve_device(input.device());
    if (residual.has_value()) {
      launch_pull<true>(params, grid, kBlockSize, stream);
    } else {
      launch_pull<false>(params, grid, kBlockSize, stream);
    }
  }
};

}  // namespace
