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
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cooperative_groups.h>

// TODO: remove dependency on the custom_all_reduce, move out common utilities
#include "../../distributed/custom_all_reduce.cuh"

namespace sglang {

struct FusionParams {
  uint8_t* input;                             // push: tensor pointer (in place); pull_mc: multicast VA of the tensor
  const uint8_t* residual;                    // may be null (compile-time kHasResidual selects)
  uint8_t* push_ws_mc;                        // push only: multicast VA of the push workspace base
  uint8_t* push_ws_local;                     // push only: local push workspace base (poll side)
  Counter* push_counter;                      // push only: per-block phase counters (local memory)
  Semaphore* pull_semaphores[kMaxWorldSize];  // pull_mc only: per-rank semaphores
  int64_t push_buffer_stride;                 // per-buffer bytes (2 * world_size buffers)
  uint32_t rank;
  uint32_t num_vecs;  // 16B vectors
  // *_norm variants only: RMSNorm epilogue over the first num_norm_rows
  // rows of the [num_vecs / kNormRowVecs, kNormDim] row view
  const uint8_t* norm_weight;
  float norm_eps;
  uint32_t num_norm_rows;
  uint32_t num_push_counters;  // cluster variant only: full counter array size
};

// The K3 latent|shared MoE buffer ([N, 3584] latent then [N, 7168] shared)
// viewed as rows of the latent width: 3N rows of 3584 bf16, of which the
// first N (the latent) get an RMSNorm epilogue. One row = 448 16B vectors =
// one 448-thread block pass, so the row sum-of-squares is a block reduce.
constexpr uint32_t kNormDim = 3584;
constexpr uint32_t kNormRowVecs = kNormDim / 8;                       // 448
constexpr uint32_t kNormWarps = kNormRowVecs / device::kWarpThreads;  // 14

// RMSNorm epilogue for one row vector, identical math to
// elementwise/rmsnorm.cuh: fp32 sum of squares (block reduce), factor =
// rsqrt(sumsq / dim + eps), out = bf16(x * factor * w). One __syncthreads
// per call: after the barrier every warp redundantly reduces the 14 warp
// partials itself (shuffle reduce, warp-local by construction), and the
// caller alternates `smem_half` between consecutive norm rows so the next
// row's partial writes can't race this row's reads (the WAR pair is two
// rows apart and separated by the intervening row's barrier). Every thread
// of the block must call this.
SGL_DEVICE device::AlignedVector<bf16x2_t, 4> apply_row_norm(
    const device::AlignedVector<bf16x2_t, 4>& x,
    const device::AlignedVector<bf16x2_t, 4>& w,
    float eps,
    float (&smem)[32],
    uint32_t smem_half) {
  using namespace device;
  float sum_of_squares = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    const auto [a, b] = cast<fp32x2_t>(x[j]);
    sum_of_squares += a * a + b * b;
  }
  sum_of_squares = warp::reduce_sum(sum_of_squares);
  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane = threadIdx.x % kWarpThreads;
  float* const half = smem + smem_half * 16;
  half[warp_id] = sum_of_squares;  // all lanes write the same value
  __syncthreads();
  const auto local_sum = lane < kNormWarps ? half[lane] : 0.0f;
  const auto norm_factor = math::rsqrt(warp::reduce_sum(local_sum) / kNormDim + eps);
  AlignedVector<bf16x2_t, 4> out;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    const auto [a, b] = cast<fp32x2_t>(x[j]);
    const auto [wa, wb] = cast<fp32x2_t>(w[j]);
    out[j] = cast<bf16x2_t>(fp32x2_t{a * norm_factor * wa, b * norm_factor * wb});
  }
  return out;
}

template <uint32_t kWorldSize, bool kHasResidual, bool kUsePDL>
__global__ __launch_bounds__(1024, 1) void all_reduce_push_res_kernel(const __grid_constant__ FusionParams params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;

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

  // stage 1: multicast-push local data, remapping all-zero bf16x2 pairs
  static_assert(fp_trait<bf16_t>::pos_zero == 0, "the empty marker is all-zero bits");
  constexpr uint32_t kNegZeroPair = 0x8000u;  // {-0.0, +0.0}: sum-neutral, non-zero
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    ld_global_16B(vec, params.input, vid);
    auto& bits = *reinterpret_cast<uint4*>(&vec);
    if (bits.x == 0) bits.x = kNegZeroPair;
    if (bits.y == 0) bits.y = kNegZeroPair;
    if (bits.z == 0) bits.z = kNegZeroPair;
    if (bits.w == 0) bits.w = kNegZeroPair;
    st_multimem_16B(vec, push_ptr, vid);
  }

  // stage 2: poll all slots, reduce (+ residual), write back in place,
  // re-establish the empty markers for the next same-phase round
  vec_t zero_vec;
  zero_vec.fill(bf16x2_t{get_pos_zero<bf16_t>(), get_pos_zero<bf16_t>()});
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec[kWorldSize + kHasResidual];
    if constexpr (kHasResidual) vec[kWorldSize].load(params.residual, vid);
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ld_relaxed_16B(vec[i], poll_ptr + i * stride_bytes, vid);
        // the producer remapped all-zero pairs, so a written u32 is never
        // 0: u32 == 0 <=> the 4B atom still holds the empty marker
        const auto bits = *reinterpret_cast<const uint4*>(&vec[i]);
        has_zero |= bits.x == 0;
        has_zero |= bits.y == 0;
        has_zero |= bits.z == 0;
        has_zero |= bits.w == 0;
      }
      if (!has_zero) break;
    } while (true);
    const auto out_vec = reduce(vec);  // fp32 accumulation over 8(+1) inputs
    st_global_16B(out_vec, params.input, vid);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      st_global_16B(zero_vec, poll_ptr + i * stride_bytes, vid);
    }
  }

  // epilogue: flip this block's phase
  device::PDLTriggerSecondary<kUsePDL>();
  __syncthreads();
  if (tx == 0) params.push_counter[bx].inc(1);  // u32 overflow is safe under mod 2
}

template <uint32_t kWorldSize, bool kHasResidual, bool kUsePDL>
__global__ __launch_bounds__(512) void all_reduce_pull_res_kernel(const __grid_constant__ FusionParams params) {
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

template <typename T2, size_t N, size_t M>
SGL_DEVICE float reduce_sqr(device::AlignedVector<T2, N>& out_vec, device::AlignedVector<T2, N> (&vec)[M]) {
  fp32x2_t acc_vec[N];
#pragma unroll
  for (size_t i = 0; i < M; ++i) {
#pragma unroll
    for (size_t j = 0; j < N; ++j) {
      const auto [x, y] = device::cast<fp32x2_t>(vec[i][j]);
      auto& [acc_x, acc_y] = acc_vec[j];
      acc_x = i == 0 ? x : acc_x + x;
      acc_y = i == 0 ? y : acc_y + y;
    }
  }
  float sum_eq = 0.0f;
#pragma unroll
  for (size_t j = 0; j < N; ++j) {
    sum_eq += acc_vec[j].x * acc_vec[j].x;
    sum_eq += acc_vec[j].y * acc_vec[j].y;
    out_vec[j] = device::cast<T2>(acc_vec[j]);
  }
  return sum_eq;
}

template <uint32_t kWorldSize, uint32_t kClusterSize, bool kUsePDL>
__global__ __launch_bounds__(kNormRowVecs / kClusterSize) __cluster_dims__(kClusterSize, 1, 1)  //
    void all_reduce_push_norm_cluster_kernel(const __grid_constant__ FusionParams params) {
  namespace cg = cooperative_groups;
  using namespace device;
  using vec_t = AlignedVector<bf16x2_t, 4>;
  constexpr uint32_t kBlockSize = kNormRowVecs / kClusterSize;
  constexpr uint32_t kNumWarps = kBlockSize / kWarpThreads;

  static_assert(kBlockSize % kWarpThreads == 0);
  static_assert(kNormRowVecs % kClusterSize == 0);
  static_assert(kNumWarps >= 1);

  const auto tx = threadIdx.x;
  const auto bx = blockIdx.x;
  const auto global_tid = bx * kBlockSize + tx;
  const auto num_vecs = params.num_vecs;
  const auto num_rows = num_vecs / kNormRowVecs;

  const auto row_idx = bx / kClusterSize;
  const auto num_row_clusters = gridDim.x / kClusterSize - 1;  // last one is the bumper
  // stage-1 grid-stride is over the ROW clusters only: the bumper early-returns
  // and never stages, so it must not be counted or its share of vids is dropped
  const auto num_threads = kBlockSize * num_row_clusters * kClusterSize;

  PDLWaitPrimary<kUsePDL>();

  // special case: the bumper cluster flips every remaining counter so the
  // whole array stays globally in phase. Only ONE block does it (threads
  // grid-stride the counters) — each counter must be inc'd exactly once,
  // independent of whether kClusterSize is odd or even.
  if (row_idx == num_row_clusters) {
    if (bx % kClusterSize == 0) {
      for (uint32_t r = num_row_clusters + tx; r < params.num_push_counters; r += kBlockSize) {
        params.push_counter[r].inc(1);
      }
    }
    return PDLTriggerSecondary<kUsePDL>();
  }

  const auto phase = params.push_counter[row_idx].get() & 1;
  const auto r = params.rank;
  const auto stride_bytes = params.push_buffer_stride;
  const auto phase_stride_bytes = (phase * kWorldSize) * stride_bytes;
  const auto push_ptr = params.push_ws_mc + r * stride_bytes + phase_stride_bytes;
  const auto poll_ptr = params.push_ws_local + phase_stride_bytes;

  // stage 1: identical multicast staging (grid-stride)
  static_assert(fp_trait<bf16_t>::pos_zero == 0, "the empty marker is all-zero bits");
  for (auto vid = global_tid; vid < num_vecs; vid += num_threads) {
    vec_t vec;
    ld_global_16B(vec, params.input, vid);
    auto& bits = *reinterpret_cast<uint4*>(&vec);
    if (bits.x == 0) bits.x = fp_trait<bf16_t>::neg_zero;
    if (bits.y == 0) bits.y = fp_trait<bf16_t>::neg_zero;
    if (bits.z == 0) bits.z = fp_trait<bf16_t>::neg_zero;
    if (bits.w == 0) bits.w = fp_trait<bf16_t>::neg_zero;
    st_multimem_16B(vec, push_ptr, vid);
  }

  // stage 2: one row per cluster pass (the bumper cluster owns no rows)
  const auto cluster = cg::this_cluster();
  const auto cluster_rank = bx % kClusterSize;
  vec_t w;
  w.load(params.norm_weight, cluster_rank * kBlockSize + tx);
  vec_t zero_vec;
  zero_vec.fill(bf16x2_t{get_pos_zero<bf16_t>(), get_pos_zero<bf16_t>()});
  __shared__ alignas(8) float smem_raw[2][kClusterSize][kNumWarps];
  uint32_t parity = 0;

  for (auto row = row_idx; row < num_rows; row += num_row_clusters) {
    const auto vid = row * kNormRowVecs + cluster_rank * kBlockSize + tx;
    vec_t vec[kWorldSize];
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ld_relaxed_16B(vec[i], poll_ptr + i * stride_bytes, vid);
        const auto bits = *reinterpret_cast<const uint4*>(&vec[i]);
        has_zero |= bits.x == 0;
        has_zero |= bits.y == 0;
        has_zero |= bits.z == 0;
        has_zero |= bits.w == 0;
      }
      if (!has_zero) break;
    } while (true);

    vec_t out_vec;
    if (row < params.num_norm_rows) {  // cluster-uniform branch
      auto& smem = smem_raw[parity];
      parity ^= 1;
      // push each warp's partial to EVERY peer's slot for this block: lane p
      // (p < kClusterSize) targets peer p, warp w selects the [.][w] slot. So
      // after the barrier every block holds all kClusterSize*kNumWarps
      // partials in its own smem and reduces them locally — the read side
      // never touches remote DSMEM, so no post-read barrier is needed to guard
      // against a peer CTA exiting (parity double-buffers across rows).
      const auto lane = tx % kWarpThreads;
      const auto warp = tx / kWarpThreads;
      const auto warp_sqr = warp::reduce_sum(reduce_sqr(out_vec, vec));
      if (lane < kClusterSize) {
        float* dst = cluster.map_shared_rank(&smem[cluster_rank][warp], lane);
        *dst = warp_sqr;
      }
      cluster.sync();
      // load local
      float total = 0.0f;
#pragma unroll
      for (uint32_t r = 0; r < kClusterSize; ++r) {
        using vec_t = AlignedVector<float, kNumWarps>;
        vec_t remote_value;
        remote_value.load(smem[r]);
#pragma unroll
        for (uint32_t w = 0; w < kNumWarps; ++w) {
          total += remote_value[w];
        }
      }
      const auto norm_factor = math::rsqrt(total / kNormDim + params.norm_eps);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const auto [a, b] = cast<fp32x2_t>(out_vec[j]);
        const auto [wa, wb] = cast<fp32x2_t>(w[j]);
        out_vec[j] = cast<bf16x2_t>(fp32x2_t{a * norm_factor * wa, b * norm_factor * wb});
      }
    } else {
      out_vec = reduce(vec);
    }

    st_global_16B(out_vec, params.input, vid);
#pragma unroll
    for (uint32_t i = 0; i < kWorldSize; ++i) {
      st_global_16B(zero_vec, poll_ptr + i * stride_bytes, vid);
    }
  }

  // epilogue: each row cluster flips its own counter; the bumper cluster
  // flips every remaining one so the whole array stays globally uniform
  PDLTriggerSecondary<kUsePDL>();
  if (cluster_rank == 0 && tx == 0) {
    params.push_counter[row_idx].set(phase ^ 1);
  }
}

template <uint32_t kWorldSize, bool kUsePDL>
__global__ __launch_bounds__(kNormRowVecs)  //
    void all_reduce_pull_norm_kernel(const __grid_constant__ FusionParams params) {
  using vec_t = device::AlignedVector<bf16x2_t, 4>;

  const auto tx = threadIdx.x;
  const auto bx = blockIdx.x;
  const auto r = params.rank;
  const auto num_rows = params.num_vecs / kNormRowVecs;

  // this rank's shard of the row range
  const auto avg_rows = num_rows / kWorldSize;
  const auto rem_rows = num_rows % kWorldSize;
  const auto row_bias = avg_rows * r + min(r, rem_rows);
  const auto my_rows = avg_rows + (r < rem_rows ? 1 : 0);

  // enter barrier (relaxed), same as the plain pull kernel
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

  vec_t wvec;
  wvec.load(params.norm_weight, tx);
  __shared__ float smem[32];
  uint32_t norm_iter = 0;
  for (auto row_local = bx; row_local < my_rows; row_local += gridDim.x) {
    const auto row = row_bias + row_local;
    const auto vid = int64_t(row) * kNormRowVecs + tx;
    vec_t vec;
    ld_multimem_16B(vec, params.input, vid);
    if (row < params.num_norm_rows) {  // block-uniform branch
      vec = apply_row_norm(vec, wvec, params.norm_eps, smem, norm_iter++ & 1);
    }
    st_multimem_16B(vec, params.input, vid);
  }

  // exit barrier (release/acquire)
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

}  // namespace sglang

using namespace sglang;

// ---------------------------------------------------------------------------
// Host entry points
// ---------------------------------------------------------------------------

template <uint32_t kWorldSize, bool kUsePDL>
struct AllReduceFusionKernel {
 private:
  using TensorView = tvm::ffi::TensorView;

  template <bool kHasResidual>
  static constexpr auto res_push_kernel = all_reduce_push_res_kernel<kWorldSize, kHasResidual, kUsePDL>;
  template <bool kHasResidual>
  static constexpr auto res_pull_kernel = all_reduce_pull_res_kernel<kWorldSize, kHasResidual, kUsePDL>;

  static FusionParams
  make_params(const host::distributed::CommunicatorObj& data, TensorView input, std::optional<TensorView> residual) {
    using namespace host;
    SymbolicSize N = {"num_elements"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    if (residual.has_value()) {
      TensorMatcher({N})  //
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(input)
          .verify(residual.value());
    } else {
      TensorMatcher({N})  //
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(input);
    }
    const auto num_elems = N.unwrap();
    CHECK_HOST(data.world_size == kWorldSize);
    CHECK_HOST(num_elems > 0 && num_elems % 8 == 0);
    FusionParams params{};
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

  /// The K3 latent|shared MoE buffer: [num_tokens, 3584] latent then
  /// [num_tokens, 7168] shared = 3 * num_tokens rows of 3584 bf16, of which
  /// the first num_tokens (the latent) get the norm.
  static FusionParams
  make_params_norm(const host::distributed::CommunicatorObj& data, TensorView input, TensorView weight, float eps) {
    using namespace host;
    auto params = make_params(data, input, std::nullopt);
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({kNormDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(weight);
    CHECK_HOST(params.num_vecs % (3 * kNormRowVecs) == 0);
    const auto num_tokens = params.num_vecs / (3 * kNormRowVecs);
    params.norm_weight = static_cast<const uint8_t*>(weight.data_ptr());
    params.norm_eps = static_cast<float>(eps);
    params.num_norm_rows = static_cast<uint32_t>(num_tokens);
    params.num_push_counters = data.num_push_blocks;
    return params;
  }

 public:
  static void push_res(CommunicatorRef ref, TensorView input, std::optional<TensorView> residual, int64_t ws_mc_base) {
    const auto& data = *ref.get();
    auto params = make_params(data, input, residual);
    CHECK_HOST(ws_mc_base != 0) << "push requires a multicast-capable workspace";
    const int64_t nbytes = int64_t(params.num_vecs) * 16;
    CHECK_HOST(nbytes <= data.push_bytes)
        << "input size " << nbytes << " exceeds push workspace size " << data.push_bytes;
    params.push_ws_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(ws_mc_base));
    const auto kernel = residual.has_value() ? res_push_kernel<true> : res_push_kernel<false>;
    host::LaunchKernel(data.num_push_blocks, choose_block_size(params.num_vecs), input.device())
        .enable_pdl(kUsePDL)(kernel, params);
  }

  static void
  pull_res(CommunicatorRef ref, TensorView input, std::optional<TensorView> residual, int64_t input_mc_ptr) {
    const auto& data = *ref.get();
    auto params = make_params(data, input, residual);
    CHECK_HOST(input_mc_ptr != 0) << "pull_res requires the input's multicast address";
    params.input = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(input_mc_ptr));
    // tuned on B200x8 (bs=64..8192 x 7168 bf16): 32 CTAs is near optimal
    constexpr uint32_t kDefaultBlocks = 32;
    constexpr uint32_t kBlockSize = 512;
    const auto num_blocks = std::min<uint32_t>(kDefaultBlocks, data.num_pull_blocks);
    const auto stream = host::LaunchKernel::resolve_device(input.device());
    const auto kernel = residual.has_value() ? res_pull_kernel<true> : res_pull_kernel<false>;
    host::LaunchKernel(num_blocks, kBlockSize, stream).enable_pdl(kUsePDL)(kernel, params);
  }

  static void push_norm(CommunicatorRef ref, TensorView input, TensorView weight, float eps, int64_t ws_mc_base) {
    constexpr auto kClusterSize = 7;
    const auto& data = *ref.get();
    auto params = make_params_norm(data, input, weight, eps);
    CHECK_HOST(ws_mc_base != 0) << "push requires a multicast-capable workspace";
    const int64_t nbytes = int64_t(params.num_vecs) * 16;
    CHECK_HOST(nbytes <= data.push_bytes)
        << "input size " << nbytes << " exceeds push workspace size " << data.push_bytes;
    params.push_ws_mc = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(ws_mc_base));
    const auto num_rows = params.num_vecs / kNormRowVecs;
    constexpr uint32_t kMaxClusters = 96;
    const auto num_row_clusters = std::max<uint32_t>(std::min(num_rows, kMaxClusters), 1);
    CHECK_HOST(num_row_clusters < data.num_push_blocks);
    host::LaunchKernel((num_row_clusters + 1) * kClusterSize, kNormRowVecs / kClusterSize, input.device())
        .enable_pdl(kUsePDL)(all_reduce_push_norm_cluster_kernel<kWorldSize, kClusterSize, kUsePDL>, params);
  }

  static void pull_norm(CommunicatorRef ref, TensorView input, TensorView weight, float eps, int64_t input_mc_ptr) {
    constexpr uint32_t kDefaultBlocks = 64;
    const auto& data = *ref.get();
    auto params = make_params_norm(data, input, weight, eps);
    CHECK_HOST(input_mc_ptr != 0) << "pull_norm requires the input's multicast address";
    params.input = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(input_mc_ptr));
    const auto num_blocks = std::min<uint32_t>(kDefaultBlocks, data.num_pull_blocks);
    host::LaunchKernel(num_blocks, kNormRowVecs, input.device())
        .enable_pdl(kUsePDL)(all_reduce_pull_norm_kernel<kWorldSize, kUsePDL>, params);
  }
};
