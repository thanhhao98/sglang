// AttnRes score kernel: per-row dot(v, cw) / rms(v) scoring.
//
// Grid: (T, NVB+1)  —  one CTA per (token, row).
// Each CTA scans H elements in a loop of H/BLOCK_H iterations,
// accumulating dot product with cw and sum-of-squares,
// then writes one fp32 score.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, type aliases
#include <sgl_kernel/vec.cuh>    // For AlignedVector
#include <sgl_kernel/warp.cuh>   // For warp::reduce_sum

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct AttnResScoreParams {
  const bf16_t* __restrict__ prefix_sum;  // [T, H]
  const bf16_t* __restrict__ bank;        // [T, NB_total, H]
  const fp32_t* __restrict__ cw;          // [H]
  fp32_t* __restrict__ scores;            // [T, MAX_ROWS]
  int64_t stride_pm;                      // prefix_sum stride along T (in elements)
  int64_t stride_bm;                      // bank stride along T (in elements)
  int64_t stride_bb;                      // bank stride along NB (in elements, = H)
  int64_t stride_sm;                      // scores stride along T (in elements, = MAX_ROWS)
  uint32_t H;                             // hidden size (e.g. 7168)
  uint32_t NVB;                           // number of valid bank rows (0..8)
  float eps;                              // RMSNorm epsilon
};

// kBlockH: number of threads per CTA = number of H elements per iteration
// Using 128-bit vectorized loads: each thread loads 8 bf16 values per vec op
template <int kBlockH, bool kUsePDL>
__global__ void attn_res_score_kernel(const AttnResScoreParams __grid_constant__ params) {
  using namespace device;

  const uint32_t pid_t = blockIdx.x;  // token index
  const uint32_t j = blockIdx.y;      // row index (0..NVB: bank rows, NVB: prefix_sum)

  if (j > params.NVB) return;

  // 128-bit vectorized: 8 bf16 elements per load
  constexpr int kVecN = 8;
  using vec_bf16_t = AlignedVector<bf16_t, kVecN>;
  using vec_fp32_t = AlignedVector<fp32_t, 4>;  // for cw: 4 fp32 = 16 bytes

  const uint32_t tid = threadIdx.x;
  const uint32_t H = params.H;

  // Each thread will process kVecN elements per step, striding by kBlockH threads
  // Total elements to process: H
  // Vectorized: H / kVecN vectors, each thread takes one vec per iteration

  float sumsq = 0.0f;
  float dotv = 0.0f;

  PDLWaitPrimary<kUsePDL>();

  // Determine the base pointer for this row
  const bf16_t* row_ptr;
  if (j < params.NVB) {
    row_ptr = params.bank + pid_t * params.stride_bm + j * params.stride_bb;
  } else {
    row_ptr = params.prefix_sum + pid_t * params.stride_pm;
  }

  // Scan H in chunks: each thread handles element indices tid*kVecN, (tid+kBlockH)*kVecN, ...
  // We process kVecN elements at a time with vectorized loads
  const uint32_t n_vecs = H / kVecN;  // H must be divisible by kVecN

  for (uint32_t vi = tid; vi < n_vecs; vi += kBlockH) {
    vec_bf16_t v_vec;
    v_vec.load(row_ptr, vi);

    // Load corresponding cw values (fp32, 4 per 128-bit load)
    // cw is fp32, so we need 2 vec loads of 4 fp32 to match 8 bf16
    vec_fp32_t cw_vec0, cw_vec1;
    cw_vec0.load(params.cw, vi * 2);
    cw_vec1.load(params.cw, vi * 2 + 1);

#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      float v_f32 = cast<fp32_t>(v_vec[i]);
      float cw_f32 = (i < 4) ? cw_vec0[i] : cw_vec1[i - 4];
      sumsq += v_f32 * v_f32;
      dotv += v_f32 * cw_f32;
    }
  }

  // CTA-level reduction via warp shuffle + shared memory
  // kBlockH threads, kBlockH/32 warps
  constexpr uint32_t kNumWarps = kBlockH / kWarpThreads;

  // Warp-level reduction
  sumsq = warp::reduce_sum(sumsq);
  dotv = warp::reduce_sum(dotv);

  // Cross-warp reduction via shared memory
  __shared__ float smem_sumsq[32];
  __shared__ float smem_dotv[32];

  const uint32_t warp_id = tid / kWarpThreads;
  const uint32_t lane_id = tid % kWarpThreads;

  if (lane_id == 0) {
    smem_sumsq[warp_id] = sumsq;
    smem_dotv[warp_id] = dotv;
  }
  __syncthreads();

  if (warp_id == 0) {
    float local_sumsq = (lane_id < kNumWarps) ? smem_sumsq[lane_id] : 0.0f;
    float local_dotv = (lane_id < kNumWarps) ? smem_dotv[lane_id] : 0.0f;
    local_sumsq = warp::reduce_sum(local_sumsq);
    local_dotv = warp::reduce_sum(local_dotv);

    if (lane_id == 0) {
      float rrms = rsqrtf(local_sumsq / static_cast<float>(H) + params.eps);
      params.scores[pid_t * params.stride_sm + j] = local_dotv * rrms;
    }
  }

  PDLTriggerSecondary<kUsePDL>();
}

// Host launcher
template <int kBlockH, bool kUsePDL>
struct AttnResScoreKernel {
  static constexpr auto kernel = attn_res_score_kernel<kBlockH, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView prefix_sum,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw,
      const tvm::ffi::TensorView scores,
      int64_t nvb,
      double eps) {
    using namespace host;

    auto T_ = SymbolicSize{"num_tokens"};
    auto H_ = SymbolicSize{"hidden_size"};
    auto NB_ = SymbolicSize{"num_bank_slots"};
    auto MR_ = SymbolicSize{"max_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(prefix_sum);
    TensorMatcher({T_, NB_, H_}).with_dtype<bf16_t>().with_device(device).verify(bank);
    TensorMatcher({H_}).with_dtype<fp32_t>().with_device(device).verify(cw);
    TensorMatcher({T_, MR_}).with_dtype<fp32_t>().with_device(device).verify(scores);

    const auto num_tokens = static_cast<uint32_t>(T_.unwrap());
    const auto H = static_cast<uint32_t>(H_.unwrap());
    const auto NVB = static_cast<uint32_t>(nvb);
    const auto max_rows = static_cast<uint32_t>(MR_.unwrap());

    RuntimeCheck(NVB <= max_rows, "NVB exceeds MAX_ROWS");
    RuntimeCheck(H % 8 == 0, "H must be divisible by 8 for vectorized bf16 loads");

    if (num_tokens == 0 || NVB + 1 == 0) return;

    const auto params = AttnResScoreParams{
        .prefix_sum = static_cast<const bf16_t*>(prefix_sum.data_ptr()),
        .bank = static_cast<const bf16_t*>(bank.data_ptr()),
        .cw = static_cast<const fp32_t*>(cw.data_ptr()),
        .scores = static_cast<fp32_t*>(scores.data_ptr()),
        .stride_pm = static_cast<int64_t>(H),
        .stride_bm = static_cast<int64_t>(NB_.unwrap()) * H,
        .stride_bb = static_cast<int64_t>(H),
        .stride_sm = static_cast<int64_t>(max_rows),
        .H = H,
        .NVB = NVB,
        .eps = static_cast<float>(eps),
    };

    dim3 grid(num_tokens, NVB + 1);
    LaunchKernel(grid, kBlockH, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
