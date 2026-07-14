// AttnRes score kernel with the upstream residual add fused in:
//   prefix = bf16(prefix_a + prefix_b)   (written out for combine/后续消费者)
// then identical per-row scoring to attn_res_score.cuh. The prefix-row CTA
// computes the add on the fly (fp32 add, bf16 round — bit-identical to the
// standalone elementwise add) and stores the rounded row; scoring uses the
// rounded values, matching the unfused add->score chain exactly.
//
// Grid: (T, NVB+1) — one CTA per (token, row), same as attn_res_score.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, type aliases
#include <sgl_kernel/vec.cuh>    // For AlignedVector
#include <sgl_kernel/warp.cuh>   // For warp::reduce_sum

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct AttnResScoreFAddParams {
  const bf16_t* __restrict__ prefix_a;  // [T, H]
  const bf16_t* __restrict__ prefix_b;  // [T, H]
  bf16_t* __restrict__ prefix_out;      // [T, H] (the materialized sum)
  const bf16_t* __restrict__ bank;      // [T, NB_total, H]
  const fp32_t* __restrict__ cw;        // [H]
  fp32_t* __restrict__ scores;          // [T, MAX_ROWS]
  int64_t stride_pm;                    // prefix_{a,b,out} stride along T
  int64_t stride_bm;
  int64_t stride_bb;
  int64_t stride_sm;
  uint32_t H;
  uint32_t NVB;
  float eps;
};

template <int kBlockH, bool kUsePDL>
__global__ void attn_res_score_fadd_kernel(const AttnResScoreFAddParams __grid_constant__ params) {
  using namespace device;

  const uint32_t pid_t = blockIdx.x;
  const uint32_t j = blockIdx.y;
  if (j > params.NVB) return;

  constexpr int kVecN = 8;
  using vec_bf16_t = AlignedVector<bf16_t, kVecN>;
  using vec_fp32_t = AlignedVector<fp32_t, 4>;

  const uint32_t tid = threadIdx.x;
  const uint32_t H = params.H;
  const uint32_t n_vecs = H / kVecN;
  const bool is_prefix_row = (j == params.NVB);

  float sumsq = 0.0f;
  float dotv = 0.0f;

  PDLWaitPrimary<kUsePDL>();

  const bf16_t* row_ptr = is_prefix_row ? nullptr : params.bank + pid_t * params.stride_bm + j * params.stride_bb;
  const bf16_t* a_ptr = params.prefix_a + pid_t * params.stride_pm;
  const bf16_t* b_ptr = params.prefix_b + pid_t * params.stride_pm;
  bf16_t* out_ptr = params.prefix_out + pid_t * params.stride_pm;

  for (uint32_t vi = tid; vi < n_vecs; vi += kBlockH) {
    vec_bf16_t v_vec;
    if (is_prefix_row) {
      vec_bf16_t av, bv;
      av.load(a_ptr, vi);
      bv.load(b_ptr, vi);
#pragma unroll
      for (int i = 0; i < kVecN; ++i) {
        // fp32 add, bf16 round — identical to the standalone add kernel.
        v_vec[i] = cast<bf16_t>(cast<fp32_t>(av[i]) + cast<fp32_t>(bv[i]));
      }
      v_vec.store(out_ptr, vi);
    } else {
      v_vec.load(row_ptr, vi);
    }

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

  constexpr uint32_t kNumWarps = kBlockH / kWarpThreads;
  sumsq = warp::reduce_sum(sumsq);
  dotv = warp::reduce_sum(dotv);

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

template <int kBlockH, bool kUsePDL>
struct AttnResScoreFAddKernel {
  static constexpr auto kernel = attn_res_score_fadd_kernel<kBlockH, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView prefix_a,
      const tvm::ffi::TensorView prefix_b,
      const tvm::ffi::TensorView prefix_out,
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

    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(prefix_a);
    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(prefix_b);
    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(prefix_out);
    TensorMatcher({T_, NB_, H_}).with_dtype<bf16_t>().with_device(device).verify(bank);
    TensorMatcher({H_}).with_dtype<fp32_t>().with_device(device).verify(cw);
    TensorMatcher({T_, MR_}).with_dtype<fp32_t>().with_device(device).verify(scores);

    const auto num_tokens = static_cast<uint32_t>(T_.unwrap());
    const auto H = static_cast<uint32_t>(H_.unwrap());
    const auto NVB = static_cast<uint32_t>(nvb);
    const auto max_rows = static_cast<uint32_t>(MR_.unwrap());

    RuntimeCheck(NVB <= max_rows, "NVB exceeds MAX_ROWS");
    RuntimeCheck(H % 8 == 0, "H must be divisible by 8");
    if (num_tokens == 0) return;

    const auto params = AttnResScoreFAddParams{
        .prefix_a = static_cast<const bf16_t*>(prefix_a.data_ptr()),
        .prefix_b = static_cast<const bf16_t*>(prefix_b.data_ptr()),
        .prefix_out = static_cast<bf16_t*>(prefix_out.data_ptr()),
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
