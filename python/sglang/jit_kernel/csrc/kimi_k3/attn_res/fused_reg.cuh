// AttnRes fully-fused kernel: per-row score -> softmax -> weighted combine
// -> output RMSNorm, one CTA per token, every row read exactly once from HBM.
//
// kDim / kMaxBankRows come from the Python side as template parameters of the
// launcher; the bank-row count is dispatched through a constexpr kernel table
// (tiny_gemm style). Requires SM100+ (fma.rn.f32.bf16).
//
// Grid: (T,)  Block: Trait::kBlockSize = kDim / kVecSize.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <array>
#include <cfloat>
#include <cstdint>
#include <utility>

namespace sglang {

struct AttnResFusedParams {
  const bf16_t* __restrict__ prefix_sum;  // [T, H]
  const bf16_t* __restrict__ bank;        // [T, NB_total, H]
  const bf16_t* __restrict__ cw;          // [H] score norm ⊙ proj weight
  const bf16_t* __restrict__ ow;          // [H] out norm weight
  bf16_t* __restrict__ out;               // [T, H]
  int64_t stride_pm;                      // prefix_sum stride along T (in elements)
  int64_t stride_bm;                      // bank stride along T (in elements)
  int64_t stride_om;                      // out stride along T (in elements)
  float eps;                              // RMSNorm epsilon
};

template <int64_t kDim_, uint32_t kMaxBankRows_>
struct AttnResFusedTrait {
  static constexpr int64_t kDim = kDim_;
  static constexpr uint32_t kMaxBankRows = kMaxBankRows_;
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(bf16_t);
  static constexpr uint32_t kBlockSize = kDim / kVecSize;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static constexpr float kScale = 1.0f / static_cast<float>(kDim);
  using vec_t = device::AlignedVector<bf16_t, kVecSize>;
  using vec2_t = device::AlignedVector<bf16x2_t, kVecSize / 2>;

  static_assert(kBlockSize * kVecSize == kDim, "kDim must be divisible by the vector width");
  static_assert(kNumWarps * device::kWarpThreads == kBlockSize, "block must be whole warps");
  static_assert(kBlockSize <= 1024, "kDim / kVecSize exceeds the maximum block size");
  static_assert(kNumWarps <= device::kWarpThreads, "cross-warp reduce must fit in one warp");
  static_assert(kMaxBankRows + 1 <= device::kWarpThreads, "softmax rows must fit in one warp");
};

SGL_DEVICE float fma_f32_bf16(bf16_t a, bf16_t b, float acc) {
  const uint16_t a_bits = __bfloat16_as_ushort(a);
  const uint16_t b_bits = __bfloat16_as_ushort(b);
  float result;
  asm("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=f"(result) : "h"(a_bits), "h"(b_bits), "f"(acc));
  return result;
}

SGL_DEVICE float fma_f32(float a, float b, float c) {
  return fmaf(a, b, c);
}

template <typename T, uint32_t kNumRows, bool kUsePDL>
__global__ void __launch_bounds__(T::kBlockSize)
    attn_res_fused_kernel(const __grid_constant__ AttnResFusedParams params) {
  using namespace device;
  constexpr uint32_t kVecSize = T::kVecSize;
  constexpr uint32_t kBlockSize = T::kBlockSize;
  constexpr uint32_t kNumWarps = T::kNumWarps;
  using vec_t = typename T::vec_t;
  using vec2_t = typename T::vec2_t;

  const auto bx = blockIdx.x;
  const auto tx = threadIdx.x;
  const auto warp_id = tx / kWarpThreads;

  const auto input_ptr = params.bank + bx * params.stride_bm;
  vec_t x[kNumRows + 1], w;

  // prefetch weight
  w.load(params.cw, tx);

  __shared__ float s_rms[kNumRows + 1][kNumWarps + 1];
  __shared__ float s_dot[kNumRows + 1][kNumWarps + 1];
  __shared__ float s_score[kWarpThreads];

  // Bank rows are produced well before the immediately-preceding kernel, so
  // they may be prefetched before the PDL wait; only the prefix row (written
  // by the preceding kernel) must wait.
#pragma unroll
  for (uint32_t i = 0; i < kNumRows; ++i) {
    x[i].load(input_ptr, tx + i * kBlockSize);
  }

#pragma unroll
  for (uint32_t i = 0; i < kNumRows + 1; ++i) {
    if (i == kNumRows - 1) {
      PDLWaitPrimary<kUsePDL>();
      x[kNumRows].load(params.prefix_sum + bx * params.stride_pm, tx);
    }
    float acc_rms = 0.0f;
    float acc_dot = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kVecSize; ++j) {
      acc_rms = fma_f32_bf16(x[i][j], x[i][j], acc_rms);
      acc_dot = fma_f32_bf16(x[i][j], w[j], acc_dot);
    }
    s_rms[i][warp_id] = warp::reduce_sum(acc_rms);
    s_dot[i][warp_id] = warp::reduce_sum(acc_dot);
  }

  // replace weight with out norm
  __syncthreads();
  if (tx < kWarpThreads) {
    float score = -FLT_MAX;
    if (tx < kNumRows + 1) {
      float warp_rms[kNumWarps];
      float warp_dot[kNumWarps];
#pragma unroll
      for (uint32_t i = 0; i < kNumWarps; ++i) {
        warp_rms[i] = s_rms[tx][i];
        warp_dot[i] = s_dot[tx][i];
      }
      float total_rms = warp_rms[0];
      float total_dot = warp_dot[0];
#pragma unroll
      for (uint32_t i = 1; i < kNumWarps; ++i) {
        total_rms += warp_rms[i];
        total_dot += warp_dot[i];
      }
      score = total_dot * rsqrtf(total_rms * T::kScale + params.eps);
    }
    // TODO: replace this with redux.sync.max.f32, single instruction
    const auto max_val = warp::reduce_max(score);
    const auto exp_val = expf(score - max_val);
    const auto sum_val = warp::reduce_sum(tx < kNumRows + 1 ? exp_val : 0.0f);
    s_score[tx] = exp_val / sum_val;
  }
  w.load(params.ow, tx);
  __syncthreads();

  float out[kVecSize] = {};
#pragma unroll
  for (uint32_t i = 0; i < kNumRows + 1; ++i) {
    auto& y = reinterpret_cast<vec2_t&>(x[i]);
    const float score = s_score[i];
#pragma unroll
    for (uint32_t j = 0; j < kVecSize / 2; ++j) {
      const auto fp32_val = cast<float2>(y[j]);
      out[2 * j + 0] = fma_f32(fp32_val.x, score, out[2 * j + 0]);
      out[2 * j + 1] = fma_f32(fp32_val.y, score, out[2 * j + 1]);
    }
  }

  float acc_rms = out[0] * out[0];
#pragma unroll
  for (uint32_t j = 1; j < kVecSize; ++j) {
    acc_rms = fma_f32(out[j], out[j], acc_rms);
  }

  s_rms[0][warp_id] = warp::reduce_sum(acc_rms);
  __syncthreads();
  if (warp_id == 0) {
    const auto value = tx < kNumWarps ? s_rms[0][tx] : 0.0f;
    const auto total = warp::reduce_sum(value);
    if (tx < kNumWarps) {
      s_rms[0][tx] = rsqrtf(total * T::kScale + params.eps);
    }
  }

  __syncthreads();
  const auto rms_scale = s_rms[0][warp_id];
  const auto& w2 = reinterpret_cast<vec2_t&>(w);
  vec2_t out_vec;
#pragma unroll
  for (uint32_t j = 0; j < kVecSize / 2; ++j) {
    const auto fp32_w = cast<float2>(w2[j]);
    const auto x = out[2 * j + 0] * rms_scale * fp32_w.x;
    const auto y = out[2 * j + 1] * rms_scale * fp32_w.y;
    out_vec[j] = cast<bf16x2_t>(make_float2(x, y));
  }

  out_vec.store(params.out + bx * params.stride_om, tx);

  PDLTriggerSecondary<kUsePDL>();
}

}  // namespace sglang

using namespace sglang;

template <int64_t kDim, uint32_t kMaxBankRows, bool kUsePDL>
struct AttnResFusedKernel {
  using Trait = AttnResFusedTrait<kDim, kMaxBankRows>;
  using KernelFn = void (*)(const AttnResFusedParams);

  // Constexpr dispatch table over the bank-row count: kTable[nvb] is the
  // instantiation with kNumRows = nvb (slot 0 unused).
  template <std::size_t... I>
  static constexpr auto make_table(std::index_sequence<I...>) {
    return std::array<KernelFn, kMaxBankRows + 1>{nullptr, attn_res_fused_kernel<Trait, I + 1, kUsePDL>...};
  }
  static constexpr auto kTable = make_table(std::make_index_sequence<kMaxBankRows>{});

  static void
  run(const tvm::ffi::TensorView prefix_sum,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView cw,
      const tvm::ffi::TensorView ow,
      const tvm::ffi::TensorView out,
      int64_t nvb,
      double eps) {
    using namespace host;

    auto T_ = SymbolicSize{"num_tokens"};
    auto H_ = SymbolicSize{"hidden_size"};
    auto NB_ = SymbolicSize{"num_bank_slots"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(prefix_sum).verify(out);
    TensorMatcher({T_, NB_, H_}).with_dtype<bf16_t>().with_device(device).verify(bank);
    TensorMatcher({H_}).with_dtype<bf16_t>().with_device(device).verify(cw).verify(ow);

    const auto num_tokens = static_cast<uint32_t>(T_.unwrap());
    const auto H = static_cast<int64_t>(H_.unwrap());
    const auto NB = static_cast<int64_t>(NB_.unwrap());

    RuntimeCheck(H == kDim, "attn_res_fused: H must be ", kDim, ", got ", H);
    RuntimeCheck(
        1 <= nvb && nvb <= kMaxBankRows && nvb <= NB,
        "attn_res_fused: nvb must be in [1, ",
        kMaxBankRows,
        "] and <= NB, got nvb=",
        nvb,
        " NB=",
        NB);

    if (num_tokens == 0) return;

    const auto params = AttnResFusedParams{
        .prefix_sum = static_cast<const bf16_t*>(prefix_sum.data_ptr()),
        .bank = static_cast<const bf16_t*>(bank.data_ptr()),
        .cw = static_cast<const bf16_t*>(cw.data_ptr()),
        .ow = static_cast<const bf16_t*>(ow.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .stride_pm = H,
        .stride_bm = NB * H,
        .stride_om = H,
        .eps = static_cast<float>(eps),
    };

    LaunchKernel(num_tokens, Trait::kBlockSize, device.unwrap()).enable_pdl(kUsePDL)(kTable[nvb], params);
  }
};
