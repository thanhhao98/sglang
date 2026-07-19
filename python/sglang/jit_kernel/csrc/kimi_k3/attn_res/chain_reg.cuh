// AttnRes optimized 3-kernel chain: score -> merge -> norm, one host call.
//
// kDim / kMaxBankRows come from the Python side as template parameters of the
// launcher; all derived tiling constants live in AttnResChainTrait. run()
// launches the whole chain and allocates the scores / partials / mixed
// workspace itself (one ffi::empty allocation, sliced).
//
// - score: one CTA per (token, row), widest vectors x kUnroll = 2 loads per
//   thread; only the prefix-row CTA PDL-waits on the previous kernel.
// - merge: softmax(scores) -> weighted sum, row count dispatched through a
//   constexpr kernel table. Rows are prefetched before the PDL wait (score
//   only writes the scores buffer). Each of the kNumSplit CTAs along H also
//   writes sum(mixed^2) of its chunk into the padded partials row.
// - norm: mixed * rsqrt(sum(partials)/kDim + eps) * ow. One vector load of
//   the partials per thread — no shared memory, no __syncthreads.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <array>
#include <bit>
#include <cstdint>
#include <utility>

namespace sglang {

template <int64_t kDim_, uint32_t kMaxBankRows_>
struct AttnResChainTrait {
  static constexpr int64_t kDim = kDim_;
  static constexpr uint32_t kMaxBankRows = kMaxBankRows_;
  // scores row width: kMaxBankRows + 1 rounded up to a power of two, so the
  // partials slice that follows it in the workspace stays vector-aligned.
  static constexpr uint32_t kScoreStride = std::bit_ceil(kMaxBankRows_ + 1);

  // score / norm tiling: widest vectors, kUnroll loads per thread.
  static constexpr uint32_t kUnroll = 2;
  static constexpr uint32_t kVecSize = device::kMaxVecBytes / sizeof(bf16_t);
  static constexpr uint32_t kWideBlockSize = kDim / (kVecSize * kUnroll);
  static constexpr uint32_t kWideNumWarps = kWideBlockSize / device::kWarpThreads;

  // merge tiling: 16B vectors, no unroll, kNumSplit CTAs along H.
  static constexpr uint32_t kMergeVecSize = 8;  // 8 bf16 = 16B
  static constexpr uint32_t kMergeThreads = 128;
  static constexpr uint32_t kMergeChunk = kMergeThreads * kMergeVecSize;
  static constexpr uint32_t kMergeNumWarps = kMergeThreads / device::kWarpThreads;
  static constexpr uint32_t kNumSplit = kDim / kMergeChunk;

  // partials row width: kNumSplit padded to whole fp32 vectors so norm can
  // load the row with aligned vector loads.
  static constexpr uint32_t kPartVecSize = device::kMaxVecBytes / sizeof(fp32_t);
  static constexpr uint32_t kNumPartials = (kNumSplit + kPartVecSize - 1) / kPartVecSize * kPartVecSize;
  static constexpr uint32_t kPartVecs = kNumPartials / kPartVecSize;

  using wide_vec_t = device::AlignedVector<bf16x2_t, kVecSize / 2>;
  using merge_vec_t = device::AlignedVector<bf16x2_t, kMergeVecSize / 2>;
  using part_vec_t = device::AlignedVector<fp32_t, kPartVecSize>;

  static_assert(kWideBlockSize * kVecSize * kUnroll == kDim, "kDim must tile into kVecSize * kUnroll");
  static_assert(kWideNumWarps * device::kWarpThreads == kWideBlockSize, "wide block must be whole warps");
  static_assert(kWideBlockSize <= 1024, "wide block exceeds the maximum block size");
  static_assert(kWideNumWarps <= device::kWarpThreads, "score cross-warp reduce must fit in one warp");
  static_assert(kNumSplit * kMergeChunk == kDim, "kDim must be divisible by the merge chunk");
  static_assert(kMergeNumWarps * device::kWarpThreads == kMergeThreads, "merge block must be whole warps");
  static_assert(kScoreStride >= kMaxBankRows + 1, "scores row must hold all rows");
  static_assert(kNumPartials >= kNumSplit && kNumPartials % kPartVecSize == 0, "invalid partials padding");

  // Workspace layout (per token, in bytes): fp32 scores row, fp32 partials
  // row, then the bf16 mixed row. Every slice stays kMaxVecBytes-aligned
  // because the fp32 rows are whole 32B vectors.
  static constexpr int64_t kScoresBytes = kScoreStride * sizeof(fp32_t);
  static constexpr int64_t kPartialsBytes = kNumPartials * sizeof(fp32_t);
  static constexpr int64_t kMixedBytes = kDim * sizeof(bf16_t);
  static_assert(kScoresBytes % device::kMaxVecBytes == 0, "scores slice breaks workspace alignment");
  static_assert(kPartialsBytes % device::kMaxVecBytes == 0, "partials slice breaks workspace alignment");
};

// ---------------------------------------------------------------------------
// Kernel 1: score — per-row dot(v, cw) * rrms(v).  Grid: (T, NVB+1).
// ---------------------------------------------------------------------------

struct AttnResChainScoreParams {
  const bf16_t* __restrict__ prefix_sum;  // [T, H]
  const bf16_t* __restrict__ bank;        // [T, NB_total, H]
  const bf16_t* __restrict__ cw;          // [H] score norm ⊙ proj weight
  fp32_t* __restrict__ scores;            // [T, kScoreStride]
  int64_t stride_pm;                      // prefix_sum stride along T (in elements)
  int64_t stride_bm;                      // bank stride along T (in elements)
  uint32_t NVB;                           // number of valid bank rows
  float eps;
};

template <typename T, bool kUsePDL>
__global__ void __launch_bounds__(T::kWideBlockSize)
    attn_res_chain_score_kernel(const __grid_constant__ AttnResChainScoreParams params) {
  using namespace device;
  constexpr uint32_t kNumWarps = T::kWideNumWarps;

  const auto pid_t = blockIdx.x;
  const auto j = blockIdx.y;  // row: 0..NVB-1 = bank, NVB = prefix
  const auto tx = threadIdx.x;
  const bool is_prefix_row = (j == params.NVB);

  __shared__ float s_rms[kNumWarps];
  __shared__ float s_dot[kNumWarps];

  // Only the prefix row is written by the immediately-preceding kernel; the
  // bank rows are older, so their CTAs skip the PDL wait.
  if (is_prefix_row) PDLWaitPrimary<kUsePDL>();
  const bf16_t* row_ptr = is_prefix_row ? params.prefix_sum + pid_t * params.stride_pm
                                        : params.bank + pid_t * params.stride_bm + j * T::kDim;

  typename T::wide_vec_t x[T::kUnroll], w[T::kUnroll];
#pragma unroll
  for (uint32_t i = 0; i < T::kUnroll; ++i) {
    x[i].load(row_ptr, tx + i * T::kWideBlockSize);
  }
#pragma unroll
  for (uint32_t i = 0; i < T::kUnroll; ++i) {
    w[i].load(params.cw, tx + i * T::kWideBlockSize);
  }
  PDLTriggerSecondary<kUsePDL>();

  float acc_rms = 0.0f;
  float acc_dot = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < T::kUnroll; ++i) {
#pragma unroll
    for (uint32_t v = 0; v < T::kVecSize / 2; ++v) {
      const auto [x0, x1] = cast<fp32x2_t>(x[i][v]);
      const auto [w0, w1] = cast<fp32x2_t>(w[i][v]);
      acc_rms += x0 * x0 + x1 * x1;
      acc_dot += x0 * w0 + x1 * w1;
    }
  }

  s_rms[tx / kWarpThreads] = warp::reduce_sum(acc_rms);
  s_dot[tx / kWarpThreads] = warp::reduce_sum(acc_dot);
  __syncthreads();

  // Only one scalar leaves the CTA, so a single warp finishes the reduction.
  if (tx < kWarpThreads) {
    const auto rms = warp::reduce_sum(tx < kNumWarps ? s_rms[tx] : 0.0f);
    const auto dot = warp::reduce_sum(tx < kNumWarps ? s_dot[tx] : 0.0f);
    if (tx == 0) {
      const auto rrms = rsqrtf(rms / static_cast<float>(T::kDim) + params.eps);
      params.scores[pid_t * T::kScoreStride + j] = dot * rrms;
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: merge — softmax(scores) -> weighted sum + per-chunk partial
// sum-of-squares.  Grid: (T, kNumSplit).  kRows = NVB is a template parameter.
// ---------------------------------------------------------------------------

struct AttnResChainMergeParams {
  const bf16_t* __restrict__ prefix_sum;  // [T, H]
  const bf16_t* __restrict__ bank;        // [T, NB_total, H]
  const fp32_t* __restrict__ scores;      // [T, kScoreStride]
  bf16_t* __restrict__ mixed;             // [T, H]
  fp32_t* __restrict__ partials;          // [T, kNumPartials] sum(mixed^2) per chunk
  int64_t stride_pm;                      // prefix_sum stride along T (in elements)
  int64_t stride_bm;                      // bank stride along T (in elements)
};

template <typename T, uint32_t kRows, bool kUsePDL>
__global__ void __launch_bounds__(T::kMergeThreads)
    attn_res_chain_merge_kernel(const __grid_constant__ AttnResChainMergeParams params) {
  using namespace device;
  constexpr uint32_t kTotalRows = kRows + 1;
  constexpr uint32_t kNumWarps = T::kMergeNumWarps;

  const auto pid_t = blockIdx.x;
  const auto chunk = blockIdx.y;
  const auto tx = threadIdx.x;
  const auto vec_idx = chunk * T::kMergeThreads + tx;

  __shared__ float s_sumsq[kNumWarps];

  // The primary (score) only writes the scores buffer: prefetch all rows
  // before the PDL wait, and wait only for the scores load.
  typename T::merge_vec_t rows[kTotalRows];
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
    rows[r].load(params.bank + pid_t * params.stride_bm + r * T::kDim, vec_idx);
  }
  rows[kRows].load(params.prefix_sum + pid_t * params.stride_pm, vec_idx);

  PDLWaitPrimary<kUsePDL>();
  float probs[kTotalRows];
#pragma unroll
  for (uint32_t r = 0; r < kTotalRows; ++r) {
    probs[r] = params.scores[pid_t * T::kScoreStride + r];
  }
  PDLTriggerSecondary<kUsePDL>();

  // Softmax, redundantly per thread (kTotalRows registers).
  float max_score = probs[0];
#pragma unroll
  for (uint32_t r = 1; r < kTotalRows; ++r) {
    max_score = fmaxf(max_score, probs[r]);
  }
  float exp_sum = 0.0f;
#pragma unroll
  for (uint32_t r = 0; r < kTotalRows; ++r) {
    probs[r] = expf(probs[r] - max_score);
    exp_sum += probs[r];
  }
  const float inv_sum = 1.0f / exp_sum;

  float acc[T::kMergeVecSize] = {};
#pragma unroll
  for (uint32_t r = 0; r < kTotalRows; ++r) {
    const float p = probs[r] * inv_sum;
#pragma unroll
    for (uint32_t v = 0; v < T::kMergeVecSize / 2; ++v) {
      const auto [x0, x1] = cast<fp32x2_t>(rows[r][v]);
      acc[2 * v + 0] += p * x0;
      acc[2 * v + 1] += p * x1;
    }
  }

  typename T::merge_vec_t out_vec;
  float sumsq = 0.0f;
#pragma unroll
  for (uint32_t v = 0; v < T::kMergeVecSize / 2; ++v) {
    const auto x0 = acc[2 * v + 0];
    const auto x1 = acc[2 * v + 1];
    sumsq += x0 * x0 + x1 * x1;
    out_vec[v] = cast<bf16x2_t>(fp32x2_t{x0, x1});
  }
  out_vec.store(params.mixed + pid_t * T::kDim, vec_idx);

  // Per-chunk partial sum(mixed^2), so norm needs no cross-CTA reduction.
  s_sumsq[tx / kWarpThreads] = warp::reduce_sum(sumsq);
  __syncthreads();
  if (tx == 0) {
    float total = s_sumsq[0];
#pragma unroll
    for (uint32_t i = 1; i < kNumWarps; ++i) {
      total += s_sumsq[i];
    }
    params.partials[pid_t * T::kNumPartials + chunk] = total;
  }
}

// ---------------------------------------------------------------------------
// Kernel 3: norm — mixed * rsqrt(sum(partials)/kDim + eps) * ow.  Grid: (T,).
// ---------------------------------------------------------------------------

struct AttnResChainNormParams {
  const bf16_t* __restrict__ mixed;     // [T, H]
  const fp32_t* __restrict__ partials;  // [T, kNumPartials]
  const bf16_t* __restrict__ ow;        // [H] out norm weight
  bf16_t* __restrict__ out;             // [T, H]
  float eps;
};

template <typename T, bool kUsePDL>
__global__ void __launch_bounds__(T::kWideBlockSize)
    attn_res_chain_norm_kernel(const __grid_constant__ AttnResChainNormParams params) {
  using namespace device;

  const auto pid_t = blockIdx.x;
  const auto tx = threadIdx.x;

  // The out-norm weight is never written by the pipeline: prefetch it before
  // the PDL wait.
  typename T::wide_vec_t w[T::kUnroll];
#pragma unroll
  for (uint32_t i = 0; i < T::kUnroll; ++i) {
    w[i].load(params.ow, tx + i * T::kWideBlockSize);
  }

  PDLWaitPrimary<kUsePDL>();
  typename T::wide_vec_t x[T::kUnroll];
#pragma unroll
  for (uint32_t i = 0; i < T::kUnroll; ++i) {
    x[i].load(params.mixed + pid_t * T::kDim, tx + i * T::kWideBlockSize);
  }
  typename T::part_vec_t part[T::kPartVecs];
#pragma unroll
  for (uint32_t i = 0; i < T::kPartVecs; ++i) {
    part[i].load(params.partials + pid_t * T::kNumPartials, i);
  }
  PDLTriggerSecondary<kUsePDL>();

  // Sum the kNumSplit valid partials in-thread (the pad slots are garbage).
  float sumsq = 0.0f;
#pragma unroll
  for (uint32_t i = 0; i < T::kNumSplit; ++i) {
    sumsq += part[i / T::kPartVecSize][i % T::kPartVecSize];
  }
  const float scale = rsqrtf(sumsq / static_cast<float>(T::kDim) + params.eps);

#pragma unroll
  for (uint32_t i = 0; i < T::kUnroll; ++i) {
    typename T::wide_vec_t out_vec;
#pragma unroll
    for (uint32_t v = 0; v < T::kVecSize / 2; ++v) {
      const auto [x0, x1] = cast<fp32x2_t>(x[i][v]);
      const auto [w0, w1] = cast<fp32x2_t>(w[i][v]);
      out_vec[v] = cast<bf16x2_t>(fp32x2_t{x0 * scale * w0, x1 * scale * w1});
    }
    out_vec.store(params.out + pid_t * T::kDim, tx + i * T::kWideBlockSize);
  }
}

}  // namespace sglang

using namespace sglang;

// ---------------------------------------------------------------------------
// Host launcher: one call runs score -> merge -> norm, allocating the
// scores / partials / mixed workspace in a single ffi::empty allocation.
// ---------------------------------------------------------------------------

template <int64_t kDim, uint32_t kMaxBankRows, bool kUsePDL>
struct AttnResChain {
  using Trait = AttnResChainTrait<kDim, kMaxBankRows>;
  using MergeFn = void (*)(const AttnResChainMergeParams);

  // Constexpr dispatch table over the bank-row count: kMergeTable[nvb] is the
  // instantiation with kRows = nvb (slot 0 unused).
  template <std::size_t... I>
  static constexpr auto make_merge_table(std::index_sequence<I...>) {
    return std::array<MergeFn, kMaxBankRows + 1>{nullptr, attn_res_chain_merge_kernel<Trait, I + 1, kUsePDL>...};
  }
  static constexpr auto kMergeTable = make_merge_table(std::make_index_sequence<kMaxBankRows>{});

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

    const auto num_tokens = static_cast<int64_t>(T_.unwrap());
    const auto H = static_cast<int64_t>(H_.unwrap());
    const auto NB = static_cast<int64_t>(NB_.unwrap());

    RuntimeCheck(H == kDim, "attn_res_chain: H must be ", kDim, ", got ", H);
    RuntimeCheck(
        1 <= nvb && nvb <= kMaxBankRows && nvb <= NB,
        "attn_res_chain: nvb must be in [1, ",
        kMaxBankRows,
        "] and <= NB, got nvb=",
        nvb,
        " NB=",
        NB);

    if (num_tokens == 0) return;

    // One workspace allocation, sliced: [scores fp32 | partials fp32 | mixed bf16].
    const auto ws_bytes = num_tokens * (Trait::kScoresBytes + Trait::kPartialsBytes + Trait::kMixedBytes);
    auto workspace = alloc_workspace_tensor(static_cast<size_t>(ws_bytes), device.unwrap());
    auto* const ws_base = static_cast<uint8_t*>(workspace.data_ptr());
    auto* const scores = reinterpret_cast<fp32_t*>(ws_base);
    auto* const partials = reinterpret_cast<fp32_t*>(ws_base + num_tokens * Trait::kScoresBytes);
    auto* const mixed = reinterpret_cast<bf16_t*>(ws_base + num_tokens * (Trait::kScoresBytes + Trait::kPartialsBytes));

    const auto prefix_ptr = static_cast<const bf16_t*>(prefix_sum.data_ptr());
    const auto bank_ptr = static_cast<const bf16_t*>(bank.data_ptr());
    const auto stride_bm = NB * kDim;

    const auto score_params = AttnResChainScoreParams{
        .prefix_sum = prefix_ptr,
        .bank = bank_ptr,
        .cw = static_cast<const bf16_t*>(cw.data_ptr()),
        .scores = scores,
        .stride_pm = kDim,
        .stride_bm = stride_bm,
        .NVB = static_cast<uint32_t>(nvb),
        .eps = static_cast<float>(eps),
    };
    dim3 score_grid(num_tokens, static_cast<uint32_t>(nvb) + 1);
    LaunchKernel(score_grid, Trait::kWideBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(attn_res_chain_score_kernel<Trait, kUsePDL>, score_params);

    const auto merge_params = AttnResChainMergeParams{
        .prefix_sum = prefix_ptr,
        .bank = bank_ptr,
        .scores = scores,
        .mixed = mixed,
        .partials = partials,
        .stride_pm = kDim,
        .stride_bm = stride_bm,
    };
    dim3 merge_grid(num_tokens, Trait::kNumSplit);
    LaunchKernel(merge_grid, Trait::kMergeThreads, device.unwrap()).enable_pdl(kUsePDL)(kMergeTable[nvb], merge_params);

    const auto norm_params = AttnResChainNormParams{
        .mixed = mixed,
        .partials = partials,
        .ow = static_cast<const bf16_t*>(ow.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .eps = static_cast<float>(eps),
    };
    LaunchKernel(num_tokens, Trait::kWideBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(attn_res_chain_norm_kernel<Trait, kUsePDL>, norm_params);
  }
};
