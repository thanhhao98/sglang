// AttnRes combine kernel: softmax(scores) -> weighted sum of rows.
//
// Grid: (T, H/kChunkH)  —  one CTA per (token, H-chunk).
// Each CTA redundantly computes softmax over the <=16 scores,
// then reads its H-chunk from all NVB+1 rows and accumulates
// the weighted sum. Output is bf16.
//
// Thread count: kChunkH / 8  (128 for kChunkH=1024).
// Each thread handles one 128-bit vector (8 bf16 elements).

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, SGL_DEVICE, type aliases
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct AttnResCombineParams {
  const bf16_t* __restrict__ prefix_sum;  // [T, H]
  const bf16_t* __restrict__ bank;        // [T, NB_total, H]
  const fp32_t* __restrict__ scores;      // [T, MAX_ROWS]
  bf16_t* __restrict__ out;               // [T, H]
  int64_t stride_pm;   // prefix_sum stride along T (in elements)
  int64_t stride_bm;   // bank stride along T (in elements)
  int64_t stride_bb;   // bank stride along NB (in elements, = H)
  int64_t stride_sm;   // scores stride along T (in elements, = MAX_ROWS)
  int64_t stride_om;   // out stride along T (in elements, = H)
  uint32_t NVB;        // number of valid bank rows (0..8)
};

// kChunkH: H elements per chunk (e.g. 1024). Grid.y = H / kChunkH.
// kMaxRows: compile-time upper bound on NVB+1 (power of 2, e.g. 16).
// Threads per CTA: kChunkH / 8 (128 for 1024-element chunks).
template <int kChunkH, int kMaxRows, bool kUsePDL>
__global__ void attn_res_combine_kernel(const AttnResCombineParams __grid_constant__ params) {
  using namespace device;

  const uint32_t pid_t = blockIdx.x;   // token index
  const uint32_t pid_h = blockIdx.y;   // H-chunk index
  const uint32_t h0 = pid_h * kChunkH; // first element of this chunk
  const uint32_t tid = threadIdx.x;
  const uint32_t NVB = params.NVB;

  PDLWaitPrimary<kUsePDL>();

  // ---- Phase 1: Softmax over scores ----
  // Redundant per chunk (<=16 fp32 ops, trivial cost).
  // All threads compute the same softmax — no reduction needed.
  float probs[kMaxRows];
  {
    float max_score = -1e30f;
#pragma unroll
    for (int j = 0; j < kMaxRows; ++j) {
      if (static_cast<uint32_t>(j) <= NVB) {
        probs[j] = params.scores[pid_t * params.stride_sm + j];
        max_score = fmaxf(max_score, probs[j]);
      } else {
        probs[j] = -1e30f;
      }
    }

    float exp_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kMaxRows; ++j) {
      if (static_cast<uint32_t>(j) <= NVB) {
        probs[j] = expf(probs[j] - max_score);
        exp_sum += probs[j];
      } else {
        probs[j] = 0.0f;
      }
    }

    float inv_sum = 1.0f / exp_sum;
#pragma unroll
    for (int j = 0; j < kMaxRows; ++j) {
      probs[j] *= inv_sum;
    }
  }

  // ---- Phase 2: Weighted sum for this thread's vector ----
  constexpr int kVecN = 8;  // 8 bf16 = 16 bytes = 128 bits
  using vec_bf16_t = AlignedVector<bf16_t, kVecN>;

  // tid is in [0, kChunkH/kVecN).  Each thread owns one kVecN-wide slice.
  // Global H-offset of this thread's vector: h0 + tid * kVecN
  // Vector index from row base: (h0 + tid * kVecN) / kVecN = h0/kVecN + tid
  const uint32_t vec_idx = h0 / kVecN + tid;

  float acc[kVecN];
#pragma unroll
  for (int i = 0; i < kVecN; ++i) {
    acc[i] = 0.0f;
  }

  for (uint32_t j = 0; j <= NVB; ++j) {
    const bf16_t* row_ptr;
    if (j < NVB) {
      row_ptr = params.bank + pid_t * params.stride_bm + j * params.stride_bb;
    } else {
      row_ptr = params.prefix_sum + pid_t * params.stride_pm;
    }

    vec_bf16_t v_vec;
    v_vec.load(row_ptr, vec_idx);

    float p = probs[j];
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      acc[i] += p * cast<fp32_t>(v_vec[i]);
    }
  }

  // Store result as bf16
  vec_bf16_t out_vec;
#pragma unroll
  for (int i = 0; i < kVecN; ++i) {
    out_vec[i] = cast<bf16_t>(acc[i]);
  }
  out_vec.store(params.out + pid_t * params.stride_om, vec_idx);

  PDLTriggerSecondary<kUsePDL>();
}

// Host launcher
template <int kChunkH, int kMaxRows, bool kUsePDL>
struct AttnResCombineKernel {
  static constexpr auto kernel = attn_res_combine_kernel<kChunkH, kMaxRows, kUsePDL>;
  static constexpr uint32_t kNumThreads = kChunkH / 8;  // 128 for 1024-element chunks

  static void run(
      const tvm::ffi::TensorView prefix_sum,
      const tvm::ffi::TensorView bank,
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView out,
      int64_t nvb) {
    using namespace host;

    auto T_ = SymbolicSize{"num_tokens"};
    auto H_ = SymbolicSize{"hidden_size"};
    auto NB_ = SymbolicSize{"num_bank_slots"};
    auto MR_ = SymbolicSize{"max_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({T_, H_})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(prefix_sum);
    TensorMatcher({T_, NB_, H_})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(bank);
    TensorMatcher({T_, MR_})
        .with_dtype<fp32_t>()
        .with_device(device)
        .verify(scores);
    TensorMatcher({T_, H_})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(out);

    const auto num_tokens = static_cast<uint32_t>(T_.unwrap());
    const auto H = static_cast<uint32_t>(H_.unwrap());
    const auto NVB = static_cast<uint32_t>(nvb);

    RuntimeCheck(H % kChunkH == 0, "H must be divisible by kChunkH");
    RuntimeCheck(NVB + 1 <= static_cast<uint32_t>(kMaxRows), "NVB+1 exceeds MAX_ROWS");

    if (num_tokens == 0) return;

    const uint32_t n_h_blocks = H / kChunkH;

    const auto params = AttnResCombineParams{
        .prefix_sum = static_cast<const bf16_t*>(prefix_sum.data_ptr()),
        .bank = static_cast<const bf16_t*>(bank.data_ptr()),
        .scores = static_cast<const fp32_t*>(scores.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .stride_pm = static_cast<int64_t>(H),
        .stride_bm = static_cast<int64_t>(NB_.unwrap()) * H,
        .stride_bb = static_cast<int64_t>(H),
        .stride_sm = static_cast<int64_t>(MR_.unwrap()),
        .stride_om = static_cast<int64_t>(H),
        .NVB = NVB,
    };

    dim3 grid(num_tokens, n_h_blocks);
    LaunchKernel(grid, kNumThreads, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
