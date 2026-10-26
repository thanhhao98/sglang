// Fused deferred-MoE finalize -> 1shot lamport push all-reduce [-> RMSNorm]
// over the CustomAllReduceV2 push plane, for decode-sized batches (bf16). The
// hidden width, top_k and cluster geometry are template parameters; the
// shared-expert add and the RMSNorm epilogue are optional, and kQuant adds the
// MXFP8 (FlashInfer's 128x4 layout) of the normalized row to the norm epilogue,
// kNvfp4 its per-token NVFP4 (FlashInfer's NVFP4QuantizePerTokenKernel, linear
// scale layout; nvfp4_quant_row).
//
// `idx == -1` marks a dropped slot (EP: the token was routed to an expert that
// is not local) and contributes nothing. Accumulation is fp32 and the bf16
// rounding points are exactly the unfused path's (moe_runner/flashinfer_trtllm.py
// finalize -> `shared.add_(routed)` -> fp32-accumulating bf16 all-reduce in rank
// order), so the kNorm=false result is bit-identical to it.
//
// The rank-local finalize never materializes in global memory: each thread
// computes one 16B vector of it and pushes it straight into every peer's push
// slot with unicast `st.relaxed.sys` stores, so no multicast mapping is needed.
//
// Push-plane protocol (see include/sgl_kernel/distributed/communicator.cuh):
//   * every rank owns 2 phases x kWorldSize slots of `slot_bytes`; a round
//     uses phase `counter & 1`, producer r writes slot r of every peer, the
//     consumer polls its own kWorldSize slots until no +0.0 marker remains,
//     reduces, and restores the +0.0 markers before it exits;
//   * +0.0 payload words are remapped to -0.0 (numerically identical) so a
//     written word is never 0 and `word == 0` means "not arrived yet";
//   * the generic push kernel owns one phase counter per block; this kernel
//     uses one per row cluster (flipped by the cluster's leader block after a
//     cluster barrier) plus a trailing "bumper" cluster that flips every
//     remaining one, so the whole array keeps one parity and both kernel
//     families can share the plane;
//   * every rank must call with the same num_tokens / hidden / top_k / epilogue:
//     slots are addressed by 16B vector index of the [T, hidden] row view.
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/communicator.cuh>
#include <sgl_kernel/distributed/ptx.cuh>

#include <tvm/ffi/extra/stl.h>

#include <cooperative_groups.h>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace sglang {

using device::distributed::PushWorkSpace;
using host::distributed::CommunicatorRef;

/// One 16B staging vector (8 bf16) viewed as the 4 u32 words the lamport marker
/// protocol tests, matching the generic push kernel's `LamportTrait<T, 8, 4>`.
using Lamport = device::distributed::LamportTrait<bf16_t, 8, /*kAtom=*/4>;
using StageVec = device::AlignedVector<bf16x2_t, 4>;

SGL_DEVICE void barrier_cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

SGL_DEVICE void barrier_cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

template <uint32_t kWorldSize, typename WeightT>
struct MoeFinalizeAllReduceParams {
  bf16_t* out;                         // [num_tokens, kHiddenDim], output-only
  const bf16_t* gemm2;                 // [P, kHiddenDim], permuted / padded rows
  const int32_t* idx;                  // [num_tokens * kTopK], -1 = dropped slot
  const WeightT* weights;              // [num_tokens, kTopK], scaling already folded in
  const bf16_t* shared;                // [num_tokens, kHiddenDim] (kHasShared only)
  const float* shared_gate = nullptr;  // Optional FP32 sigmoid, one per token.
  const bf16_t* norm_weight;           // [kHiddenDim] (kNorm only)
  float norm_eps;                      // kNorm only
  // Caller's promise that everything read before the PDL wait is complete when
  // the predecessor merely *triggers*: no all-reduce on this plane right before
  // it, and the routing metadata's producers finished (PDL completion is not
  // transitive through early-triggering kernels). False (the default) waits first.
  bool prefetch_metadata;
  uint32_t rank;
  uint32_t num_tokens;
  uint32_t num_push_counters;  // full counter array size (bumper range end)
  PushWorkSpace<kWorldSize> ws;
  bf16_t* mhc_out = nullptr;
  const bf16_t* residual = nullptr;
  const float* post = nullptr;
  const float* comb = nullptr;
  const float* pre = nullptr;
  bf16_t* normalized = nullptr;
  fp8_e4m3_t* quantized = nullptr;
  uint8_t* scales = nullptr;
  // kNorm only: residual_out = bf16(reduced + add_residual), normalized in fp32
  const bf16_t* add_residual = nullptr;
  bf16_t* residual_out = nullptr;
  float norm_weight_offset = 0.0f;  // 1.0 for Gemma RMSNorm
  // kNvfp4 only (nvfp4_quant_row): the per-token NVFP4 of the normalized row
  uint8_t* fp4 = nullptr;                   // [num_tokens, kHiddenDim / 2] packed E2M1 codes
  uint8_t* fp4_scales = nullptr;            // [num_tokens * kHiddenDim / 16] E4M3, linear layout
  float* token_scales = nullptr;            // [num_tokens]
  const float* global_scale_inv = nullptr;  // [1]
  // MXFP8 form (kQuant, not mHC) only: every CTA signals the dependent grid's launch
  // (griddepcontrol.launch_dependents) at its entry, before its PDL wait, instead of
  // after the epilogue. Caller's promise: the dependent reads nothing before its own
  // griddepcontrol.wait that this kernel or a kernel before it writes (FlashInfer's
  // low-latency MXFP8 GEMM, say, which loads only its weights there; they then
  // overlap this kernel). False (the default) keeps the end trigger only.
  bool trigger_at_entry = false;
};

template <uint32_t kHiddenDim, uint32_t kWorldSize, typename WeightT>
SGL_DEVICE void mhc_quant_vec(
    const MoeFinalizeAllReduceParams<kWorldSize, WeightT>& params,
    const StageVec& value,
    uint32_t token,
    uint32_t hvec) {
  using namespace device;
  fp32x2_t v[4];
  float amax = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    v[j] = cast<fp32x2_t>(value[j]);
    amax = fmaxf(amax, fmaxf(fabsf(v[j].x), fabsf(v[j].y)));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 4));
  const float normalized = amax * (1.0f / 448.0f);
  const uint32_t bits = __float_as_uint(normalized);
  const uint32_t exponent = (bits >> 23) & 255;
  const uint32_t mantissa = bits & 0x7fffff;
  const bool bump = mantissa != 0 && !(exponent == 0 && mantissa <= 0x400000);
  const uint32_t sf = normalized <= 0 ? 0 : min(exponent + uint32_t(bump), 254u);
  const float inv_scale = __uint_as_float(sf == 0 ? 0 : (254 - sf) << 23);
  AlignedVector<fp8x2_e4m3_t, 4> q;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    q[j] = cast<fp8x2_e4m3_t>(
        fp32x2_t{fminf(fmaxf(v[j].x * inv_scale, -448.0f), 448.0f), fminf(fmaxf(v[j].y * inv_scale, -448.0f), 448.0f)});
  }
  q.store(params.quantized + static_cast<int64_t>(token) * kHiddenDim, hvec);
  if (hvec % 4 == 0) {
    const uint32_t g = hvec / 4;
    const uint32_t off = (g / 4) * 512 + ((token % 32) * 4 + (token / 32) % 4) * 4 + g % 4;
    params.scales[off] = sf;
  }
}

/// MXFP8 of one 16B vector (8 bf16) of a normalized row, by FlashInfer's rule
/// (mxfp8_quantize, cute-dsl backend, 128x4 scales; quantization_cute_dsl_utils.py):
/// the 4 lanes holding a 32-element group share its amax (PTX max, which skips NaN;
/// lane xor 1, then xor 2), the UE8M0 scale is the biased exponent of amax / 448
/// rounded up (float_to_ue8m0_fast: 0 for amax 0, at most 254), and each value is
/// multiplied by 2^(127 - scale) (ue8m0_to_inv_scale_fast), clamped to [-448, 448]
/// (min first, so NaN becomes 448 as there) and rounded to nearest e4m3. The scale
/// byte goes to the 128x4 swizzle of a single 128-row tile; the caller zeros the
/// tile's padding rows. `hvec` is the vector's index within its row. The mHC entry
/// keeps mhc_quant_vec, which differs only on NaN: its amax starts at 0 and its
/// clamp takes the max first, so NaN becomes -448 and an all-NaN group scale 0.
template <uint32_t kHiddenDim, uint32_t kWorldSize, typename WeightT>
SGL_DEVICE void mxfp8_quant_vec(
    const MoeFinalizeAllReduceParams<kWorldSize, WeightT>& params,
    const StageVec& value,
    uint32_t token,
    uint32_t hvec) {
  using namespace device;
  fp32x2_t v[4];
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    v[j] = cast<fp32x2_t>(value[j]);
  }
  // seeded with the group's own values, so an all-NaN group keeps a NaN amax
  float amax = fmaxf(fabsf(v[0].x), fabsf(v[0].y));
#pragma unroll
  for (uint32_t j = 1; j < 4; ++j) {
    amax = fmaxf(amax, fmaxf(fabsf(v[j].x), fabsf(v[j].y)));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2, 4));
  const float normalized = amax * (1.0f / 448.0f);
  const uint32_t bits = __float_as_uint(normalized);
  const uint32_t exponent = (bits >> 23) & 255;
  const uint32_t mantissa = bits & 0x7fffff;
  const bool bump = mantissa != 0 && !(exponent == 0 && mantissa <= 0x400000);
  const uint32_t sf = normalized <= 0 ? 0 : min(exponent + uint32_t(bump), 254u);
  const float inv_scale = __uint_as_float(sf == 0 ? 0 : (254 - sf) << 23);
  AlignedVector<fp8x2_e4m3_t, 4> q;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    q[j] = cast<fp8x2_e4m3_t>(
        fp32x2_t{fmaxf(fminf(v[j].x * inv_scale, 448.0f), -448.0f), fmaxf(fminf(v[j].y * inv_scale, 448.0f), -448.0f)});
  }
  q.store(params.quantized + static_cast<int64_t>(token) * kHiddenDim, hvec);
  if (hvec % 4 == 0) {
    const uint32_t g = hvec / 4;
    const uint32_t off = (g / 4) * 512 + ((token % 32) * 4 + (token / 32) % 4) * 4 + g % 4;
    params.scales[off] = sf;
  }
}

// --- the per-token NVFP4 of a normalized row (kNvfp4) -------------------------
// FlashInfer's NVFP4QuantizePerTokenKernel (flashinfer/quantization/kernels/
// nvfp4_quantize.py, 0.7.0) of a bf16 row, in the configuration SGLang's routed MoE
// runs it in (fast math: FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH unset; no 4over6:
// FLASHINFER_NVFP4_4OVER6 unset; linear scale layout), bit for bit: the same PTX in
// the same order, every instruction FlashInfer writes as inline PTX copied from
// there (cute_dsl/fp4_common.py, quantization/quantization_cute_dsl_utils.py).
// FlashInfer runs one CTA per row; here the row's cluster does, each thread with the
// 16B vectors (8 values) of the row it holds:
//   row_amax = max(0, |x|) over the row: max.f32, which returns the other operand of
//     a NaN, so the order is free (FlashInfer: per-thread, warp, block reductions);
//   row_amax == 0: per_token = 0, encode = FLT_MAX; else per_token = row_amax *
//     global_scale_inv, encode = rcp.approx.ftz(per_token)  (_row_scales);
//   per 16 values, two adjacent lanes' vectors: block_max = the max.bf16x2 tree over
//     |x| (bfloat2_max_abs_8: words 0-3 of the first vector, 4-7 of the second, the
//     first lane's half first) then max.f32 of its halves; scale = encode *
//     (block_max * rcp.approx.ftz(6)), to e4m3 (cvt.rn.satfinite); the output scale
//     rcp.approx.ftz(e4m3 * rcp.approx.ftz(encode)), 0 for a zero scale
//     (nvfp4_compute_output_scale); each value times it (mul.f32) to e2m1
//     (cvt.rn.satfinite.e2m1x2).
// The compare and the three multiplies FlashInfer leaves to CuTe-DSL (plain Python
// `==` and `*`) are setp.eq / mul.rn here, with .ftz under kFtz: which one the DSL
// emits is not in the source, and the two differ only on rows whose amax is below
// about 3e-32, so a caller probes FlashInfer's kernel for it (Nvfp4RowQuantKernel).
namespace nvfp4 {

template <bool kFtz>
SGL_DEVICE float mul(float a, float b) {
  float r;
  if constexpr (kFtz) {
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  } else {
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  }
  return r;
}

template <bool kFtz>
SGL_DEVICE bool is_zero(float a) {
  uint32_t r;
  if constexpr (kFtz) {
    asm("{\n.reg .pred p;\nsetp.eq.ftz.f32 p, %1, 0f00000000;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(r) : "f"(a));
  } else {
    asm("{\n.reg .pred p;\nsetp.eq.f32 p, %1, 0f00000000;\nselp.u32 %0, 1, 0, p;\n}" : "=r"(r) : "f"(a));
  }
  return r != 0;
}

// fp4_common.py rcp_approx_ftz
SGL_DEVICE float rcp_approx_ftz(float a) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}

// fp4_common.py fmax_f32
SGL_DEVICE float max_f32(float a, float b) {
  float r;
  asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// fp4_common.py bfloat2_hmax2
SGL_DEVICE uint32_t max_bf16x2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("max.bf16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

// quantization_cute_dsl_utils.py bfloat2_hmax_reduce_to_f32
SGL_DEVICE float hmax_to_f32(uint32_t x) {
  float r;
  asm("{\n"
      ".reg .b32 lo, hi;\n"
      ".reg .f32 f0, f1;\n"
      "and.b32 lo, %1, 0xFFFF;\n"
      "shr.b32 hi, %1, 16;\n"
      "shl.b32 lo, lo, 16;\n"
      "shl.b32 hi, hi, 16;\n"
      "mov.b32 f0, lo;\n"
      "mov.b32 f1, hi;\n"
      "max.f32 %0, f0, f1;\n"
      "}"
      : "=f"(r)
      : "r"(x));
  return r;
}

// fp4_common.py cvt_f32_to_e4m3: the e4m3 code in the low byte
SGL_DEVICE uint32_t cvt_f32_to_e4m3(float a) {
  uint32_t r;
  asm("{\n"
      ".reg .b16 fp8_pair;\n"
      ".reg .f32 zero;\n"
      "mov.f32 zero, 0f00000000;\n"
      "cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, %1;\n"
      "cvt.u32.u16 %0, fp8_pair;\n"
      "}"
      : "=r"(r)
      : "f"(a));
  return r;
}

// fp4_common.py nvfp4_compute_output_scale
SGL_DEVICE float output_scale(uint32_t fp8, float global_scale) {
  float r;
  asm("{\n"
      ".reg .pred p_zero;\n"
      ".reg .b16 fp8_pair;\n"
      ".reg .b32 h2_32;\n"
      ".reg .b16 h_lo, h_hi;\n"
      ".reg .f32 scale_f32, rcp_gs, product, result;\n"
      "cvt.u16.u32 fp8_pair, %1;\n"
      "cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;\n"
      "mov.b32 {h_lo, h_hi}, h2_32;\n"
      "cvt.f32.f16 scale_f32, h_lo;\n"
      "rcp.approx.ftz.f32 rcp_gs, %2;\n"
      "mul.f32 product, scale_f32, rcp_gs;\n"
      "rcp.approx.ftz.f32 result, product;\n"
      "setp.eq.f32 p_zero, scale_f32, 0f00000000;\n"
      "selp.f32 %0, 0f00000000, result, p_zero;\n"
      "}"
      : "=f"(r)
      : "r"(fp8), "f"(global_scale));
  return r;
}

// quantization_cute_dsl_utils.py bfloat2_to_float2_scaled: a bf16 pair to f32, each
// times `scale`
SGL_DEVICE void scaled_pair(uint32_t bf2, float scale, float& lo, float& hi) {
  asm("{\n"
      ".reg .b32 lo, hi;\n"
      ".reg .f32 f0, f1;\n"
      "and.b32 lo, %2, 0xFFFF;\n"
      "shr.b32 hi, %2, 16;\n"
      "shl.b32 lo, lo, 16;\n"
      "shl.b32 hi, hi, 16;\n"
      "mov.b32 f0, lo;\n"
      "mov.b32 f1, hi;\n"
      "mul.f32 %0, f0, %3;\n"
      "mul.f32 %1, f1, %3;\n"
      "}"
      : "=f"(lo), "=f"(hi)
      : "r"(bf2), "f"(scale));
}

// quantization_cute_dsl_utils.py bfloat2x8_to_e2m1x16_packed, one half: the 8 values
// of 4 bf16 pairs times `scale` to 8 E2M1 codes (cvt_e2m1x8_f32; value 0 in the low
// nibble of byte 0)
SGL_DEVICE uint32_t e2m1x8(const uint32_t (&w)[4], float scale) {
  float s[8];
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    scaled_pair(w[i], scale, s[2 * i], s[2 * i + 1]);
  }
  uint32_t r;
#if SGL_ARCH_BLACKWELL_OR_GREATER
  asm("{\n"
      ".reg .b8 byte0, byte1, byte2, byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(r)
      : "f"(s[0]), "f"(s[1]), "f"(s[2]), "f"(s[3]), "f"(s[4]), "f"(s[5]), "f"(s[6]), "f"(s[7]));
#else
  // Never compiled into a kernel: nvfp4_quant_row, the only caller, fails the build
  // without SGL_ARCH_BLACKWELL_OR_GREATER (and require_nvfp4_arch declines first)
  r = 0;
  __trap();
#endif
  return r;
}

}  // namespace nvfp4

/// FlashInfer's per-token NVFP4 quantize (above) of row `token`, of which this thread
/// holds the kVecs 16B vectors `vec0 + v * kBlockSize` (`row[v]`). Every thread of the
/// row's cluster calls it: the row's amax goes through `smem_amax` (a cluster
/// barrier). Writes the vector's 4 code bytes (`fp4`: [T, kHiddenDim / 2]), the
/// block's e4m3 scale from the first lane of each pair (`scales`: [T * kHiddenDim /
/// 16], linear), and the per-token scale from the cluster's thread 0 (`token_scale`).
template <uint32_t kHiddenDim, uint32_t kClusterSize, uint32_t kBlockSize, uint32_t kVecs, bool kFtz>
SGL_DEVICE void nvfp4_quant_row(
    const StageVec (&row)[kVecs],
    uint32_t token,
    uint32_t vec0,
    uint32_t cluster_rank,
    float (&smem_amax)[kClusterSize][kBlockSize / device::kWarpThreads],
    uint8_t* fp4,
    uint8_t* scales,
    float* token_scale,
    const float* global_scale_inv) {
  namespace cg = cooperative_groups;
  using namespace device;
  static_assert(kHiddenDim % 16 == 0 && kBlockSize % kWarpThreads == 0 && kClusterSize <= kWarpThreads);
  // e2m1x8's conversions need SM100+ and a CUDA 12.9+ nvcc (the macro); elsewhere the
  // build fails here, as a module that cannot load, instead of a kernel that traps
  static_assert(
      SGL_ARCH_BLACKWELL_OR_GREATER || host::dependent_false_v<std::integral_constant<uint32_t, kHiddenDim>>,
      "the NVFP4 epilogue needs SGL_ARCH_BLACKWELL_OR_GREATER (SM100+, CUDA 12.9+)");
  constexpr uint32_t kNumWarps = kBlockSize / kWarpThreads;
  constexpr uint32_t kAbs = 0x7fff7fffu;  // bfloat2_habs2
  uint32_t bits[kVecs][4];
  float amax = 0.0f;
#pragma unroll
  for (uint32_t v = 0; v < kVecs; ++v) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      bits[v][j] = reinterpret_cast<const uint32_t*>(row[v].data())[j];
      amax = nvfp4::max_f32(amax, nvfp4::hmax_to_f32(bits[v][j] & kAbs));
    }
  }
#pragma unroll
  for (uint32_t offset = kWarpThreads / 2; offset > 0; offset /= 2) {
    amax = nvfp4::max_f32(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
  }
  // every warp's amax to every block of the cluster, as the norm's sum of squares
  const uint32_t lane = threadIdx.x % kWarpThreads;
  const uint32_t warp = threadIdx.x / kWarpThreads;
  const auto cluster = cg::this_cluster();
  if (lane < kClusterSize) {
    *cluster.map_shared_rank(&smem_amax[cluster_rank][warp], lane) = amax;
  }
  cluster.sync();
  float row_amax = 0.0f;
#pragma unroll
  for (uint32_t c = 0; c < kClusterSize; ++c) {
#pragma unroll
    for (uint32_t w = 0; w < kNumWarps; ++w) {
      row_amax = nvfp4::max_f32(row_amax, smem_amax[c][w]);
    }
  }
  float per_token = 0.0f;
  float encode = __uint_as_float(0x7f7fffffu);  // FLOAT32_MAX
  if (!nvfp4::is_zero<kFtz>(row_amax)) {
    per_token = nvfp4::mul<kFtz>(row_amax, *global_scale_inv);
    encode = nvfp4::rcp_approx_ftz(per_token);
  }
  if (cluster_rank == 0 && threadIdx.x == 0) token_scale[token] = per_token;
  const float fp4_max_rcp = nvfp4::rcp_approx_ftz(6.0f);
#pragma unroll
  for (uint32_t v = 0; v < kVecs; ++v) {
    const uint32_t vec = vec0 + v * kBlockSize;
    const bool first = vec % 2 == 0;
    const uint32_t mine = nvfp4::max_bf16x2(
        nvfp4::max_bf16x2(bits[v][0] & kAbs, bits[v][1] & kAbs),
        nvfp4::max_bf16x2(bits[v][2] & kAbs, bits[v][3] & kAbs));
    const uint32_t other = __shfl_xor_sync(0xffffffffu, mine, 1);
    const float block_max = nvfp4::hmax_to_f32(first ? nvfp4::max_bf16x2(mine, other) : nvfp4::max_bf16x2(other, mine));
    const float scale = nvfp4::mul<kFtz>(encode, nvfp4::mul<kFtz>(block_max, fp4_max_rcp));
    const uint32_t scale_fp8 = nvfp4::cvt_f32_to_e4m3(scale);
    const uint32_t codes = nvfp4::e2m1x8(bits[v], nvfp4::output_scale(scale_fp8, encode));
    reinterpret_cast<uint32_t*>(fp4 + static_cast<int64_t>(token) * (kHiddenDim / 2))[vec] = codes;
    if (first) {
      scales[static_cast<int64_t>(token) * (kHiddenDim / 16) + vec / 2] = static_cast<uint8_t>(scale_fp8 & 0xffu);
    }
  }
}

/// HC=4 post mixing of an already BF16-rounded all-reduce vector, in
/// mhc_post_split_h's order: round comb[0]*residual[0], FMA post*x, then the
/// remaining three residual streams.
template <uint32_t kHiddenDim, bool kCollapse = false, uint32_t kWorldSize, typename WeightT>
SGL_DEVICE StageVec mhc_post_vec(
    const MoeFinalizeAllReduceParams<kWorldSize, WeightT>& params, const StageVec& red, uint32_t token, uint32_t hvec) {
  using namespace device;
  StageVec residual[4];
  fp32x2_t collapsed[4] = {};
#pragma unroll
  for (uint32_t c = 0; c < 4; ++c) {
    residual[c].load(params.residual + (static_cast<int64_t>(token) * 4 + c) * kHiddenDim, hvec);
  }
#pragma unroll
  for (uint32_t c = 0; c < 4; ++c) {
    const float post = params.post[token * 4 + c];
    float comb[4];
#pragma unroll
    for (uint32_t r = 0; r < 4; ++r)
      comb[r] = params.comb[token * 16 + r * 4 + c];
    StageVec out;
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      const auto x = cast<fp32x2_t>(red[j]);
      const auto r0 = cast<fp32x2_t>(residual[0][j]);
      fp32x2_t acc{fmaf(post, x.x, __fmul_rn(comb[0], r0.x)), fmaf(post, x.y, __fmul_rn(comb[0], r0.y))};
#pragma unroll
      for (uint32_t r = 1; r < 4; ++r) {
        const auto v = cast<fp32x2_t>(residual[r][j]);
        acc.x = fmaf(comb[r], v.x, acc.x);
        acc.y = fmaf(comb[r], v.y, acc.y);
      }
      out[j] = cast<bf16x2_t>(acc);
      if constexpr (kCollapse) {
        const auto rounded = cast<fp32x2_t>(out[j]);
        const float pre = params.pre[token * 4 + c];
        collapsed[j].x = fmaf(rounded.x, pre, collapsed[j].x);
        collapsed[j].y = fmaf(rounded.y, pre, collapsed[j].y);
      }
    }
    out.store(params.mhc_out + (static_cast<int64_t>(token) * 4 + c) * kHiddenDim, hvec);
  }
  StageVec result;
  if constexpr (kCollapse) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j)
      result[j] = cast<bf16x2_t>(collapsed[j]);
  }
  return result;
}

/// Row geometry: kVecs 16B vectors per thread, one cluster per row, so the block
/// size follows from the hidden width, the cluster size (the tuning knob) and kVecs.
template <uint32_t kHiddenDim, uint32_t kClusterSize, uint32_t kVecs = 1>
struct RowClusterTrait {
  static constexpr uint32_t kRowVecs = kHiddenDim / 8;                       // 16B vectors per row
  static constexpr uint32_t kBlockSize = kRowVecs / (kClusterSize * kVecs);  // threads per block
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static_assert(kHiddenDim % 8 == 0, "hidden must be a whole number of 16B vectors");
  static_assert(1 <= kClusterSize && kClusterSize <= 8, "portable cluster sizes only");
  static_assert(kRowVecs % (kClusterSize * kVecs) == 0, "cluster size must divide the row's vector count");
  static_assert(kBlockSize % device::kWarpThreads == 0, "block must be whole warps");
  static_assert(kBlockSize <= 1024, "block too large: raise the cluster size");
};

// --- stage 1: the deferred finalize of one 16B vector ------------------------
// The shared-expert vector is loaded first so that load is in flight while the
// routing rows and the kTopK gathers are fetched.
template <
    uint32_t kHiddenDim,
    uint32_t kTopK,
    bool kHasShared,
    bool kUsePDL,
    uint32_t kWorldSize,
    typename WeightT,
    bool kGatedShared = false>
SGL_DEVICE StageVec
finalize_vec(const MoeFinalizeAllReduceParams<kWorldSize, WeightT>& params, uint32_t token, uint32_t hvec) {
  using namespace device;
  const auto* idx = params.idx + static_cast<int64_t>(token) * kTopK;
  const auto* weights = params.weights + static_cast<int64_t>(token) * kTopK;
  int32_t rows[kTopK];
  WeightT w[kTopK];
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    rows[k] = idx[k];
    w[k] = weights[k];
  }

  // delay PDL wait until here
  PDLWaitPrimary<kUsePDL>();

  StageVec shared_in;
  if constexpr (kHasShared) {
    shared_in.load(params.shared + static_cast<int64_t>(token) * kHiddenDim, hvec);
  }

  StageVec in[kTopK];
#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    if (rows[k] >= 0) in[k].load(params.gemm2 + static_cast<int64_t>(rows[k]) * kHiddenDim, hvec);
  }

  fp32x2_t acc[4];
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    acc[j] = fp32x2_t{0.0f, 0.0f};
  }

#pragma unroll
  for (uint32_t k = 0; k < kTopK; ++k) {
    if (rows[k] < 0) continue;
#if SGL_ARCH_BLACKWELL_OR_GREATER
    if constexpr (std::is_same_v<WeightT, bf16_t>) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        acc[j].x = math::fma_f32_bf16(in[k][j].x, w[k], acc[j].x);
        acc[j].y = math::fma_f32_bf16(in[k][j].y, w[k], acc[j].y);
      }
    } else
#endif
    {
      const auto w_fp32 = cast<fp32_t>(w[k]);
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const auto [x, y] = cast<fp32x2_t>(in[k][j]);
        acc[j].x = fmaf(x, w_fp32, acc[j].x);
        acc[j].y = fmaf(y, w_fp32, acc[j].y);
      }
    }
  }
  // Two deliberate roundings -- the routed combine, then the bf16 + bf16 add --
  // keep the staged vector bit-identical to the unfused rank-local result.
  StageVec out;
#pragma unroll
  for (uint32_t j = 0; j < 4; ++j) {
    if constexpr (kHasShared) {
      const auto routed = cast<fp32x2_t>(cast<bf16x2_t>(acc[j]));
      const auto sh = cast<fp32x2_t>(shared_in[j]);
      if constexpr (kGatedShared) {
        const float gate = params.shared_gate[token];
        out[j] = cast<bf16x2_t>(fp32x2_t{fmaf(gate, sh.x, routed.x), fmaf(gate, sh.y, routed.y)});
      } else {
        out[j] = cast<bf16x2_t>(fp32x2_t{routed.x + sh.x, routed.y + sh.y});
      }
    } else {
      out[j] = cast<bf16x2_t>(acc[j]);
    }
  }
  return out;
}

// kTopK == 0: no finalize, `gemm2` is the rank-local [T, kHiddenDim] partial sum.
template <uint32_t kHiddenDim, bool kUsePDL, uint32_t kWorldSize, typename WeightT>
SGL_DEVICE StageVec
plain_vec(const MoeFinalizeAllReduceParams<kWorldSize, WeightT>& params, uint32_t token, uint32_t hvec) {
  using namespace device;
  PDLWaitPrimary<kUsePDL>();
  StageVec in;
  in.load(params.gemm2 + static_cast<int64_t>(token) * kHiddenDim, hvec);
  return in;
}

// --- the kernel --------------------------------------------------------------
// Grid dim3(num_tokens [+ 1], kClusterSize) with the cluster along y, so
// blockIdx.x is the token row (and its phase counter) and blockIdx.y the rank
// inside the cluster. The extra cluster (blockIdx.x == num_tokens) is the
// bumper: it only flips the leftover counters [num_tokens, num_push_counters).
//
// kTwoShot: row t is owned by rank t % kWorldSize. Each rank
// pushes its partial only to the owner (slot `src = r`), the owner reduces and
// pushes the bf16 sum to slot `src = owner` of every other rank, and every rank
// runs the epilogue on that sum. A non-owner never receives a partial for the
// row, so the owner's slot in its workspace is free for the gathered sum. The
// owner keeps its own partial in registers, so its (r, r) slot stays unused.
template <
    uint32_t kWorldSize,
    uint32_t kHiddenDim,
    uint32_t kTopK,
    uint32_t kClusterSize,
    bool kUsePDL,
    bool kHasShared,
    bool kNorm,
    typename WeightT,
    bool kMhc = false,
    bool kQuant = false,
    bool kCollapse = false,
    bool kTwoShot = false,
    uint32_t kVecs = 1,
    bool kGatedShared = false,
    bool kNvfp4 = false,
    bool kNvfp4Ftz = false>
__global__ __launch_bounds__(RowClusterTrait<kHiddenDim, kClusterSize, kVecs>::kBlockSize)
    __cluster_dims__(1, kClusterSize, 1) void moe_finalize_all_reduce_kernel(
        const __grid_constant__ MoeFinalizeAllReduceParams<kWorldSize, WeightT> params) {
  namespace cg = cooperative_groups;
  using namespace device;
  using T = RowClusterTrait<kHiddenDim, kClusterSize, kVecs>;
  static_assert(!kTwoShot || !kMhc, "two-shot has no mHC epilogue");
  static_assert(kVecs == 1 || (kNorm && !kMhc), "several vectors per thread is a norm-epilogue path");
  // FlashInfer pads the scale columns to a multiple of 4; none here, so every byte is written
  static_assert(!kQuant || kHiddenDim % 128 == 0, "the MXFP8 epilogue needs whole 4-column scale groups");
  static_assert(!kNvfp4 || (!kMhc && kHiddenDim % 16 == 0), "the NVFP4 epilogue quantizes whole 16-value blocks");
  constexpr uint32_t kRowVecs = T::kRowVecs;
  constexpr uint32_t kBlockSize = T::kBlockSize;
  constexpr uint32_t kNumWarps = T::kNumWarps;

  const auto tx = threadIdx.x;
  const auto row_idx = blockIdx.x;
  const auto cluster_rank = blockIdx.y;
  // this thread's first vector within a row: cluster rank picks the block's chunk,
  // and vector v sits v * kBlockSize further so a warp's accesses stay contiguous
  const auto hvec = cluster_rank * kBlockSize * kVecs + tx;

  // trigger_at_entry: the dependent grid may launch once every CTA of this grid has
  // signalled or exited, so every CTA, the bumper cluster's included, signals first:
  // the dependent then starts while this grid waits on its predecessor and
  // all-reduces. It changes no store of this kernel and no order of its own: its
  // waits, stores and end trigger below are as without it. The dependent's
  // griddepcontrol.wait still returns only once this whole grid has completed and
  // its stores are visible (PTX ISA, griddepcontrol), so it reads every output of
  // this kernel, the ones stored after the trigger included, after its wait; the
  // end trigger then repeats the instruction in the same CTA, which the PTX ISA
  // defines as having no effect past the first.
  if constexpr (kQuant && !kMhc) {
    if (params.trigger_at_entry) PDLTriggerSecondary<kUsePDL>();
  }

  // Reading the epoch before the PDL wait can see a predecessor all-reduce
  // mid-flip on this plane; prefetch_metadata defers the wait to finalize_vec.
  if (!params.prefetch_metadata) PDLWaitPrimary<kUsePDL>();

  if (row_idx == params.num_tokens) {
    PDLWaitPrimary<kUsePDL>();
    if constexpr (kQuant) {
      // The SF tile is padded to 128 rows. Active rows are written by the norm
      // epilogue; the bumper zeros only the disjoint padding rows, as FlashInfer
      // does. One pass of the cluster's threads covers kRowVecs / kVecs bytes.
      for (uint32_t off = cluster_rank * kBlockSize + tx; off < (kHiddenDim / 32) * 128;
           off += kClusterSize * kBlockSize) {
        const uint32_t swizzled_row = (off % 512) / 4;
        const uint32_t row = swizzled_row / 4 + (swizzled_row % 4) * 32;
        if (row >= params.num_tokens) params.scales[off] = 0;
      }
    }
    if (cluster_rank == 0) {
      const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};
      __syncthreads();
      epoch.unsafe_flip_range(row_idx, params.num_push_counters);
    }
    return PDLTriggerSecondary<kUsePDL>();
  }

  // this cluster's epoch: the counter at blockIdx.x, one per row cluster
  // (every block of the cluster reads the same one)
  const auto epoch = distributed::PushEpoch<kWorldSize>{params.ws};
  const auto r = params.rank;
  const auto owner = row_idx % kWorldSize;
  const bool is_owner = !kTwoShot || owner == r;
  // my slot (`src = r`) inside every peer's workspace, and every peer's slot
  // inside mine (`dst = r`), for this epoch
  void* push_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    push_ptrs[i] = epoch.slot_ptr(/*dst=*/i, /*src=*/r);
  }

  // stage 1: finalize this row's vectors in registers and push them to every peer
  // (two-shot: to the row's owner only)
  StageVec vec[kVecs][kWorldSize];
  uint32_t vid[kVecs];
#pragma unroll
  for (uint32_t v = 0; v < kVecs; ++v) {
    vid[v] = row_idx * kRowVecs + hvec + v * kBlockSize;
    StageVec staged;
    if constexpr (kTopK == 0) {
      staged = plain_vec<kHiddenDim, kUsePDL>(params, row_idx, hvec + v * kBlockSize);
    } else {
      staged = finalize_vec<kHiddenDim, kTopK, kHasShared, kUsePDL, kWorldSize, WeightT, kGatedShared>(
          params, row_idx, hvec + v * kBlockSize);
    }
    Lamport::clear_pos_zero(staged.data());
    if constexpr (kTwoShot) {
      if (is_owner) {
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          if (i == r) vec[v][i] = staged;
        }
      } else {
        ptx::st_relaxed_16B(staged, epoch.slot_ptr(/*dst=*/owner, /*src=*/r), vid[v]);
      }
    } else {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        ptx::st_relaxed_16B(staged, push_ptrs[i], vid[v]);
      }
    }
  }

  // ensure epoch is consumed, so flipping it won't lead to error
  if constexpr (!kNorm) barrier_cluster_arrive_relaxed();

  // stage 2: poll own slots, reduce across ranks, [norm], write, reset markers
  void* poll_ptrs[kWorldSize];
#pragma unroll
  for (uint32_t i = 0; i < kWorldSize; ++i) {
    poll_ptrs[i] = epoch.slot_ptr(/*dst=*/r, /*src=*/i);
  }
  if (is_owner) {
    do {
      bool has_zero = false;
#pragma unroll
      for (uint32_t v = 0; v < kVecs; ++v) {
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          if (!kTwoShot || i != r) ptx::ld_relaxed_16B(vec[v][i], poll_ptrs[i], vid[v]);
        }
      }
#pragma unroll
      for (uint32_t v = 0; v < kVecs; ++v) {
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          // the producer remapped +0.0 words, so a written word is never 0:
          // word == 0 <=> the slot still holds the empty marker
          if (!kTwoShot || i != r) has_zero |= Lamport::has_pos_zero(vec[v][i].data());
        }
      }
      if (!has_zero) break;
    } while (true);
  }

  StageVec red[kVecs];
  if constexpr (kTwoShot) {
    if (is_owner) {
#pragma unroll
      for (uint32_t v = 0; v < kVecs; ++v) {
        red[v] = reduce_vec(vec[v]);
        // the same -0.0 remap the peers receive keeps every rank bit-identical
        Lamport::clear_pos_zero(red[v].data());
#pragma unroll
        for (uint32_t i = 0; i < kWorldSize; ++i) {
          if (i != r) ptx::st_relaxed_16B(red[v], push_ptrs[i], vid[v]);
        }
      }
    } else {
      const auto gather_ptr = epoch.slot_ptr(/*dst=*/r, /*src=*/owner);
      bool has_zero;
      do {
        has_zero = false;
#pragma unroll
        for (uint32_t v = 0; v < kVecs; ++v) {
          ptx::ld_relaxed_16B(red[v], gather_ptr, vid[v]);
          has_zero |= Lamport::has_pos_zero(red[v].data());
        }
      } while (has_zero);
    }
  } else {
#pragma unroll
    for (uint32_t v = 0; v < kVecs; ++v) {
      red[v] = reduce_vec(vec[v]);
    }
  }
  if constexpr (!kNorm) {
    ptx::st_global_16B(red[0], params.out, vid[0]);
    if constexpr (kMhc) {
      if constexpr (kCollapse) {
        const auto combined = mhc_post_vec<kHiddenDim, true>(params, red[0], row_idx, hvec);
        ptx::st_global_16B(combined, params.normalized, vid[0]);
      } else {
        mhc_post_vec<kHiddenDim>(params, red[0], row_idx, hvec);
      }
    }
    // ensure epoch is consumed, so flipping it won't lead to error
    barrier_cluster_wait();
  } else {
    __shared__ float smem_sq[kClusterSize][kNumWarps];
    if constexpr (kMhc) {
      ptx::st_global_16B(red[0], params.out, vid[0]);
      red[0] = mhc_post_vec<kHiddenDim, true>(params, red[0], row_idx, hvec);
    }
    const auto cluster = cg::this_cluster();
    fp32x2_t acc[kVecs][4];
    float sq = 0.0f;
#pragma unroll
    for (uint32_t v = 0; v < kVecs; ++v) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        acc[v][j] = cast<fp32x2_t>(red[v][j]);
      }
      if (params.add_residual != nullptr) {
        StageVec res;
        res.load(params.add_residual + static_cast<int64_t>(row_idx) * kHiddenDim, hvec + v * kBlockSize);
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          const auto [ra, rb] = cast<fp32x2_t>(res[j]);
          acc[v][j] = fp32x2_t{acc[v][j].x + ra, acc[v][j].y + rb};
          res[j] = cast<bf16x2_t>(acc[v][j]);
        }
        res.store(params.residual_out + static_cast<int64_t>(row_idx) * kHiddenDim, hvec + v * kBlockSize);
      }
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        sq = fmaf(acc[v][j].x, acc[v][j].x, sq);
        sq = fmaf(acc[v][j].y, acc[v][j].y, sq);
      }
    }
    sq = warp::reduce_sum(sq);
    const auto lane = tx % kWarpThreads;
    const auto warp = tx / kWarpThreads;
    if (lane < kClusterSize) {
      *cluster.map_shared_rank(&smem_sq[cluster_rank][warp], lane) = sq;
    }
    cluster.sync();
    float total = 0.0f;
#pragma unroll
    for (uint32_t c = 0; c < kClusterSize; ++c) {
#pragma unroll
      for (uint32_t wp = 0; wp < kNumWarps; ++wp) {
        total += smem_sq[c][wp];
      }
    }
    const auto factor = math::rsqrt(total / static_cast<float>(kHiddenDim) + params.norm_eps);
    const auto off = params.norm_weight_offset;
    [[maybe_unused]] StageVec normed[kVecs];
#pragma unroll
    for (uint32_t v = 0; v < kVecs; ++v) {
      StageVec w;
      w.load(params.norm_weight, hvec + v * kBlockSize);
      StageVec out;
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        const auto [wa, wb] = cast<fp32x2_t>(w[j]);
        out[j] = cast<bf16x2_t>(fp32x2_t{acc[v][j].x * factor * (wa + off), acc[v][j].y * factor * (wb + off)});
      }
      ptx::st_global_16B(out, kMhc ? params.normalized : params.out, vid[v]);
      // the mHC entry keeps its own NaN rule (mhc_quant_vec, one vector per thread)
      if constexpr (kQuant) {
        if constexpr (kMhc) {
          mhc_quant_vec<kHiddenDim>(params, out, row_idx, hvec);
        } else {
          mxfp8_quant_vec<kHiddenDim>(params, out, row_idx, hvec + v * kBlockSize);
        }
      }
      if constexpr (kNvfp4) normed[v] = out;
    }
    if constexpr (kNvfp4) {
      __shared__ float smem_amax[kClusterSize][kNumWarps];
      nvfp4_quant_row<kHiddenDim, kClusterSize, kBlockSize, kVecs, kNvfp4Ftz>(
          normed,
          row_idx,
          hvec,
          cluster_rank,
          smem_amax,
          params.fp4,
          params.fp4_scales,
          params.token_scales,
          params.global_scale_inv);
    }
  }
  PDLTriggerSecondary<kUsePDL>();

  // re-establish the empty markers for the next same-phase round
  StageVec zero_vec;
  Lamport::fill_pos_zero(zero_vec.data());
#pragma unroll
  for (uint32_t v = 0; v < kVecs; ++v) {
    if (is_owner) {
#pragma unroll
      for (uint32_t i = 0; i < kWorldSize; ++i) {
        if (!kTwoShot || i != r) ptx::st_global_16B(zero_vec, poll_ptrs[i], vid[v]);
      }
    } else {
      ptx::st_global_16B(zero_vec, epoch.slot_ptr(/*dst=*/r, /*src=*/owner), vid[v]);
    }
  }

  if (cluster_rank == 0) epoch.flip();
}

// --- host --------------------------------------------------------------------

template <
    uint32_t kWorldSize,
    uint32_t kHiddenDim,
    uint32_t kTopK,
    uint32_t kClusterSize,
    bool kUsePDL,
    typename WeightT,
    bool kMhc = false,
    bool kQuant = false,
    bool kCollapse = false,
    bool kGatedShared = false,
    bool kNvfp4 = false,
    bool kNvfp4Ftz = false>
struct MoeFinalizeAllReduceKernel {
 private:
  static_assert(std::is_same_v<WeightT, bf16_t> || std::is_same_v<WeightT, fp32_t>);
  using TensorView = tvm::ffi::TensorView;
  using Params = MoeFinalizeAllReduceParams<kWorldSize, WeightT>;
  using Trait = RowClusterTrait<kHiddenDim, kClusterSize>;

  template <bool kHasShared, bool kNorm, bool kTwoShot = false, uint32_t kVecs = 1>
  static constexpr auto kernel = moe_finalize_all_reduce_kernel<
      kWorldSize,
      kHiddenDim,
      kTopK,
      kClusterSize,
      kUsePDL,
      kHasShared,
      kNorm,
      WeightT,
      kMhc,
      kQuant,
      kCollapse,
      kTwoShot,
      kVecs,
      kGatedShared,
      kNvfp4,
      kNvfp4Ftz>;
  static constexpr bool kTwoVecsFit = Trait::kRowVecs % (kClusterSize * 2 * device::kWarpThreads) == 0;

 public:
  /// out = [allreduce over ranks of] finalize(gemm2_out, idx, weights) [+ shared] [-> RMSNorm(norm_weight, eps)].
  /// `out` ([T, kHiddenDim] bf16) is output-only. `shared_output` and `norm_weight`
  /// select the epilogue at runtime (four kernel instantiations per module).
  static void
  run(CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      std::optional<TensorView> norm_weight,
      double eps,
      bool prefetch_metadata) {
    static_assert(!kMhc && !kQuant && !kNvfp4);
    run_impl(
        ref,
        out,
        gemm2_out,
        permuted_idx,
        expert_weights,
        shared_output,
        norm_weight,
        eps,
        prefetch_metadata,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt);
  }

  // Preserve routed BF16 rounding, then apply the FP32 shared gate with FMA.
  static void run_shared_gate(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      TensorView shared_output,
      TensorView shared_gate) {
    static_assert(kGatedShared && !kMhc && !kNvfp4);
    run_impl(
        ref,
        out,
        gemm2_out,
        permuted_idx,
        expert_weights,
        shared_output,
        std::nullopt,
        0.0,
        false,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        0.0,
        false,
        1,
        shared_gate);
  }

  /// run() with the norm epilogue preceded by a residual add:
  /// residual_out = bf16(allreduce(...) + add_residual), out = RMSNorm(fp32 sum).
  /// `residual_out` may alias `add_residual`; `weight_offset` is 1.0 for Gemma RMSNorm.
  static void run_residual_norm(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      TensorView norm_weight,
      double eps,
      TensorView add_residual,
      TensorView residual_out,
      double weight_offset,
      bool two_shot,
      int64_t vecs_per_thread) {
    static_assert(!kMhc && !kQuant && !kNvfp4 && kTopK != 0);
    run_impl(
        ref,
        out,
        gemm2_out,
        permuted_idx,
        expert_weights,
        shared_output,
        norm_weight,
        eps,
        false,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        add_residual,
        residual_out,
        weight_offset,
        two_shot,
        vecs_per_thread);
  }

  /// kTopK == 0: out = allreduce(input) [-> residual add] [-> RMSNorm], no finalize.
  static void run_plain(
      CommunicatorRef ref,
      TensorView out,
      TensorView input,
      std::optional<TensorView> norm_weight,
      double eps,
      std::optional<TensorView> add_residual,
      std::optional<TensorView> residual_out,
      double weight_offset,
      bool two_shot,
      int64_t vecs_per_thread) {
    static_assert(!kMhc && !kQuant && !kNvfp4 && kTopK == 0);
    run_impl(
        ref,
        out,
        input,
        input,
        input,
        std::nullopt,
        norm_weight,
        eps,
        false,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        add_residual,
        residual_out,
        weight_offset,
        two_shot,
        vecs_per_thread);
  }

  /// run_residual_norm() that also writes the MXFP8 of the normalized output, by
  /// FlashInfer's mxfp8_quantize rule (mxfp8_quant_vec): `quantized` [T, kHiddenDim]
  /// e4m3 and `scales` [(kHiddenDim / 32) * 128] UE8M0 in the 128x4 swizzle of one
  /// 128-row tile (T <= 128, padding rows zeroed). `out` and `residual_out` are
  /// run_residual_norm()'s, bit for bit. `trigger_at_entry`: see
  /// MoeFinalizeAllReduceParams::trigger_at_entry.
  static void run_residual_norm_quant(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      TensorView norm_weight,
      double eps,
      TensorView add_residual,
      TensorView residual_out,
      double weight_offset,
      bool two_shot,
      int64_t vecs_per_thread,
      TensorView quantized,
      TensorView scales,
      bool trigger_at_entry) {
    static_assert(!kMhc && kQuant && !kNvfp4 && kTopK != 0);
    run_impl(
        ref,
        out,
        gemm2_out,
        permuted_idx,
        expert_weights,
        shared_output,
        norm_weight,
        eps,
        false,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        quantized,
        scales,
        add_residual,
        residual_out,
        weight_offset,
        two_shot,
        vecs_per_thread,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        trigger_at_entry);
  }

  /// kTopK == 0: out = RMSNorm(allreduce(input) + add_residual), residual_out the
  /// sum, plus the MXFP8 of `out` as in run_residual_norm_quant().
  static void run_plain_quant(
      CommunicatorRef ref,
      TensorView out,
      TensorView input,
      TensorView norm_weight,
      double eps,
      TensorView add_residual,
      TensorView residual_out,
      double weight_offset,
      bool two_shot,
      int64_t vecs_per_thread,
      TensorView quantized,
      TensorView scales,
      bool trigger_at_entry) {
    static_assert(!kMhc && kQuant && !kNvfp4 && kTopK == 0);
    run_impl(
        ref,
        out,
        input,
        input,
        input,
        std::nullopt,
        norm_weight,
        eps,
        false,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        quantized,
        scales,
        add_residual,
        residual_out,
        weight_offset,
        two_shot,
        vecs_per_thread,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        trigger_at_entry);
  }

  /// run_plain_quant() that also writes the per-token NVFP4 of `out`, as FlashInfer's
  /// NVFP4QuantizePerTokenKernel does with `global_scale_inv` [1] fp32 in the linear
  /// layout (nvfp4_quant_row; kNvfp4Ftz: the subnormal mode the caller found FlashInfer's
  /// kernel to have): `fp4` [T, kHiddenDim / 2] packed e2m1 codes, `fp4_scales`
  /// [T * kHiddenDim / 16] e4m3 block scales, `token_scales` [T] fp32; every byte of
  /// the T rows written. `out`, `residual_out`, `quantized` and `scales` are
  /// run_plain_quant()'s, bit for bit.
  static void run_plain_quant_nvfp4(
      CommunicatorRef ref,
      TensorView out,
      TensorView input,
      TensorView norm_weight,
      double eps,
      TensorView add_residual,
      TensorView residual_out,
      double weight_offset,
      bool two_shot,
      int64_t vecs_per_thread,
      TensorView quantized,
      TensorView scales,
      TensorView fp4,
      TensorView fp4_scales,
      TensorView token_scales,
      TensorView global_scale_inv) {
    static_assert(!kMhc && kQuant && kNvfp4 && kTopK == 0);
    run_impl(
        ref,
        out,
        input,
        input,
        input,
        std::nullopt,
        norm_weight,
        eps,
        false,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        quantized,
        scales,
        add_residual,
        residual_out,
        weight_offset,
        two_shot,
        vecs_per_thread,
        std::nullopt,
        fp4,
        fp4_scales,
        token_scales,
        global_scale_inv);
  }

  /// Finalize + all-reduce + HC=4 post; original reduced output is retained.
  static void run_mhc_post(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb) {
    static_assert(kMhc);
    run_impl(
        ref,
        out,
        gemm2_out,
        permuted_idx,
        expert_weights,
        shared_output,
        std::nullopt,
        0.0,
        false,
        mhc_out,
        residual,
        post,
        comb);
  }

  static void run_mhc_post_combine_norm(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb,
      TensorView pre,
      TensorView norm_weight,
      double eps,
      TensorView normalized) {
    static_assert(kMhc);
    run_impl(
        ref,
        out,
        gemm2_out,
        permuted_idx,
        expert_weights,
        shared_output,
        norm_weight,
        eps,
        false,
        mhc_out,
        residual,
        post,
        comb,
        pre,
        normalized);
  }

  static void run_mhc_post_combine_norm_quant(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      TensorView mhc_out,
      TensorView residual,
      TensorView post,
      TensorView comb,
      TensorView pre,
      TensorView norm_weight,
      double eps,
      TensorView normalized,
      TensorView quantized,
      TensorView scales) {
    static_assert(kMhc && kQuant);
    run_impl(
        ref,
        out,
        gemm2_out,
        permuted_idx,
        expert_weights,
        shared_output,
        norm_weight,
        eps,
        false,
        mhc_out,
        residual,
        post,
        comb,
        pre,
        normalized,
        quantized,
        scales);
  }

 private:
  static void run_impl(
      CommunicatorRef ref,
      TensorView out,
      TensorView gemm2_out,
      TensorView permuted_idx,
      TensorView expert_weights,
      std::optional<TensorView> shared_output,
      std::optional<TensorView> norm_weight,
      double eps,
      bool prefetch_metadata,
      std::optional<TensorView> mhc_out,
      std::optional<TensorView> residual,
      std::optional<TensorView> post,
      std::optional<TensorView> comb,
      std::optional<TensorView> pre = std::nullopt,
      std::optional<TensorView> normalized = std::nullopt,
      std::optional<TensorView> quantized = std::nullopt,
      std::optional<TensorView> scales = std::nullopt,
      std::optional<TensorView> add_residual = std::nullopt,
      std::optional<TensorView> residual_out = std::nullopt,
      double weight_offset = 0.0,
      bool two_shot = false,
      int64_t vecs_per_thread = 1,
      std::optional<TensorView> shared_gate = std::nullopt,
      std::optional<TensorView> fp4 = std::nullopt,
      std::optional<TensorView> fp4_scales = std::nullopt,
      std::optional<TensorView> token_scales = std::nullopt,
      std::optional<TensorView> global_scale_inv = std::nullopt,
      bool trigger_at_entry = false) {
    using namespace host;
    const auto& comm = *ref.get();
    const auto& push = comm.get_push_obj();
    CHECK_HOST(push.world_size == kWorldSize)
        << "communicator holds " << push.world_size << " ranks, kernel built for " << kWorldSize;

    auto T = SymbolicSize{"num_tokens"};
    auto P = SymbolicSize{"num_permuted_rows"};
    auto TK = SymbolicSize{"num_expanded"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({T, kHiddenDim})
        .with_strides({kHiddenDim, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(out);
    TensorMatcher({P, kHiddenDim})
        .with_strides({kHiddenDim, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(gemm2_out);
    if constexpr (kTopK == 0) {
      CHECK_HOST(P.unwrap() == T.unwrap()) << "plain input must be [num_tokens, hidden]";
    } else {
      TensorMatcher({T, kTopK})
          .with_strides({kTopK, 1})
          .with_dtype<WeightT>()
          .template with_device<kDLCUDA>(device)
          .verify(expert_weights);
      TK.set_value(T.unwrap() * kTopK);
      TensorMatcher({TK}).with_strides({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(permuted_idx);
    }
    CHECK_HOST(add_residual.has_value() == residual_out.has_value());
    if (add_residual.has_value()) {
      CHECK_HOST(norm_weight.has_value()) << "the residual add is part of the norm epilogue";
      TensorMatcher({T, kHiddenDim})
          .with_strides({kHiddenDim, 1})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(add_residual.value())
          .verify(residual_out.value());
    }
    if (shared_output.has_value()) {
      TensorMatcher({T, kHiddenDim})
          .with_strides({kHiddenDim, 1})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(shared_output.value());
    }
    if constexpr (kGatedShared) {
      CHECK_HOST(shared_output.has_value() && shared_gate.has_value());
      TensorMatcher({T}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(shared_gate.value());
    }
    if (norm_weight.has_value()) {
      TensorMatcher({kHiddenDim})
          .with_strides({1})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(norm_weight.value());
    }
    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    if constexpr (kQuant && !kMhc) {
      CHECK_HOST(num_tokens <= 128) << "the quant epilogue writes one 128-row scale tile";
      CHECK_HOST(norm_weight.has_value()) << "the quant epilogue quantizes the normalized row";
      CHECK_HOST(quantized.has_value() && scales.has_value()) << "the quant epilogue needs its outputs";
      TensorMatcher({T, kHiddenDim})
          .with_strides({kHiddenDim, 1})
          .with_dtype<fp8_e4m3_t>()
          .with_device<kDLCUDA>(device)
          .verify(quantized.value());
      TensorMatcher({(kHiddenDim / 32) * 128})
          .with_strides({1})
          .with_dtype<uint8_t>()
          .with_device<kDLCUDA>(device)
          .verify(scales.value());
    }
    // DeepSeek-V4's mHC entry keeps the checks it shipped with
    if constexpr (kQuant && kMhc) {
      CHECK_HOST(num_tokens <= 128) << "the quant epilogue writes one 128-row scale tile";
      CHECK_HOST(norm_weight.has_value());
      TensorMatcher({T, kHiddenDim}).with_dtype<fp8_e4m3_t>().with_device<kDLCUDA>(device).verify(quantized.value());
      TensorMatcher({(kHiddenDim / 32) * 128})
          .with_dtype<uint8_t>()
          .with_device<kDLCUDA>(device)
          .verify(scales.value());
    }
    if constexpr (kNvfp4) {
      CHECK_HOST(norm_weight.has_value()) << "the NVFP4 epilogue quantizes the normalized row";
      CHECK_HOST(fp4.has_value() && fp4_scales.has_value() && token_scales.has_value() && global_scale_inv.has_value())
          << "the NVFP4 epilogue needs its outputs and global scale";
      TensorMatcher({T, kHiddenDim / 2})
          .with_strides({kHiddenDim / 2, 1})
          .with_dtype<uint8_t>()
          .with_device<kDLCUDA>(device)
          .verify(fp4.value());
      auto SF = SymbolicSize{"num_nvfp4_scales"};
      SF.set_value(T.unwrap() * (kHiddenDim / 16));
      TensorMatcher({SF}).with_strides({1}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(
          fp4_scales.value());
      TensorMatcher({T}).with_strides({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(
          token_scales.value());
      TensorMatcher({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(global_scale_inv.value());
      CHECK_HOST(reinterpret_cast<uintptr_t>(fp4.value().data_ptr()) % 4 == 0) << "fp4 must be 4B aligned";
    }
    if constexpr (kMhc) {
      static_assert(kHiddenDim == 5120);
      if (norm_weight.has_value() || kCollapse) {
        TensorMatcher({T, 4}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(pre.value());
        TensorMatcher({T, kHiddenDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(normalized.value());
      }
      TensorMatcher({T, 4, kHiddenDim})
          .with_dtype<bf16_t>()
          .with_device<kDLCUDA>(device)
          .verify(mhc_out.value())
          .verify(residual.value());
      TensorMatcher({T, 4}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(post.value());
      TensorMatcher({T, 4, 4}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(comb.value());
    }
    CHECK_HOST(num_tokens > 0) << "num_tokens must be positive";
    CHECK_HOST(reinterpret_cast<uintptr_t>(gemm2_out.data_ptr()) % 16 == 0) << "gemm2_out must be 16B aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0) << "out must be 16B aligned";

    // the whole [T, hidden] row view is staged by vector index, so it must fit
    // one push slot; the generic push kernel's callers pick the slot size
    const int64_t nbytes = num_tokens * int64_t(kHiddenDim) * sizeof(bf16_t);
    CHECK_HOST(nbytes <= push.slot_bytes) << "num_tokens * hidden * 2 = " << nbytes << " bytes exceeds the "
                                          << push.slot_bytes << "-byte push slot (reduce the batch or enlarge "
                                          << "max_push_size)";
    // one cluster (and phase counter) per row; the bumper cluster is launched
    // only when counters are left over for it to flip
    CHECK_HOST(num_tokens <= push.num_blocks)
        << "num_tokens = " << num_tokens << " exceeds the " << push.num_blocks << " push phase counters of the plane";
    const uint32_t num_clusters = num_tokens + (num_tokens < push.num_blocks ? 1 : 0);
    if constexpr (kQuant && !kMhc) {
      CHECK_HOST(num_tokens == 128 || num_tokens < push.num_blocks)
          << "the bumper cluster zeros the scale tile's padding rows; the plane has no counter left for it";
    }

    const auto params = Params{
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .gemm2 = static_cast<const bf16_t*>(gemm2_out.data_ptr()),
        .idx = static_cast<const int32_t*>(permuted_idx.data_ptr()),
        .weights = static_cast<const WeightT*>(expert_weights.data_ptr()),
        .shared = shared_output.has_value() ? static_cast<const bf16_t*>(shared_output.value().data_ptr()) : nullptr,
        .shared_gate = shared_gate.has_value() ? static_cast<const float*>(shared_gate.value().data_ptr()) : nullptr,
        .norm_weight = norm_weight.has_value() ? static_cast<const bf16_t*>(norm_weight.value().data_ptr()) : nullptr,
        .norm_eps = static_cast<float>(eps),
        .prefetch_metadata = prefetch_metadata,
        .rank = push.rank,
        .num_tokens = num_tokens,
        .num_push_counters = push.num_blocks,
        .ws = push.get_workspace<kWorldSize>(nbytes),
        .mhc_out = mhc_out.has_value() ? static_cast<bf16_t*>(mhc_out.value().data_ptr()) : nullptr,
        .residual = residual.has_value() ? static_cast<const bf16_t*>(residual.value().data_ptr()) : nullptr,
        .post = post.has_value() ? static_cast<const float*>(post.value().data_ptr()) : nullptr,
        .comb = comb.has_value() ? static_cast<const float*>(comb.value().data_ptr()) : nullptr,
        .pre = pre.has_value() ? static_cast<const float*>(pre.value().data_ptr()) : nullptr,
        .normalized = normalized.has_value() ? static_cast<bf16_t*>(normalized.value().data_ptr()) : nullptr,
        .quantized = quantized.has_value() ? static_cast<fp8_e4m3_t*>(quantized.value().data_ptr()) : nullptr,
        .scales = scales.has_value() ? static_cast<uint8_t*>(scales.value().data_ptr()) : nullptr,
        .add_residual =
            add_residual.has_value() ? static_cast<const bf16_t*>(add_residual.value().data_ptr()) : nullptr,
        .residual_out = add_residual.has_value() ? static_cast<bf16_t*>(residual_out.value().data_ptr()) : nullptr,
        .norm_weight_offset = static_cast<float>(weight_offset),
        .fp4 = fp4.has_value() ? static_cast<uint8_t*>(fp4.value().data_ptr()) : nullptr,
        .fp4_scales = fp4_scales.has_value() ? static_cast<uint8_t*>(fp4_scales.value().data_ptr()) : nullptr,
        .token_scales = token_scales.has_value() ? static_cast<float*>(token_scales.value().data_ptr()) : nullptr,
        .global_scale_inv =
            global_scale_inv.has_value() ? static_cast<const float*>(global_scale_inv.value().data_ptr()) : nullptr,
        .trigger_at_entry = trigger_at_entry,
    };

    const auto has_shared = shared_output.has_value();
    const auto has_norm = norm_weight.has_value();
    auto kern = has_shared ? (has_norm ? kernel<true, true> : kernel<true, false>)
                           : (has_norm ? kernel<false, true> : kernel<false, false>);
    if constexpr (!kMhc) {
      if (two_shot)
        kern = has_shared ? (has_norm ? kernel<true, true, true> : kernel<true, false, true>)
                          : (has_norm ? kernel<false, true, true> : kernel<false, false, true>);
    }
    uint32_t block_size = Trait::kBlockSize;
    CHECK_HOST(vecs_per_thread == 1 || vecs_per_thread == 2) << "vecs_per_thread must be 1 or 2";
    if (vecs_per_thread == 2) {
      CHECK_HOST(has_norm) << "two vectors per thread is a norm-epilogue path";
      if constexpr (!kMhc && kTwoVecsFit) {
        kern = two_shot ? (has_shared ? kernel<true, true, true, 2> : kernel<false, true, true, 2>)
                        : (has_shared ? kernel<true, true, false, 2> : kernel<false, true, false, 2>);
        block_size = RowClusterTrait<kHiddenDim, kClusterSize, 2>::kBlockSize;
      } else {
        CHECK_HOST(false) << "cluster size " << kClusterSize << " leaves partial warps at two vectors per thread";
      }
    }
    // __cluster_dims__(1, kClusterSize, 1) is compiled in, so a plain launch
    // already forms the clusters along y
    LaunchKernel(dim3(num_clusters, kClusterSize), block_size, out.device()).enable_pdl(kUsePDL)(kern, params);
  }
};

// --- the NVFP4 epilogue on rows from memory ------------------------------------
// nvfp4_quant_row on bf16 rows read from global memory, each row by one cluster laid
// out as the norm epilogue lays out a row (kClusterSize blocks, kVecs 16B vectors a
// thread), without a push plane: a caller checks the epilogue's arithmetic against
// FlashInfer's kernel with it (and picks kNvfp4Ftz), and the tests use it.

struct Nvfp4RowQuantParams {
  const bf16_t* input;  // [num_tokens, kHiddenDim]
  uint8_t* fp4;
  uint8_t* scales;
  float* token_scales;
  const float* global_scale_inv;
};

template <uint32_t kHiddenDim, uint32_t kClusterSize, uint32_t kVecs, bool kFtz>
__global__ __launch_bounds__(RowClusterTrait<kHiddenDim, kClusterSize, kVecs>::kBlockSize) __cluster_dims__(
    1, kClusterSize, 1) void nvfp4_row_quant_kernel(const __grid_constant__ Nvfp4RowQuantParams params) {
  using T = RowClusterTrait<kHiddenDim, kClusterSize, kVecs>;
  const uint32_t token = blockIdx.x;
  const uint32_t cluster_rank = blockIdx.y;
  const uint32_t vec0 = cluster_rank * T::kBlockSize * kVecs + threadIdx.x;
  StageVec row[kVecs];
#pragma unroll
  for (uint32_t v = 0; v < kVecs; ++v) {
    row[v].load(params.input + static_cast<int64_t>(token) * kHiddenDim, vec0 + v * T::kBlockSize);
  }
  __shared__ float smem_amax[kClusterSize][T::kNumWarps];
  nvfp4_quant_row<kHiddenDim, kClusterSize, T::kBlockSize, kVecs, kFtz>(
      row,
      token,
      vec0,
      cluster_rank,
      smem_amax,
      params.fp4,
      params.scales,
      params.token_scales,
      params.global_scale_inv);
}

template <uint32_t kHiddenDim, uint32_t kClusterSize, uint32_t kVecs>
struct Nvfp4RowQuantKernel {
  using TensorView = tvm::ffi::TensorView;

  /// The per-token NVFP4 of `input` [T, kHiddenDim] bf16 into `fp4` [T, kHiddenDim / 2],
  /// `scales` [T * kHiddenDim / 16] and `token_scales` [T], as the kNvfp4 epilogue writes
  /// them (subnormals kept).
  static void
  run(TensorView input, TensorView fp4, TensorView scales, TensorView token_scales, TensorView global_scale_inv) {
    launch<false>(input, fp4, scales, token_scales, global_scale_inv);
  }

  /// run() with kNvfp4Ftz: the compare and the multiplies flush subnormals.
  static void
  run_ftz(TensorView input, TensorView fp4, TensorView scales, TensorView token_scales, TensorView global_scale_inv) {
    launch<true>(input, fp4, scales, token_scales, global_scale_inv);
  }

 private:
  template <bool kFtz>
  static void
  launch(TensorView input, TensorView fp4, TensorView scales, TensorView token_scales, TensorView global_scale_inv) {
    using namespace host;
    using Trait = RowClusterTrait<kHiddenDim, kClusterSize, kVecs>;
    auto T = SymbolicSize{"num_tokens"};
    auto SF = SymbolicSize{"num_scales"};
    SymbolicDevice device;
    device.set_options<kDLCUDA>();
    TensorMatcher({T, kHiddenDim})
        .with_strides({kHiddenDim, 1})
        .with_dtype<bf16_t>()
        .with_device<kDLCUDA>(device)
        .verify(input);
    TensorMatcher({T, kHiddenDim / 2})
        .with_strides({kHiddenDim / 2, 1})
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device)
        .verify(fp4);
    SF.set_value(T.unwrap() * (kHiddenDim / 16));
    TensorMatcher({SF}).with_strides({1}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(scales);
    TensorMatcher({T}).with_strides({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(token_scales);
    TensorMatcher({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device).verify(global_scale_inv);
    const auto num_tokens = static_cast<uint32_t>(T.unwrap());
    CHECK_HOST(num_tokens > 0) << "num_tokens must be positive";
    CHECK_HOST(reinterpret_cast<uintptr_t>(input.data_ptr()) % 16 == 0) << "input must be 16B aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(fp4.data_ptr()) % 4 == 0) << "fp4 must be 4B aligned";
    const auto params = Nvfp4RowQuantParams{
        .input = static_cast<const bf16_t*>(input.data_ptr()),
        .fp4 = static_cast<uint8_t*>(fp4.data_ptr()),
        .scales = static_cast<uint8_t*>(scales.data_ptr()),
        .token_scales = static_cast<float*>(token_scales.data_ptr()),
        .global_scale_inv = static_cast<const float*>(global_scale_inv.data_ptr()),
    };
    LaunchKernel(dim3(num_tokens, kClusterSize), Trait::kBlockSize, input.device())(
        nvfp4_row_quant_kernel<kHiddenDim, kClusterSize, kVecs, kFtz>, params);
  }
};

}  // namespace sglang
