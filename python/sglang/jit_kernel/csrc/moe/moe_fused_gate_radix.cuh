// Native-CUDA radix-select top-K router for Kimi K3 (bs=1 decode).
// Reproduces the Triton _router_triton_kernel selection semantics (sigmoid + bias
// ranking-only, biased-NaN -> -1e30, smallest-expert-id tie-break, renormalize by the
// selected raw-sigmoid sum), replacing 16 dependent argmax rounds with byte-histogram
// radix narrowing. One CTA per token; all threads cooperate (parallel histogram,
// parallel suffix-sum split-bin search, parallel boundary-rank + compaction) so the
// dependency depth stays shallow rather than serializing on one thread.
//
// Correctness holes addressed (independently design-reviewed):
//   - canonicalize -0.0 -> +0.0 before the float->key transform;
//   - biased-NaN floored to -1e30 BEFORE keying; the raw activated weight is NOT sanitized;
//   - the 16th-place threshold is a FULL 32-bit key; min-id tie-break applies only among
//     experts whose FULL key equals the threshold (via an exclusive prefix count);
//   - padding lanes n>=N never enter the histograms or winner collection;
//   - final order is (key desc == biased desc, id asc).

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t
#include <sgl_kernel/utils.cuh>  // For LaunchKernel

#include <cub/cub.cuh>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace route_radix {

constexpr int kThreads = 512;  // 512 gave the best eager A/B geomean (1024 marginally fewer ncu cycles
                               // but worse eager due to launch/scheduling overhead; 256 fewer warps to
                               // hide the barrier-bound latency). bs=1 is CTA-barrier-bound.
constexpr float kNanFloor = -1e30f;

// tl.sigmoid(x) = 1/(1+exp(-x)). __fdividef/__expf are the closest match; exact PTX matching
// for id bit-exactness under arbitrary bias is a further tuning step (ids already match on
// all validated cases; weights match to <=~2.4e-7).
__device__ __forceinline__ float sigmoid_match(float x) {
  return __fdividef(1.0f, 1.0f + __expf(-x));
}

// Monotone unsigned key: larger biased -> larger key. Caller must have floored biased-NaN.
__device__ __forceinline__ uint32_t biased_to_key(float biased) {
  if (biased == 0.0f) biased = 0.0f;  // canonicalize -0.0 -> +0.0
  uint32_t u = __float_as_uint(biased);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// One block per row (token). blockDim.x == kThreads. Template on N (experts) and K (top-k).
template <int N, int K>
__global__ void route_radix_kernel(
    const __nv_bfloat16* __restrict__ scores,
    const float* __restrict__ bias,
    float* __restrict__ out_w,
    int32_t* __restrict__ out_i,
    int M,
    long long stride_sm,
    long long stride_wm,
    long long stride_im,
    float routed_scaling_factor,
    int renormalize,
    int apply_scale) {
  constexpr int IPT = (N + kThreads - 1) / kThreads;  // items per thread (id = t*IPT + i)
  using BScan = cub::BlockScan<int, kThreads>;

  const int row = blockIdx.x;
  if (row >= M) return;
  const int t = threadIdx.x;

  __shared__ typename BScan::TempStorage scan_temp;
  __shared__ float s_act[N];     // raw sigmoid (weight source) — never NaN-sanitized
  __shared__ uint32_t s_key[N];  // monotone key of biased (ranking)
  __shared__ int s_hist[256];
  __shared__ int s_suf[256];  // suffix sums of the histogram
  __shared__ uint32_t s_T;    // running threshold prefix -> full 32-bit key
  __shared__ int s_needed;    // count to take from key == T
  __shared__ int s_wid[K];
  __shared__ uint32_t s_wkey[K];
  __shared__ float s_wact[K];

  const __nv_bfloat16* srow = scores + (long long)row * stride_sm;
  for (int n = t; n < N; n += kThreads) {
    float a = sigmoid_match(__bfloat162float(srow[n]));
    float b = a + bias[n];
    if (!(b == b)) b = kNanFloor;  // NaN-floor biased ONLY
    s_act[n] = a;
    s_key[n] = biased_to_key(b);
  }
  if (t == 0) {
    s_T = 0u;
    s_needed = K;
  }
  __syncthreads();

// Radix narrowing, MSB -> LSB, over experts matching the prefix in the higher bytes.
#pragma unroll
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = t; i < 256; i += kThreads)
      s_hist[i] = 0;
    __syncthreads();
    const uint32_t hi_mask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
    const uint32_t pref = s_T;
    for (int n = t; n < N; n += kThreads) {
      const uint32_t k = s_key[n];
      if ((k & hi_mask) == (pref & hi_mask)) atomicAdd(&s_hist[(k >> shift) & 0xFFu], 1);
    }
    __syncthreads();
    // Parallel suffix sum: suf[b] = sum_{j>=b} hist[j]  (exclusive prefix -> suffix).
    // Only the first 256 threads carry a histogram bin; the block scan spans kThreads.
    int prefix, total;
    const int hv = (t < 256) ? s_hist[t] : 0;
    BScan(scan_temp).ExclusiveSum(hv, prefix, total);
    if (t < 256) s_suf[t] = total - prefix;
    __syncthreads();
    // Split bin = the unique b with suf[b] >= needed > suf[b+1] (suf is monotone decreasing).
    const int need = s_needed;
    if (t < 256) {
      const int suf_b = s_suf[t];
      const int suf_b1 = (t < 255) ? s_suf[t + 1] : 0;
      if (suf_b >= need && suf_b1 < need) {
        s_T = (pref & hi_mask) | ((uint32_t)t << shift);
        s_needed = need - suf_b1;  // remaining to take at key == T
      }
    }
    __syncthreads();
  }

  const uint32_t T = s_T;
  const int sneed = s_needed;

  // Parallel min-id boundary: rank[id] = # experts with key==T and smaller id.
  int is_eq[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const int n = t * IPT + i;
    is_eq[i] = (n < N && s_key[n] == T) ? 1 : 0;
  }
  int eq_rank[IPT], eq_total;
  BScan(scan_temp).ExclusiveSum(is_eq, eq_rank, eq_total);
  __syncthreads();

  // selected = (key > T) strict winners  OR  (key == T and among the sneed smallest ids).
  int is_sel[IPT];
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const int n = t * IPT + i;
    bool sel = false;
    if (n < N) {
      const uint32_t k = s_key[n];
      sel = (k > T) || (k == T && eq_rank[i] < sneed);
    }
    is_sel[i] = sel ? 1 : 0;
  }
  int pos[IPT], sel_total;
  BScan(scan_temp).ExclusiveSum(is_sel, pos, sel_total);  // pos = compaction slot
  __syncthreads();
#pragma unroll
  for (int i = 0; i < IPT; ++i) {
    const int n = t * IPT + i;
    if (is_sel[i] && pos[i] < K) {
      s_wid[pos[i]] = n;
      s_wkey[pos[i]] = s_key[n];
      s_wact[pos[i]] = s_act[n];
    }
  }
  __syncthreads();

  // Parallel rank-sort of the K winners by (key desc, id asc): each winner counts how many
  // winners outrank it, then scatters to that slot. Winners have unique (key,id) so ranks are
  // unique. Removes the serial O(K^2) chain from one thread.
  __shared__ int s_sid[K];
  __shared__ float s_sact[K];
  if (t < K) {
    const uint32_t ka = s_wkey[t];
    const int ia = s_wid[t];
    int rank = 0;
    for (int b = 0; b < K; ++b)
      if (s_wkey[b] > ka || (s_wkey[b] == ka && s_wid[b] < ia)) ++rank;
    s_sid[rank] = ia;
    s_sact[rank] = s_wact[t];
  }
  __syncthreads();
  // Renorm sum in output (biased-descending) order to match the reference's op order.
  __shared__ float s_norm;
  if (t == 0) {
    float sum = 0.0f;
    for (int a = 0; a < K; ++a)
      sum += s_sact[a];
    s_norm = (sum > 0.0f) ? sum : 1.0f;
  }
  __syncthreads();
  if (t < K) {
    float w = s_sact[t];
    if (renormalize) w = w / s_norm;
    if (apply_scale) w = w * routed_scaling_factor;
    out_w[(long long)row * stride_wm + t] = w;
    out_i[(long long)row * stride_im + t] = s_sid[t];
  }
}

}  // namespace route_radix

namespace {

struct MoeFusedGateRadixKernel {
  static void
  run(const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView out_w,
      const tvm::ffi::TensorView out_i,
      int64_t topk,
      double routed_scaling_factor,
      bool renormalize,
      bool apply_scale) {
    using namespace host;

    constexpr int kNumExperts = 896;
    constexpr int kTopK = 16;

    auto M_ = SymbolicSize{"num_tokens"};
    auto N_ = SymbolicSize{"num_experts"};
    auto K_ = SymbolicSize{"topk"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M_, N_}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(scores);
    TensorMatcher({N_}).with_dtype<fp32_t>().with_device(device).verify(bias);
    TensorMatcher({M_, K_}).with_dtype<fp32_t>().with_device(device).verify(out_w);
    TensorMatcher({M_, K_}).with_dtype<int32_t>().with_device(device).verify(out_i);

    // Specialized for the K3 decode regime; the caller routes every other
    // shape/dtype combination to the triton router.
    RuntimeCheck(
        N_.unwrap() == kNumExperts && K_.unwrap() == kTopK && topk == kTopK,
        "moe_fused_gate_radix is specialized for N=896, K=16");

    const auto M = static_cast<uint32_t>(M_.unwrap());
    if (M == 0) return;

    LaunchKernel(M, route_radix::kThreads, device.unwrap())(
        route_radix::route_radix_kernel<kNumExperts, kTopK>,
        static_cast<const bf16_t*>(scores.data_ptr()),
        static_cast<const fp32_t*>(bias.data_ptr()),
        static_cast<fp32_t*>(out_w.data_ptr()),
        static_cast<int32_t*>(out_i.data_ptr()),
        static_cast<int>(M),
        static_cast<long long>(scores.stride(0)),
        static_cast<long long>(out_w.stride(0)),
        static_cast<long long>(out_i.stride(0)),
        static_cast<float>(routed_scaling_factor),
        renormalize ? 1 : 0,
        apply_scale ? 1 : 0);
  }
};

}  // namespace
