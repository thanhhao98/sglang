// Tiny bf16 GEMV for skinny decode projections: out[T, N] = x[T, K] @ W[N, K]^T.
//
// cublas dispatches these shapes (K3: b+f_a merged [7168 -> 140], f_b
// [128 -> 1536]) to gemvx/dot kernel pairs at 4-5us each; a plain
// CTA-per-output kernel is bandwidth/latency-optimal at ~1-2us.
//
// Grid: (N, T). One CTA per output element: kThreads lanes stride the K
// dimension with 128-bit loads, block-reduce, lane 0 writes the scalar.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, type aliases
#include <sgl_kernel/vec.cuh>    // For AlignedVector
#include <sgl_kernel/warp.cuh>   // For warp::reduce_sum

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct TinyGemvParams {
  const bf16_t* __restrict__ x;  // [T, K]
  const bf16_t* __restrict__ w;  // [N, K] row-major
  bf16_t* __restrict__ out;      // [T, N]
  int64_t stride_xt;             // x stride along T (elements)
  int64_t stride_ot;             // out stride along T (elements)
  uint32_t K;
  uint32_t N;
};

template <int kThreads, bool kUsePDL>
__global__ void tiny_gemv_kernel(const TinyGemvParams __grid_constant__ params) {
  using namespace device;

  constexpr int kVecN = 8;  // 8 bf16 = 128 bits
  using vec_bf16_t = AlignedVector<bf16_t, kVecN>;

  const uint32_t n = blockIdx.x;   // output index
  const uint32_t t = blockIdx.y;   // token index
  const uint32_t tid = threadIdx.x;
  const uint32_t n_vecs = params.K / kVecN;

  const bf16_t* x_row = params.x + t * params.stride_xt;
  const bf16_t* w_row = params.w + static_cast<int64_t>(n) * params.K;

  PDLWaitPrimary<kUsePDL>();

  float acc = 0.0f;
  for (uint32_t vi = tid; vi < n_vecs; vi += kThreads) {
    vec_bf16_t xv, wv;
    xv.load(x_row, vi);
    wv.load(w_row, vi);
#pragma unroll
    for (int i = 0; i < kVecN; ++i) {
      acc += cast<fp32_t>(xv[i]) * cast<fp32_t>(wv[i]);
    }
  }

  acc = warp::reduce_sum(acc);
  if constexpr (kThreads > 32) {
    __shared__ float smem[kThreads / 32];
    const uint32_t warp_id = tid / 32;
    const uint32_t lane_id = tid % 32;
    if (lane_id == 0) smem[warp_id] = acc;
    __syncthreads();
    if (warp_id == 0) {
      acc = (lane_id < kThreads / 32) ? smem[lane_id] : 0.0f;
      acc = warp::reduce_sum(acc);
    }
  }
  if (tid == 0) {
    params.out[t * params.stride_ot + n] = cast<bf16_t>(acc);
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <int kThreads, bool kUsePDL>
struct TinyGemvKernel {
  static constexpr auto kernel = tiny_gemv_kernel<kThreads, kUsePDL>;

  static void run(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView w,
      const tvm::ffi::TensorView out) {
    using namespace host;

    auto T_ = SymbolicSize{"num_tokens"};
    auto K_ = SymbolicSize{"in_features"};
    auto N_ = SymbolicSize{"out_features"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    // x may be a row slice of a wider fused buffer: allow stride_xt != K.
    TensorMatcher({T_, K_})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, 1})
        .verify(x);
    TensorMatcher({N_, K_})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(w);
    TensorMatcher({T_, N_})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, 1})
        .verify(out);

    const auto num_tokens = static_cast<uint32_t>(T_.unwrap());
    const auto K = static_cast<uint32_t>(K_.unwrap());
    const auto N = static_cast<uint32_t>(N_.unwrap());

    RuntimeCheck(K % 8 == 0, "K must be divisible by 8 for vectorized loads");
    if (num_tokens == 0 || N == 0) return;

    const auto params = TinyGemvParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .w = static_cast<const bf16_t*>(w.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .stride_xt = static_cast<int64_t>(x.stride(0)),
        .stride_ot = static_cast<int64_t>(out.stride(0)),
        .K = K,
        .N = N,
    };

    dim3 grid(N, num_tokens);
    LaunchKernel(grid, kThreads, device.unwrap())
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
