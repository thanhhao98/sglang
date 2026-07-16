// K3 latent-MoE tail residual add:
//   out = bf16( bf16(a + b) + c )
// replacing the two separate elementwise adds (up_out + shared, then
// + residual). The double rounding replicates the unfused pair exactly
// (bit-identical), one launch and one memory pass instead of two.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck

#include <sgl_kernel/type.cuh>   // For bf16_t, fp32_t, device::cast
#include <sgl_kernel/utils.cuh>  // For LaunchKernel
#include <sgl_kernel/vec.cuh>    // For AlignedVector

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

struct MoeTailAddParams {
  const bf16_t* __restrict__ a;  // [T, H] contiguous
  const bf16_t* __restrict__ b;  // [T, H] row stride may differ (buffer slice)
  const bf16_t* __restrict__ c;  // [T, H] contiguous
  bf16_t* __restrict__ out;      // [T, H] contiguous
  int64_t stride_bt;             // b stride along T (elements)
  uint32_t H;
};

template <int kThreads, bool kUsePDL>
__global__ void moe_tail_add_kernel(const MoeTailAddParams __grid_constant__ params) {
  using namespace device;

  constexpr int kVecN = 8;
  using vec_bf16_t = AlignedVector<bf16_t, kVecN>;

  const uint32_t t = blockIdx.y;
  const uint32_t v = blockIdx.x * kThreads + threadIdx.x;
  const uint32_t n_vecs = params.H / kVecN;
  if (v >= n_vecs) return;

  PDLWaitPrimary<kUsePDL>();

  vec_bf16_t av, bv, cv, ov;
  av.load(params.a + static_cast<int64_t>(t) * params.H, v);
  bv.load(params.b + t * params.stride_bt, v);
  cv.load(params.c + static_cast<int64_t>(t) * params.H, v);
#pragma unroll
  for (int i = 0; i < kVecN; ++i) {
    // Match the unfused pair bit-for-bit: round (a+b) to bf16 first.
    const float ab = cast<fp32_t>(av[i]) + cast<fp32_t>(bv[i]);
    const bf16_t ab_r = cast<bf16_t>(ab);
    ov[i] = cast<bf16_t>(cast<fp32_t>(ab_r) + cast<fp32_t>(cv[i]));
  }
  ov.store(params.out + static_cast<int64_t>(t) * params.H, v);

  PDLTriggerSecondary<kUsePDL>();
}

template <int kThreads, bool kUsePDL>
struct MoeTailAddKernel {
  static constexpr auto kernel = moe_tail_add_kernel<kThreads, kUsePDL>;

  static void
  run(const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView b,
      const tvm::ffi::TensorView c,
      const tvm::ffi::TensorView out) {
    using namespace host;

    auto T_ = SymbolicSize{"num_tokens"};
    auto H_ = SymbolicSize{"hidden"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(a);
    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(b);
    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(c);
    TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device(device).verify(out);

    const auto T = static_cast<uint32_t>(T_.unwrap());
    const auto H = static_cast<uint32_t>(H_.unwrap());
    RuntimeCheck(H % 8 == 0, "H must be divisible by 8");
    if (T == 0) return;

    const auto params = MoeTailAddParams{
        .a = static_cast<const bf16_t*>(a.data_ptr()),
        .b = static_cast<const bf16_t*>(b.data_ptr()),
        .c = static_cast<const bf16_t*>(c.data_ptr()),
        .out = static_cast<bf16_t*>(out.data_ptr()),
        .stride_bt = b.stride(0),
        .H = H,
    };
    const uint32_t n_vecs = H / 8;
    dim3 grid((n_vecs + kThreads - 1) / kThreads, T);
    LaunchKernel(grid, kThreads, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
