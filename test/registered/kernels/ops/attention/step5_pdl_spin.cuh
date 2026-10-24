#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

namespace sglang {

// A PDL primary for test_step5_deepselect_region_pack.py: it lets its dependent
// launch at once, spins for `spin_cycles` SM clocks, and only then copies src
// into dst. A dependent that read dst before its griddepcontrol.wait would see
// dst's old contents.
__global__ void step5_pdl_spin_copy_kernel(const float *src, float *dst,
                                           int64_t numel, int64_t spin_cycles) {
  device::PDLTriggerSecondary<true>();
  const long long start = clock64();
  while (clock64() - start < spin_cycles) {
  }
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < numel; i += step) {
    dst[i] = src[i];
  }
}

void step5_pdl_spin_copy(tvm::ffi::TensorView src, tvm::ffi::TensorView dst,
                         int64_t spin_cycles) {
  using namespace host;
  auto N = SymbolicSize{"numel"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({N}).with_dtype<fp32_t>().with_device(device).verify(src);
  TensorMatcher({N}).with_dtype<fp32_t>().with_device(device).verify(dst);
  // Few CTAs, so the dependent's CTAs find free SMs while this grid spins.
  LaunchKernel(8, 256, device.unwrap())(
      step5_pdl_spin_copy_kernel, static_cast<const float *>(src.data_ptr()),
      static_cast<float *>(dst.data_ptr()), N.unwrap(), spin_cycles);
}

} // namespace sglang
