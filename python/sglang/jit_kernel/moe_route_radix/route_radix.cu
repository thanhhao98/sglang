// Torch-extension host binding for the K3 radix-select router.
// Exposes the shared destination-passing ABI `run(...)` (matches
// baseline/route_triton_baseline.run and prior_art/route_tiny_v3.cuh RouteTinyKernel::run).
// Launches on at::cuda::getCurrentCUDAStream(). Authored but NOT yet built on GPU.
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include "route_radix.cuh"

void run(torch::Tensor scores, torch::Tensor bias, torch::Tensor out_w, torch::Tensor out_i,
         int64_t topk, double routed_scaling_factor, bool renormalize, bool apply_scale) {
    TORCH_CHECK(scores.dim() == 2 && bias.dim() == 1, "scores must be 2D, bias 1D");
    TORCH_CHECK(scores.scalar_type() == torch::kBFloat16, "scores must be bf16");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be fp32");
    TORCH_CHECK(out_w.scalar_type() == torch::kFloat32, "out_w must be fp32");
    TORCH_CHECK(out_i.scalar_type() == torch::kInt32, "out_i must be int32");
    TORCH_CHECK(scores.stride(1) == 1, "scores inner stride must be 1");
    const int M = scores.size(0);
    const int N = scores.size(1);
    const int K = static_cast<int>(topk);
    // This kernel is specialized for the K3 decode regime; the runtime dispatcher routes
    // every other shape/dtype/scalar combination to the recovered Triton baseline.
    TORCH_CHECK(N == 896 && K == 16, "route_radix specialized for N=896, K=16");
    TORCH_CHECK(bias.size(0) == N, "bias size must equal N");
    TORCH_CHECK(out_w.size(0) == M && out_w.size(1) == K, "out_w shape [M,K]");
    TORCH_CHECK(out_i.size(0) == M && out_i.size(1) == K, "out_i shape [M,K]");

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(M);
    dim3 block(route_radix::kThreads);
    route_radix::route_radix_kernel<896, 16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(scores.data_ptr()),
        bias.data_ptr<float>(), out_w.data_ptr<float>(), out_i.data_ptr<int32_t>(),
        M, scores.stride(0), out_w.stride(0), out_i.stride(0),
        static_cast<float>(routed_scaling_factor), renormalize ? 1 : 0, apply_scale ? 1 : 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "K3 sigmoid-bias radix-select top-16 router (destination-passing)");
}
