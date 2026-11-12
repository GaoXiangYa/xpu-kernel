#include "gelu_mul/gelu_mul.cuh"

#include "common/cuda_check.cuh"

namespace xpu::gelu_mul {
namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
  constexpr float kCubicCoefficient = 0.044715f;
  const float x_cubed = x * x * x;
  return 0.5f * x * (1.0f + tanhf(kSqrtTwoOverPi * (x + kCubicCoefficient * x_cubed)));
}

__global__ void gelu_mul_kernel_v0(const bf16* __restrict__ x, const bf16* __restrict__ gate, bf16* __restrict__ out,
                                   int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const float xf = bf16_to_float(x[index]);
    const float gf = bf16_to_float(gate[index]);
    out[index] = bf16_from_float(gelu_tanh(xf) * gf);
  }
}

}  // namespace

void launch_v0(const bf16* x, const bf16* gate, bf16* out, int n, cudaStream_t stream) {
  if (n <= 0) {
    return;
  }

  const int num_blocks = (n - 1) / kBlockSize + 1;
  gelu_mul_kernel_v0<<<num_blocks, kBlockSize, 0, stream>>>(x, gate, out, n);
  XPU_CUDA_LAUNCH_CHECK();
}

}  // namespace xpu::gelu_mul
