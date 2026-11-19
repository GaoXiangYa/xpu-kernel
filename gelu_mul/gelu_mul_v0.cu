#include "gelu_mul/gelu_mul.cuh"

#include "common/cuda_check.cuh"

namespace xpu::gelu_mul {
namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ float gelu_exact(float x) {
  constexpr float kInvSqrtTwo = 0.70710678118654752440f;
  return 0.5f * x * (1.0f + erff(x * kInvSqrtTwo));
}

__global__ void gelu_mul_kernel_v0(const bf16* __restrict__ input, bf16* __restrict__ out, long hidden_size) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const long row = blockIdx.y;

  if (col < hidden_size) {
    const long input_offset = row * 2 * hidden_size;
    const float x = bf16_to_float(input[input_offset + col]);
    const float gate = bf16_to_float(input[input_offset + hidden_size + col]);
    out[row * hidden_size + col] = bf16_from_float(gelu_exact(x) * gate);
  }
}

}  // namespace

void launch_v0(const bf16* input, bf16* out, long batch_size, long hidden_size, cudaStream_t stream) {
  if (batch_size <= 0 || hidden_size <= 0) {
    return;
  }

  // CUDA grid.y is limited to 65535; the XPUOJ constraints keep batch_size far
  // below that limit.
  const unsigned int blocks_x = static_cast<unsigned int>((hidden_size + kBlockSize - 1) / kBlockSize);
  const unsigned int blocks_y = static_cast<unsigned int>(batch_size);
  const dim3 grid(blocks_x, blocks_y);

  gelu_mul_kernel_v0<<<grid, kBlockSize, 0, stream>>>(input, out, hidden_size);
  XPU_CUDA_LAUNCH_CHECK();
}

}  // namespace xpu::gelu_mul

extern "C" void run_kernel(const __nv_bfloat16* input, __nv_bfloat16* out, long batch_size, long hidden_size) {
  xpu::gelu_mul::launch_v0(input, out, batch_size, hidden_size, nullptr);
}
