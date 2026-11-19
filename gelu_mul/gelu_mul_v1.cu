// Self-contained submission skeleton for XPUOJ "Gelu and Mul".
//
// Copy this file to the judge as-is; all OJ-facing pieces live here:
//   * bf16 conversion helpers
//   * one __global__ kernel
//   * launch_v0 (grid/block configuration)
//   * extern "C" run_kernel entry point
//
// TODO:
//   1. Implement the exact erf GELU.
//   2. Map one output element to one thread, split input row [2 * hidden_size]
//      into x1 = input[row, col] and x2 = input[row, hidden_size + col].
//   3. Configure a 2D grid (columns x rows) and launch the kernel.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace xpu {
namespace gelu_mul {

using bf16 = __nv_bfloat16;
using bf162 = __nv_bfloat162;

__host__ __device__ __forceinline__ bf16 bf16_from_float(float value) {
  return __float2bfloat16(value);
}

__host__ __device__ __forceinline__ float bf16_to_float(bf16 value) {
  return __bfloat162float(value);
}

// TODO: return 0.5 * x * (1 + erf(x / sqrt(2))).
__device__ __forceinline__ float gelu_exact(float x) {
  constexpr float kInvSqrt2 = 0.707106781186;
  return 0.5 * x * (1 + erff(x * kInvSqrt2));
}

__device__ __forceinline__ float2 gelu2_exact(float2 v) {
  constexpr float kInvSqrt2 = 0.707106781186;
  return make_float2(0.5 * v.x * (1 + erff(v.x * kInvSqrt2)), 0.5 * v.y * (1 + erff(v.y * kInvSqrt2)));
}

__global__ void gelu_mul_kernel_normal(const bf16* __restrict__ input, bf16* __restrict__ out, long hidden_size) {
  // TODO: compute one output element per thread.
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < hidden_size) {
    const int input_base = (row << 1) * hidden_size + col;
    float gelu = gelu_exact(bf16_to_float(input[input_base]));
    float gate = bf16_to_float(input[input_base + hidden_size]);
    out[row * hidden_size + col] =  bf16_from_float(gelu * gate);
  }
}

__global__ void gelu_mul_kernel_v1(const bf16* __restrict__ input, bf16* __restrict__ out, long hidden_size) {
  // TODO: compute one output element per thread.
  const int row = blockIdx.y;
  const int col = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);

  if (col < hidden_size) {
    const int input_base = (row << 1) * hidden_size + col;
    const float2 v = __bfloat1622float2(*reinterpret_cast<const bf162*>(input + input_base));
    const float2 gelu2 = gelu2_exact(v);
    const float2 gate2 = __bfloat1622float2(*reinterpret_cast<const bf162*>(input + input_base + hidden_size));
    float2 ret = make_float2(gelu2.x * gate2.x, gelu2.y * gate2.y);
    bf162 bf_ret = __float22bfloat162_rn(ret);
    __stwb(reinterpret_cast<bf162*>(out + row * hidden_size + col), bf_ret);
  }
}

void launch_v1(const __nv_bfloat16* input, __nv_bfloat16* out, long batch_size, long hidden_size, cudaStream_t stream) {
  if (batch_size <= 0 || hidden_size <= 0) {
    return;
  }

  // TODO: choose block size, build a 2D grid (columns x rows), then launch:
  if (hidden_size % 2 == 0) {
    constexpr int kBlockSize = 512;
    const int block_x = (hidden_size + 2 * kBlockSize - 1) / (2 * kBlockSize);
    const int block_y = batch_size;
    dim3 grid(block_x, block_y);

    gelu_mul_kernel_v1<<<grid, kBlockSize, 0, stream>>>(input, out, hidden_size);
  } else {
    constexpr int kBlockSize = 512;
    const int block_x = (hidden_size + kBlockSize - 1) / kBlockSize;
    const int block_y = batch_size;
    dim3 grid(block_x, block_y);

    gelu_mul_kernel_normal<<<grid, kBlockSize, 0, stream>>>(input, out, hidden_size);
  }
}

}  // namespace gelu_mul
}  // namespace xpu

// XPUOJ entry point. Must stay a C symbol with exactly this signature.
//
// For local multi-version builds CMake defines XPU_DISABLE_OJ_ENTRY on every
// version file except v0, so only one run_kernel is linked. When this file is
// submitted to XPUOJ standalone, the macro is not defined and the entry point
// is included.
#ifndef XPU_DISABLE_OJ_ENTRY
extern "C" void run_kernel(const __nv_bfloat16* input, __nv_bfloat16* out, long batch_size, long hidden_size) {
  xpu::gelu_mul::launch_v1(input, out, batch_size, hidden_size, nullptr);
}
#endif