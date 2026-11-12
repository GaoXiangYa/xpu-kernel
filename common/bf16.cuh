#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace xpu {

// All kernels in this project operate on bfloat16 tensors. Use these helpers
// to convert explicitly; kernels normally upcast to fp32 for the math and
// round the result back to bf16.
using bf16 = __nv_bfloat16;

__host__ __device__ inline bf16 bf16_from_float(float value) {
  return __float2bfloat16(value);
}

__host__ __device__ inline float bf16_to_float(bf16 value) {
  return __bfloat162float(value);
}

}  // namespace xpu
