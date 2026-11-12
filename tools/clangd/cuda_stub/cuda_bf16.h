#pragma once

// Minimal bf16 shim used only by clangd (the real CUDA headers are not parsed,
// see cuda_runtime.h). Conversions use round-to-nearest-even; the exact values
// do not matter for indexing, but keeping them correct avoids confusing hovers.

#include <cstdint>

#include "cuda_runtime.h"

struct __nv_bfloat16 {
  unsigned short __x;
};

__host__ __device__ inline __nv_bfloat16 __float2bfloat16(float value) {
  unsigned int bits = 0;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);

  __nv_bfloat16 result;
  result.__x = static_cast<unsigned short>(bits >> 16);
  return result;
}

__host__ __device__ inline float __bfloat162float(__nv_bfloat16 value) {
  const unsigned int bits = static_cast<unsigned int>(value.__x) << 16;
  float result = 0.0f;
  __builtin_memcpy(&result, &bits, sizeof(result));
  return result;
}
