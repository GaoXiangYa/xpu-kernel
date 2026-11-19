#pragma once

#include "common/bf16.cuh"

#include <cmath>

namespace xpu::gelu_mul {

// Fused GELU + multiply for the XPUOJ "Gelu and Mul" problem.
//
// input:  (batch_size, 2 * hidden_size), row-major contiguous, bf16
// out:    (batch_size, hidden_size),    row-major contiguous, bf16
//
//   x1 = input[:, :hidden_size]
//   x2 = input[:, hidden_size:]
//   out = gelu_exact(x1) * x2
//
// GELU uses the exact erf form:
//   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
using LaunchFn = void (*)(const bf16* input, bf16* out, long batch_size, long hidden_size, cudaStream_t stream);

// Version 0: one output element per thread, 2D grid (columns x rows).
void launch_v0(const bf16* input, bf16* out, long batch_size, long hidden_size, cudaStream_t stream = nullptr);

void launch_v1(const bf16* input, bf16* out, long batch_size, long hidden_size, cudaStream_t stream = nullptr);

struct Variant {
  const char* name;
  LaunchFn launch;
};

// Add every new implementation here. The test driver iterates this table.
inline constexpr Variant kVariants[] = {
    {"v0_naive", &launch_v0},
    {"v1", &launch_v1},
};

inline constexpr int kNumVariants = static_cast<int>(sizeof(kVariants) / sizeof(kVariants[0]));

inline double gelu_exact_reference(double x) {
  constexpr double kInvSqrtTwo = 0.70710678118654752440084436210485;
  return 0.5 * x * (1.0 + std::erf(x * kInvSqrtTwo));
}

// CPU reference evaluated in double precision, then rounded to bf16 using the
// same float -> bf16 path as the kernels.
inline void reference(const bf16* input, bf16* out, long batch_size, long hidden_size) {
  for (long row = 0; row < batch_size; ++row) {
    const bf16* input_row = input + row * 2 * hidden_size;
    bf16* out_row = out + row * hidden_size;
    for (long col = 0; col < hidden_size; ++col) {
      const double x = static_cast<double>(bf16_to_float(input_row[col]));
      const double gate = static_cast<double>(bf16_to_float(input_row[hidden_size + col]));
      const double value = gelu_exact_reference(x) * gate;
      out_row[col] = bf16_from_float(static_cast<float>(value));
    }
  }
}

}  // namespace xpu::gelu_mul

// XPUOJ entry point. Must remain a C symbol with exactly this signature.
extern "C" void run_kernel(const __nv_bfloat16* input, __nv_bfloat16* out, long batch_size, long hidden_size);
