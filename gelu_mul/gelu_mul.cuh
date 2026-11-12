#pragma once

#include "common/bf16.cuh"

#include <cmath>

namespace xpu::gelu_mul {

// Fused GELU + multiply:
//
//   out[i] = gelu(x[i]) * gate[i]
//
// using the tanh approximation
//
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Tensors are contiguous bf16 arrays of `n` elements. Compute in fp32 and round
// the result back to bf16. `stream` may be nullptr (the legacy default stream).
using LaunchFn = void (*)(const bf16* x, const bf16* gate, bf16* out, int n, cudaStream_t stream);

// Version 0: one element per thread, scalar loads/stores.
void launch_v0(const bf16* x, const bf16* gate, bf16* out, int n, cudaStream_t stream = nullptr);

struct Variant {
  const char* name;
  LaunchFn launch;
};

// Add every new implementation here. The test driver iterates this table.
inline constexpr Variant kVariants[] = {
    {"v0_naive", &launch_v0},
};

inline constexpr int kNumVariants = static_cast<int>(sizeof(kVariants) / sizeof(kVariants[0]));

inline double gelu_tanh_reference(double x) {
  constexpr double kSqrtTwoOverPi = 0.79788456080286535588;
  constexpr double kCubicCoefficient = 0.044715;
  const double x_cubed = x * x * x;
  return 0.5 * x * (1.0 + std::tanh(kSqrtTwoOverPi * (x + kCubicCoefficient * x_cubed)));
}

// CPU reference evaluated in double precision, then rounded to bf16 using the
// same float -> bf16 path as the kernels.
inline void reference(const bf16* x, const bf16* gate, bf16* out, int n) {
  for (int i = 0; i < n; ++i) {
    const double xf = static_cast<double>(bf16_to_float(x[i]));
    const double gf = static_cast<double>(bf16_to_float(gate[i]));
    const double value = gelu_tanh_reference(xf) * gf;
    out[i] = bf16_from_float(static_cast<float>(value));
  }
}

}  // namespace xpu::gelu_mul
