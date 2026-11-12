#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "common/bf16.cuh"

namespace xpu::test {

struct CompareResult {
  bool passed = true;
  std::size_t num_mismatches = 0;
  std::size_t first_mismatch = 0;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double actual_at_first_mismatch = 0.0;
  double expected_at_first_mismatch = 0.0;
};

inline double to_double(float value) {
  return static_cast<double>(value);
}

inline double to_double(double value) {
  return value;
}

inline double to_double(xpu::bf16 value) {
  return static_cast<double>(xpu::bf16_to_float(value));
}

// Element-wise comparison with the usual combined tolerance:
//   |actual - expected| <= atol + rtol * |expected|
// NaN matches NaN, and infinities must have the same sign. bf16 values are
// compared after conversion to double, so the tolerance is in real units.
template <typename T>
CompareResult compare_arrays(const T* actual, const T* expected, std::size_t n, double rtol = 1e-5,
                             double atol = 1e-5) {
  CompareResult result;

  for (std::size_t i = 0; i < n; ++i) {
    const double a = to_double(actual[i]);
    const double e = to_double(expected[i]);

    const bool both_nan = std::isnan(a) && std::isnan(e);
    const bool same_inf = std::isinf(a) && std::isinf(e) && (std::signbit(a) == std::signbit(e));

    double abs_error = 0.0;
    double rel_error = 0.0;
    bool matches = both_nan || same_inf;

    if (!matches) {
      abs_error = std::fabs(a - e);
      // Guard against division by zero when expected is 0.
      rel_error = abs_error / std::max(std::fabs(e), std::numeric_limits<double>::min());
      matches = abs_error <= atol + rtol * std::fabs(e);
    }

    if (std::isfinite(a) && std::isfinite(e)) {
      result.max_abs_error = std::max(result.max_abs_error, abs_error);
      result.max_rel_error = std::max(result.max_rel_error, rel_error);
    }

    if (!matches) {
      if (result.num_mismatches == 0) {
        result.first_mismatch = i;
        result.actual_at_first_mismatch = a;
        result.expected_at_first_mismatch = e;
      }
      ++result.num_mismatches;
    }
  }

  result.passed = result.num_mismatches == 0;
  return result;
}

template <typename T>
struct RandomCast {
  static T cast(double value) { return static_cast<T>(value); }
};

template <>
struct RandomCast<xpu::bf16> {
  static xpu::bf16 cast(double value) { return xpu::bf16_from_float(static_cast<float>(value)); }
};

template <typename T>
std::vector<T> make_random_vector(std::size_t n, std::uint32_t seed, double low = -8.0, double high = 8.0) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(low, high);

  std::vector<T> values(n);
  for (T& value : values) {
    value = RandomCast<T>::cast(distribution(generator));
  }
  return values;
}

}  // namespace xpu::test
