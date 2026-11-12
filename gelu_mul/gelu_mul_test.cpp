#include "gelu_mul/gelu_mul.cuh"

#include "common/bf16.cuh"
#include "common/cuda_check.cuh"
#include "common/cuda_test.h"
#include "common/device_buffer.cuh"
#include "common/test_utils.cuh"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kShapes[] = {1, 127, 256, 1024, 4096, 65536, 1000003};

// bf16 has ~3 decimal digits, so compare with a bf16-sized tolerance rather
// than the fp32 tolerance used by most CUDA samples.
constexpr double kRtol = 1e-2;
constexpr double kAtol = 1e-4;

class GeluMulTest : public xpu::test::CudaTest {};

// CPU-only sanity check for the reference implementation. Keeping this test
// independent of CUDA means it runs even on machines without a GPU.
TEST(GeluMulReference, KnownValues) {
  const float x_values[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
  const float gate_values[] = {1.0f, 2.0f, 1.0f, 1.0f, 2.0f};
  const float expected[] = {-0.00363159179688f, -0.318359375f, 0.0f, 0.83984375f, 6.0f};

  std::vector<xpu::bf16> x(5);
  std::vector<xpu::bf16> gate(5);
  std::vector<xpu::bf16> actual(5);
  for (int i = 0; i < 5; ++i) {
    x[i] = xpu::bf16_from_float(x_values[i]);
    gate[i] = xpu::bf16_from_float(gate_values[i]);
  }

  xpu::gelu_mul::reference(x.data(), gate.data(), actual.data(), 5);

  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(xpu::bf16_to_float(actual[i]), expected[i], 1e-6f) << "index=" << i;
  }
}

// The operator contract is that n <= 0 is a no-op. This test does not need a
// GPU because the launcher must return before touching the device.
TEST(GeluMul, NonPositiveSizeIsNoOp) {
  for (const auto& variant : xpu::gelu_mul::kVariants) {
    SCOPED_TRACE(::testing::Message() << "variant=" << variant.name);
    EXPECT_NO_THROW(variant.launch(nullptr, nullptr, nullptr, 0, nullptr));
    EXPECT_NO_THROW(variant.launch(nullptr, nullptr, nullptr, -1, nullptr));
  }
}

TEST_F(GeluMulTest, AllVersionsMatchReference) {
  for (const int n : kShapes) {
    SCOPED_TRACE(::testing::Message() << "n=" << n);

    const auto h_x = xpu::test::make_random_vector<xpu::bf16>(static_cast<std::size_t>(n),
                                                              0x9E3779B9u ^ static_cast<std::uint32_t>(n), -8.0, 8.0);
    const auto h_gate = xpu::test::make_random_vector<xpu::bf16>(
        static_cast<std::size_t>(n), 0x85EBCA6Bu ^ static_cast<std::uint32_t>(n), -8.0, 8.0);

    std::vector<xpu::bf16> h_expected(static_cast<std::size_t>(n));
    xpu::gelu_mul::reference(h_x.data(), h_gate.data(), h_expected.data(), n);

    const auto d_x = xpu::make_device_buffer(h_x);
    const auto d_gate = xpu::make_device_buffer(h_gate);
    xpu::DeviceBuffer<xpu::bf16> d_out(static_cast<std::size_t>(n));

    for (const auto& variant : xpu::gelu_mul::kVariants) {
      SCOPED_TRACE(::testing::Message() << "variant=" << variant.name);

      // Poison the output so an implementation that skips elements fails.
      const std::vector<xpu::bf16> poisoned(static_cast<std::size_t>(n),
                                            xpu::bf16_from_float(std::numeric_limits<float>::quiet_NaN()));
      d_out.copy_from_host(poisoned);

      ASSERT_NO_THROW(variant.launch(d_x.data(), d_gate.data(), d_out.data(), n, nullptr)) << "kernel launch failed";
      ASSERT_NO_THROW(XPU_CUDA_SYNC_CHECK()) << "kernel execution failed";

      const auto h_out = d_out.to_host();
      const auto result = xpu::test::compare_arrays(h_out.data(), h_expected.data(), h_out.size(), kRtol, kAtol);

      EXPECT_TRUE(result.passed) << result.num_mismatches << "/" << n << " elements mismatch, first at index "
                                 << result.first_mismatch << ": actual=" << result.actual_at_first_mismatch
                                 << ", expected=" << result.expected_at_first_mismatch
                                 << ", max_abs=" << result.max_abs_error;
    }
  }
}

}  // namespace
