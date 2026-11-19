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

struct Shape {
  long batch;
  long hidden;
};

// The XPUOJ shapes are much larger; the unit tests keep a mix of small and
// medium cases so they remain fast, while the benchmark covers the real sizes.
constexpr Shape kShapes[] = {
    {1, 1}, {2, 3}, {127, 129}, {256, 1024}, {1024, 4096}, {4096, 4096},
};

// bf16 has ~3 decimal digits, so compare with a bf16-sized tolerance rather
// than the fp32 tolerance used by most CUDA samples.
constexpr double kRtol = 1e-2;
constexpr double kAtol = 1e-4;

class GeluMulTest : public xpu::test::CudaTest {};

std::size_t input_elements(const Shape& shape) {
  return static_cast<std::size_t>(shape.batch) * static_cast<std::size_t>(shape.hidden) * 2;
}

std::size_t output_elements(const Shape& shape) {
  return static_cast<std::size_t>(shape.batch) * static_cast<std::size_t>(shape.hidden);
}

// CPU-only sanity check for the reference implementation. Keeping this test
// independent of CUDA means it runs even on machines without a GPU.
TEST(GeluMulReference, KnownValues) {
  // [1, 2, 3, 4]   -> x1 = [1, 2], x2 = [3, 4]
  // [0.5, 1.5, 2.5, 3.5] -> x1 = [0.5, 1.5], x2 = [2.5, 3.5]
  const float input_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.5f, 2.5f, 3.5f};
  const float expected[] = {2.53125f, 7.8125f, 0.86328125f, 4.90625f};

  std::vector<xpu::bf16> input(8);
  std::vector<xpu::bf16> out(4);
  for (std::size_t i = 0; i < 8; ++i) {
    input[i] = xpu::bf16_from_float(input_values[i]);
  }

  xpu::gelu_mul::reference(input.data(), out.data(), 2, 2);

  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(xpu::bf16_to_float(out[i]), expected[i], 1e-6f) << "index=" << i;
  }
}

// The operator contract is that empty sizes are a no-op. This test does not
// need a GPU because the launcher must return before touching the device.
TEST(GeluMul, NonPositiveSizeIsNoOp) {
  for (const auto& variant : xpu::gelu_mul::kVariants) {
    SCOPED_TRACE(::testing::Message() << "variant=" << variant.name);
    EXPECT_NO_THROW(variant.launch(nullptr, nullptr, 0, 0, nullptr));
    EXPECT_NO_THROW(variant.launch(nullptr, nullptr, -1, -1, nullptr));
  }

  EXPECT_NO_THROW(run_kernel(nullptr, nullptr, 0, 0));
  EXPECT_NO_THROW(run_kernel(nullptr, nullptr, -1, -1));
}

TEST_F(GeluMulTest, AllVersionsMatchReference) {
  for (const Shape& shape : kShapes) {
    SCOPED_TRACE(::testing::Message() << "batch=" << shape.batch << ", hidden=" << shape.hidden);

    const auto h_input = xpu::test::make_random_vector<xpu::bf16>(
        input_elements(shape), 0x9E3779B9u ^ static_cast<std::uint32_t>(shape.hidden), -8.0, 8.0);

    std::vector<xpu::bf16> h_expected(output_elements(shape));
    xpu::gelu_mul::reference(h_input.data(), h_expected.data(), shape.batch, shape.hidden);

    const auto d_input = xpu::make_device_buffer(h_input);
    xpu::DeviceBuffer<xpu::bf16> d_out(output_elements(shape));

    for (const auto& variant : xpu::gelu_mul::kVariants) {
      SCOPED_TRACE(::testing::Message() << "variant=" << variant.name);

      // Poison the output so an implementation that skips elements fails.
      const std::vector<xpu::bf16> poisoned(output_elements(shape),
                                            xpu::bf16_from_float(std::numeric_limits<float>::quiet_NaN()));
      d_out.copy_from_host(poisoned);

      ASSERT_NO_THROW(variant.launch(d_input.data(), d_out.data(), shape.batch, shape.hidden, nullptr))
          << "kernel launch failed";
      ASSERT_NO_THROW(XPU_CUDA_SYNC_CHECK()) << "kernel execution failed";

      const auto h_out = d_out.to_host();
      const auto result = xpu::test::compare_arrays(h_out.data(), h_expected.data(), h_out.size(), kRtol, kAtol);

      EXPECT_TRUE(result.passed) << result.num_mismatches << "/" << h_out.size()
                                 << " elements mismatch, first at index " << result.first_mismatch
                                 << ": actual=" << result.actual_at_first_mismatch
                                 << ", expected=" << result.expected_at_first_mismatch
                                 << ", max_abs=" << result.max_abs_error;
    }
  }
}

// Verify the exact XPUOJ entry point, not just the internal launch functions.
TEST_F(GeluMulTest, RunKernelEntryPointMatchesReference) {
  constexpr long kBatch = 33;
  constexpr long kHidden = 129;

  const auto h_input = xpu::test::make_random_vector<xpu::bf16>(kBatch * kHidden * 2, 0x12345678u, -8.0, 8.0);
  std::vector<xpu::bf16> h_expected(kBatch * kHidden);
  xpu::gelu_mul::reference(h_input.data(), h_expected.data(), kBatch, kHidden);

  const auto d_input = xpu::make_device_buffer(h_input);
  xpu::DeviceBuffer<xpu::bf16> d_out(kBatch * kHidden);
  const std::vector<xpu::bf16> poisoned(kBatch * kHidden,
                                        xpu::bf16_from_float(std::numeric_limits<float>::quiet_NaN()));
  d_out.copy_from_host(poisoned);

  ASSERT_NO_THROW(run_kernel(d_input.data(), d_out.data(), kBatch, kHidden)) << "run_kernel launch failed";
  ASSERT_NO_THROW(XPU_CUDA_SYNC_CHECK()) << "run_kernel execution failed";

  const auto h_out = d_out.to_host();
  const auto result = xpu::test::compare_arrays(h_out.data(), h_expected.data(), h_out.size(), kRtol, kAtol);
  EXPECT_TRUE(result.passed) << result.num_mismatches << "/" << h_out.size() << " elements mismatch, first at index "
                             << result.first_mismatch << ": actual=" << result.actual_at_first_mismatch
                             << ", expected=" << result.expected_at_first_mismatch
                             << ", max_abs=" << result.max_abs_error;
}

}  // namespace
