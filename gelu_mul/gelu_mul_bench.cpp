#include "gelu_mul/gelu_mul.cuh"

#include "common/bench_options.h"
#include "common/benchmark.cuh"
#include "common/bf16.cuh"
#include "common/cuda_check.cuh"
#include "common/device_buffer.cuh"

#include <cstddef>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

struct Shape {
  long batch;
  long hidden;
};

// Includes the three official XPUOJ test shapes plus a couple of small cases.
constexpr Shape kShapes[] = {
    {2, 3}, {1024, 4096}, {12288, 12288}, {16384, 12288}, {12288, 16384},
};

std::size_t input_elements(const Shape& shape) {
  return static_cast<std::size_t>(shape.batch) * static_cast<std::size_t>(shape.hidden) * 2;
}

std::size_t output_elements(const Shape& shape) {
  return static_cast<std::size_t>(shape.batch) * static_cast<std::size_t>(shape.hidden);
}

int run_benchmark(const xpu::test::BenchCliOptions& options) {
  cudaDeviceProp properties{};
  XPU_CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device));

  int sm_major = 0;
  int sm_minor = 0;
  XPU_CUDA_CHECK(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, options.device));
  XPU_CUDA_CHECK(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, options.device));

  std::printf("== gelu_mul benchmark ==\n");
  std::printf("out[b, h] = gelu_exact(input[b, h]) * input[b, h + hidden_size]  (bf16, fp32 math)\n");
  std::printf("device %d: %s (sm_%d%d, %.1f GiB)\n", options.device, properties.name, sm_major, sm_minor,
              static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
  std::printf("variants:");
  for (const auto& variant : xpu::gelu_mul::kVariants) {
    std::printf(" %s", variant.name);
  }
  std::printf("\n");

  int cases = 0;
  int failures = 0;

  for (const Shape& shape : kShapes) {
    const std::size_t input_count = input_elements(shape);
    const std::size_t output_count = output_elements(shape);
    const double input_mib = static_cast<double>(input_count * sizeof(xpu::bf16)) / (1024.0 * 1024.0);
    const double output_mib = static_cast<double>(output_count * sizeof(xpu::bf16)) / (1024.0 * 1024.0);

    std::printf("\nbatch = %ld, hidden = %ld  (input %.1f MiB, out %.1f MiB)\n", shape.batch, shape.hidden, input_mib,
                output_mib);

    // Benchmark throughput does not depend on the actual values, so use a
    // constant fill to avoid spending seconds generating random input.
    const std::vector<xpu::bf16> h_input(input_count, xpu::bf16_from_float(1.0f));

    const auto d_input = xpu::make_device_buffer(h_input);
    xpu::DeviceBuffer<xpu::bf16> d_out(output_count);

    for (const auto& variant : xpu::gelu_mul::kVariants) {
      const xpu::test::BenchOptions bench_options{options.warmup, options.iters, options.rounds};
      try {
        const auto stats = xpu::test::benchmark(
            [&] { variant.launch(d_input.data(), d_out.data(), shape.batch, shape.hidden, nullptr); }, nullptr,
            bench_options);
        std::printf("  %-12s bench median=%.4f ms  min=%.4f  max=%.4f  (iters=%d, rounds=%d)\n", variant.name,
                    stats.median_ms, stats.min_ms, stats.max_ms, stats.iters, stats.rounds);
        ++cases;
      } catch (const std::exception& error) {
        std::printf("  %-12s FAIL  %s\n", variant.name, error.what());
        ++failures;
      }
    }
  }

  std::printf("\nBenchmarked %d case(s), %d failure(s)\n", cases, failures);
  return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const auto options = xpu::test::parse_bench_options(argc, argv, "gelu_mul: CUDA kernel benchmark");

  try {
    XPU_CUDA_CHECK(cudaSetDevice(options.device));
    return run_benchmark(options);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "fatal: %s\n", error.what());
    return 1;
  }
}
