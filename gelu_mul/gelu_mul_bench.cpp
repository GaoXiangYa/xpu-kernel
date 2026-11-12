#include "gelu_mul/gelu_mul.cuh"

#include "common/bench_options.h"
#include "common/benchmark.cuh"
#include "common/bf16.cuh"
#include "common/cuda_check.cuh"
#include "common/device_buffer.cuh"
#include "common/test_utils.cuh"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

constexpr int kShapes[] = {1, 127, 256, 1024, 4096, 65536, 1000003};

int run_benchmark(const xpu::test::BenchCliOptions& options) {
  cudaDeviceProp properties{};
  XPU_CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device));

  int sm_major = 0;
  int sm_minor = 0;
  XPU_CUDA_CHECK(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, options.device));
  XPU_CUDA_CHECK(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, options.device));

  std::printf("== gelu_mul benchmark ==\n");
  std::printf("out[i] = gelu_tanh(x[i]) * gate[i]  (bf16, fp32 math)\n");
  std::printf("device %d: %s (sm_%d%d, %.1f GiB)\n", options.device, properties.name, sm_major, sm_minor,
              static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
  std::printf("variants:");
  for (const auto& variant : xpu::gelu_mul::kVariants) {
    std::printf(" %s", variant.name);
  }
  std::printf("\n");

  int cases = 0;
  int failures = 0;

  for (const int n : kShapes) {
    std::printf("\nn = %d\n", n);

    const auto h_x = xpu::test::make_random_vector<xpu::bf16>(static_cast<std::size_t>(n),
                                                              0x9E3779B9u ^ static_cast<std::uint32_t>(n), -8.0, 8.0);
    const auto h_gate = xpu::test::make_random_vector<xpu::bf16>(
        static_cast<std::size_t>(n), 0x85EBCA6Bu ^ static_cast<std::uint32_t>(n), -8.0, 8.0);

    const auto d_x = xpu::make_device_buffer(h_x);
    const auto d_gate = xpu::make_device_buffer(h_gate);
    xpu::DeviceBuffer<xpu::bf16> d_out(static_cast<std::size_t>(n));

    for (const auto& variant : xpu::gelu_mul::kVariants) {
      const xpu::test::BenchOptions bench_options{options.warmup, options.iters, options.rounds};
      try {
        const auto stats = xpu::test::benchmark(
            [&] { variant.launch(d_x.data(), d_gate.data(), d_out.data(), n, nullptr); }, nullptr, bench_options);
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
