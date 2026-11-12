#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.cuh"

namespace xpu::test {

struct BenchOptions {
  int warmup = 20;  // Untimed launches before measuring.
  int iters = 100;  // Launches per timed round.
  int rounds = 5;   // Timed rounds; statistics are computed across rounds.
};

struct BenchResult {
  double min_ms = 0.0;
  double median_ms = 0.0;
  double mean_ms = 0.0;
  double max_ms = 0.0;
  int iters = 0;
  int rounds = 0;
};

namespace detail {

class ScopedCudaEvent {
 public:
  ScopedCudaEvent() { XPU_CUDA_CHECK(cudaEventCreate(&event_)); }

  ~ScopedCudaEvent() {
    if (event_ != nullptr) {
      (void)cudaEventDestroy(event_);
    }
  }

  ScopedCudaEvent(const ScopedCudaEvent&) = delete;
  ScopedCudaEvent& operator=(const ScopedCudaEvent&) = delete;

  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

}  // namespace detail

// Measures `launch` on `stream`. Each round records the total time of `iters`
// launches and divides by `iters`, so per-kernel timing is not polluted by
// cudaEventRecord overhead. The median across rounds is the most robust number.
inline BenchResult benchmark(const std::function<void()>& launch, cudaStream_t stream, const BenchOptions& options) {
  if (!launch) {
    throw std::invalid_argument("benchmark: launch function is empty");
  }
  if (options.warmup < 0 || options.iters <= 0 || options.rounds <= 0) {
    throw std::invalid_argument("benchmark: invalid BenchOptions");
  }

  detail::ScopedCudaEvent start;
  detail::ScopedCudaEvent stop;

  for (int i = 0; i < options.warmup; ++i) {
    launch();
  }
  XPU_CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(options.rounds));

  for (int round = 0; round < options.rounds; ++round) {
    XPU_CUDA_CHECK(cudaEventRecord(start.get(), stream));
    for (int i = 0; i < options.iters; ++i) {
      launch();
    }
    XPU_CUDA_CHECK(cudaEventRecord(stop.get(), stream));
    XPU_CUDA_CHECK(cudaEventSynchronize(stop.get()));

    float elapsed_ms = 0.0f;
    XPU_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
    samples.push_back(static_cast<double>(elapsed_ms) / static_cast<double>(options.iters));
  }

  std::sort(samples.begin(), samples.end());

  const std::size_t middle = samples.size() / 2;
  const double median = (samples.size() % 2 == 0) ? 0.5 * (samples[middle - 1] + samples[middle]) : samples[middle];
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);

  BenchResult result;
  result.min_ms = samples.front();
  result.median_ms = median;
  result.mean_ms = sum / static_cast<double>(samples.size());
  result.max_ms = samples.back();
  result.iters = options.iters;
  result.rounds = options.rounds;
  return result;
}

}  // namespace xpu::test
