#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace xpu {

// Throws std::runtime_error with a readable message. Throwing instead of
// aborting lets the test driver report a failure for one variant and continue
// with the remaining ones.
inline void throw_cuda_error(cudaError_t error, const char* expression, const char* file, int line) {
  if (error == cudaSuccess) {
    return;
  }

  const char* name = cudaGetErrorName(error);
  const char* description = cudaGetErrorString(error);

  std::string message = "CUDA error ";
  message += name != nullptr ? name : "unknown";
  message += " (";
  message += description != nullptr ? description : "no description";
  message += ") at ";
  message += file;
  message += ":";
  message += std::to_string(line);
  message += " while executing `";
  message += expression;
  message += "`";

  throw std::runtime_error(message);
}

}  // namespace xpu

#define XPU_CUDA_CHECK(expression) ::xpu::throw_cuda_error((expression), #expression, __FILE__, __LINE__)

// Call after a kernel launch to catch configuration errors (bad grid/block,
// invalid arguments, ...).
#define XPU_CUDA_LAUNCH_CHECK() XPU_CUDA_CHECK(cudaGetLastError())

#define XPU_CUDA_SYNC_CHECK() XPU_CUDA_CHECK(cudaDeviceSynchronize())
