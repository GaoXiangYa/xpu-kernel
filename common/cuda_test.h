#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace xpu::test {

// GoogleTest fixture for tests that need a CUDA device. The test is skipped
// when no device is visible. Select a specific GPU with CUDA_VISIBLE_DEVICES.
class CudaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess) {
      GTEST_SKIP() << "cudaGetDeviceCount failed: " << cudaGetErrorString(error);
    }
    if (device_count == 0) {
      GTEST_SKIP() << "no CUDA device available";
    }

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess) << "failed to select CUDA device 0";
  }
};

}  // namespace xpu::test
