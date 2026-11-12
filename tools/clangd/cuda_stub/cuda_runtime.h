#pragma once

// Minimal CUDA runtime shim used only by clangd.
//
// The real CUDA headers are intentionally not parsed by clangd: clangd 18
// supports CUDA <= 12.3, while this machine has CUDA 13.2, and mixing them
// produces thousands of bogus diagnostics. clangd is launched with
// -nocudainc and -I<this directory>, so this file shadows cuda_runtime.h.
//
// nvcc never sees this directory. Only declarations needed for indexing are
// provided; add more when new kernels use additional runtime APIs.

#include <cstddef>
#include <cstdlib>

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __forceinline__ inline __attribute__((always_inline))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#define __restrict__ __restrict

struct dim3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;

  constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};

struct uint3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

extern __device__ uint3 threadIdx;
extern __device__ uint3 blockIdx;
extern __device__ uint3 blockDim;
extern __device__ uint3 gridDim;

extern "C" int __cudaPushCallConfiguration(dim3 grid, dim3 block, std::size_t shared_mem = 0, void* stream = nullptr);
extern "C" __device__ void __syncthreads();

using cudaError_t = enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
};

using cudaStream_t = void*;
using cudaEvent_t = void*;

using cudaMemcpyKind = enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4,
};

using cudaDeviceAttr = enum cudaDeviceAttr {
  cudaDevAttrComputeCapabilityMajor = 75,
  cudaDevAttrComputeCapabilityMinor = 76,
  cudaDevAttrMaxThreadsPerBlock = 1,
  cudaDevAttrMaxSharedMemoryPerBlock = 8,
  cudaDevAttrMultiProcessorCount = 16,
  cudaDevAttrWarpSize = 37,
};

struct cudaDeviceProp {
  char name[256];
  std::size_t totalGlobalMem;
  std::size_t sharedMemPerBlock;
  int major;
  int minor;
  int multiProcessorCount;
  int maxThreadsPerBlock;
  int maxThreadsPerMultiProcessor;
  int warpSize;
};

extern "C" {
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* properties, int device);
cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attribute, int device);

cudaError_t cudaMalloc(void** pointer, std::size_t size);
cudaError_t cudaFree(void* pointer);
cudaError_t cudaMemcpy(void* destination, const void* source, std::size_t size, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* destination, const void* source, std::size_t size, cudaMemcpyKind kind,
                            cudaStream_t stream);
cudaError_t cudaMemset(void* pointer, int value, std::size_t size);
cudaError_t cudaMemsetAsync(void* pointer, int value, std::size_t size, cudaStream_t stream);

cudaError_t cudaDeviceSynchronize();
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);

cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* milliseconds, cudaEvent_t start, cudaEvent_t end);
}
