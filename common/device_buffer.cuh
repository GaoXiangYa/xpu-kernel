#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/cuda_check.cuh"

namespace xpu {

// RAII owner of a device allocation. Non-copyable, movable.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t count) { allocate(count); }

  ~DeviceBuffer() { release(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      count_ = other.count_;
      other.data_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void allocate(std::size_t count) {
    release();
    if (count == 0) {
      return;
    }
    XPU_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    count_ = count;
  }

  void release() noexcept {
    if (data_ != nullptr) {
      (void)cudaFree(data_);  // Never throw from a destructor.
      data_ = nullptr;
      count_ = 0;
    }
  }

  void copy_from_host(const T* source) {
    if (count_ == 0) {
      return;
    }
    XPU_CUDA_CHECK(cudaMemcpy(data_, source, count_ * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copy_from_host(const std::vector<T>& source) {
    if (source.size() != count_) {
      throw std::invalid_argument("DeviceBuffer::copy_from_host: size mismatch");
    }
    copy_from_host(source.data());
  }

  void copy_to_host(T* destination) const {
    if (count_ == 0) {
      return;
    }
    XPU_CUDA_CHECK(cudaMemcpy(destination, data_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
  }

  std::vector<T> to_host() const {
    std::vector<T> host(count_);
    if (count_ != 0) {
      copy_to_host(host.data());
    }
    return host;
  }

  T* data() noexcept { return data_; }

  const T* data() const noexcept { return data_; }

  std::size_t size() const noexcept { return count_; }

  bool empty() const noexcept { return count_ == 0; }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0;
};

template <typename T>
DeviceBuffer<T> make_device_buffer(const std::vector<T>& host) {
  DeviceBuffer<T> device(host.size());
  if (!host.empty()) {
    device.copy_from_host(host);
  }
  return device;
}

}  // namespace xpu
