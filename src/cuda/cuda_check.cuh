#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    const cudaError_t err_ = (call);                                              \
    if (err_ != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), \
                   __FILE__, __LINE__, cudaGetErrorString(err_));                 \
      std::exit(3);                                                               \
    }                                                                             \
  } while (0)

// Launch errors surface on the next API call; check them right away instead.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

inline const char* device_name() {
  static cudaDeviceProp prop;
  static bool init = false;
  if (!init) {
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    init = true;
  }
  return prop.name;
}

// One CUDA event pair; elapsed() returns milliseconds.
struct EventTimer {
  cudaEvent_t start{}, stop{};
  EventTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~EventTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void begin(cudaStream_t s) { CUDA_CHECK(cudaEventRecord(start, s)); }
  float end(cudaStream_t s) {
    CUDA_CHECK(cudaEventRecord(stop, s));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};
