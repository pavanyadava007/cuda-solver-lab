// Measures the practical ceilings used in the roofline: DRAM bandwidth
// (copy / read / write / triad kernels + cudaMemcpy D2D) and FP64 / FP32 FMA
// throughput. Arrays are 512 MiB each, far larger than the 48 MiB L2.
//
//   bw_probe --csv results/raw/peaks.csv
#include <algorithm>
#include <cstdio>
#include <string>

#include "common.hpp"
#include "cuda_check.cuh"

namespace {

constexpr int kBlock = 256;

__global__ void copy_f64(long n, const double* __restrict__ a, double* __restrict__ b) {
  for (long k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) b[k] = a[k];
}

__global__ void copy_f64x2(long n2, const double2* __restrict__ a, double2* __restrict__ b) {
  for (long k = blockIdx.x * blockDim.x + threadIdx.x; k < n2; k += gridDim.x * blockDim.x) b[k] = a[k];
}

__global__ void triad_f64(long n, const double* __restrict__ a, const double* __restrict__ b,
                          double* __restrict__ c) {
  for (long k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x)
    c[k] = a[k] + 3.0 * b[k];
}

__global__ void read_f64(long n, const double* __restrict__ a, double* __restrict__ sink) {
  double s = 0.0;
  for (long k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) s += a[k];
  if (s == 123.456) *sink = s;  // never true for our data; keeps the loads alive
}

__global__ void write_f64(long n, double* __restrict__ a) {
  for (long k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) a[k] = 1.0;
}

// 8 independent FMA chains per thread, `iters` rounds: 2 * 8 * iters flops/thread.
template <typename T>
__global__ void fma_peak(int iters, T seed, T* __restrict__ out) {
  T a0 = seed + threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
  T a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  const T m = static_cast<T>(0.999999), c = static_cast<T>(1e-7);
  for (int i = 0; i < iters; ++i) {
    a0 = a0 * m + c; a1 = a1 * m + c; a2 = a2 * m + c; a3 = a3 * m + c;
    a4 = a4 * m + c; a5 = a5 * m + c; a6 = a6 * m + c; a7 = a7 * m + c;
  }
  const T s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  if (s == static_cast<T>(-1)) out[0] = s;
}

const char* kHeader = "date,device,probe,bytes_or_flops,best_ms,value,unit";

void emit(const std::string& csv, const char* probe, double amount, float ms, double value,
          const char* unit) {
  char row[512];
  std::snprintf(row, sizeof row, "%s,%s,%s,%.6e,%.4f,%.2f,%s", common::today_iso().c_str(),
                device_name(), probe, amount, ms, value, unit);
  std::printf("%s\n", row);
  common::append_csv(csv, kHeader, row);
}

template <typename F>
float best_of(int reps, F&& f) {
  EventTimer t;
  f();  // warm-up
  float best = 1e30f;
  for (int r = 0; r < reps; ++r) {
    t.begin(0);
    f();
    best = std::min(best, t.end(0));
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  const common::Args args(argc, argv);
  const std::string csv = args.str("--csv", "");
  const int reps = static_cast<int>(args.num("--reps", 20));
  const long n = 64L << 20;  // 64 Mi doubles = 512 MiB per array
  const size_t bytes = n * sizeof(double);

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  const int grid = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / kBlock);

  double *a, *b, *c, *sink;
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  CUDA_CHECK(cudaMalloc(&c, bytes));
  CUDA_CHECK(cudaMalloc(&sink, 64));
  CUDA_CHECK(cudaMemset(a, 0, bytes));
  CUDA_CHECK(cudaMemset(b, 0, bytes));

  std::printf("%s\n", kHeader);
  float ms;
  ms = best_of(reps, [&] { copy_f64<<<grid, kBlock>>>(n, a, b); });
  emit(csv, "copy_f64", 2.0 * bytes, ms, 2.0 * bytes / ms / 1e6, "GB/s");
  ms = best_of(reps, [&] {
    copy_f64x2<<<grid, kBlock>>>(n / 2, reinterpret_cast<double2*>(a), reinterpret_cast<double2*>(b));
  });
  emit(csv, "copy_f64x2", 2.0 * bytes, ms, 2.0 * bytes / ms / 1e6, "GB/s");
  ms = best_of(reps, [&] { triad_f64<<<grid, kBlock>>>(n, a, b, c); });
  emit(csv, "triad_f64", 3.0 * bytes, ms, 3.0 * bytes / ms / 1e6, "GB/s");
  ms = best_of(reps, [&] { read_f64<<<grid, kBlock>>>(n, a, sink); });
  emit(csv, "read_f64", 1.0 * bytes, ms, 1.0 * bytes / ms / 1e6, "GB/s");
  ms = best_of(reps, [&] { write_f64<<<grid, kBlock>>>(n, a); });
  emit(csv, "write_f64", 1.0 * bytes, ms, 1.0 * bytes / ms / 1e6, "GB/s");
  ms = best_of(reps, [&] { CUDA_CHECK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice)); });
  emit(csv, "memcpy_d2d", 2.0 * bytes, ms, 2.0 * bytes / ms / 1e6, "GB/s");
  CUDA_CHECK_LAUNCH();

  // Compute peaks: enough blocks to fill every SM several times.
  const int fgrid = prop.multiProcessorCount * 32;
  const int iters = 1 << 14;
  const double flops = 2.0 * 8.0 * iters * double(fgrid) * kBlock;
  ms = best_of(5, [&] { fma_peak<double><<<fgrid, kBlock>>>(iters, 1.0, reinterpret_cast<double*>(sink)); });
  emit(csv, "fma_f64", flops, ms, flops / ms / 1e6, "GFLOP/s");
  ms = best_of(5, [&] { fma_peak<float><<<fgrid, kBlock>>>(iters, 1.0f, reinterpret_cast<float*>(sink)); });
  emit(csv, "fma_f32", flops, ms, flops / ms / 1e6, "GFLOP/s");
  CUDA_CHECK_LAUNCH();

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  cudaFree(sink);
  return 0;
}
