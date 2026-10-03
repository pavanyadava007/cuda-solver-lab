// Jacobi sweeps for the same Poisson problem: three GPU kernels + CPU OpenMP.
//
//   u_new(i,j) = ( h^2 f(i,j) + u(i-1,j) + u(i+1,j) + u(i,j-1) + u(i,j+1) ) / 4
//
// Arrays are (n+2) x (n+2) with a zero halo (the Dirichlet boundary), so the
// kernels need no boundary branches inside the domain.
//
//   jacobi --n 4096 --sweeps 200 --csv results/raw/jacobi.csv   (benchmark)
//   jacobi --n 257 --check                                       (GPU vs CPU)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <omp.h>

#include "common.hpp"
#include "cuda_check.cuh"
#include "poisson.hpp"

namespace {

constexpr int kTx = 32, kTy = 8;   // 2D block for naive / shared-memory kernels
constexpr int kRowsPerThread = 16;  // register-streaming kernel: rows walked per thread
constexpr int kStreamBlock = 128;   // register-streaming kernel: threads per block (x only)

// (1) Naive: every neighbour load goes to global memory (served by L1/L2).
__global__ void jacobi_naive(int n, const double* __restrict__ rhs, const double* __restrict__ u,
                             double* __restrict__ u_new) {
  const int W = n + 2;
  const int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int j = 1 + blockIdx.y * blockDim.y + threadIdx.y;
  if (i > n || j > n) return;
  const long k = static_cast<long>(j) * W + i;
  u_new[k] = 0.25 * (rhs[k] + u[k - 1] + u[k + 1] + u[k - W] + u[k + W]);
}

// (2) Shared-memory tile with a 1-point halo: each u value is read from DRAM
// once per block and then served from shared memory to its 4 neighbours.
__global__ void jacobi_smem(int n, const double* __restrict__ rhs, const double* __restrict__ u,
                            double* __restrict__ u_new) {
  __shared__ double tile[kTy + 2][kTx + 2];
  const int W = n + 2;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int i = 1 + blockIdx.x * kTx + tx;  // may exceed n in the last block
  const int j = 1 + blockIdx.y * kTy + ty;
  const bool inside = (i <= n && j <= n);
  // Clamp loads to the padded array so threads past the edge stay in bounds.
  const int ic = min(i, n + 1), jc = min(j, n + 1);
  const long k = static_cast<long>(jc) * W + ic;

  // Threads past the right/top edge load the (zero) halo column/row as their
  // "centre", which is exactly the neighbour the last interior thread needs.
  tile[ty + 1][tx + 1] = u[k];
  if (tx == 0) tile[ty + 1][0] = u[k - 1];
  if (tx == kTx - 1) tile[ty + 1][kTx + 1] = u[static_cast<long>(jc) * W + min(ic + 1, n + 1)];
  if (ty == 0) tile[0][tx + 1] = u[k - W];
  if (ty == kTy - 1) tile[kTy + 1][tx + 1] = u[static_cast<long>(min(jc + 1, n + 1)) * W + ic];
  __syncthreads();

  if (!inside) return;
  u_new[k] = 0.25 * (rhs[k] + tile[ty + 1][tx] + tile[ty + 1][tx + 2] + tile[ty][tx + 1] +
                     tile[ty + 2][tx + 1]);
}

// (3) Register streaming: one thread per column walks kRowsPerThread rows and
// keeps the (j-1, j, j+1) values in registers, so each u value is loaded once
// vertically; horizontal neighbours come through L1.
__global__ void jacobi_regs(int n, const double* __restrict__ rhs, const double* __restrict__ u,
                            double* __restrict__ u_new) {
  const int W = n + 2;
  const int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
  const int j0 = 1 + blockIdx.y * kRowsPerThread;
  if (i > n) return;
  const int j_end = min(j0 + kRowsPerThread - 1, n);
  double up = u[static_cast<long>(j0 - 1) * W + i];
  double mid = u[static_cast<long>(j0) * W + i];
  for (int j = j0; j <= j_end; ++j) {
    const long k = static_cast<long>(j) * W + i;
    const double down = u[k + W];
    u_new[k] = 0.25 * (rhs[k] + u[k - 1] + u[k + 1] + up + down);
    up = mid;
    mid = down;
  }
}

enum class Kernel { kNaive, kSmem, kRegs };
const struct {
  const char* name;
  Kernel kind;
} kKernels[] = {{"naive", Kernel::kNaive}, {"smem_tile", Kernel::kSmem}, {"regs_stream", Kernel::kRegs}};

void launch(Kernel kind, int n, const double* rhs, const double* u, double* u_new) {
  switch (kind) {
    case Kernel::kNaive: {
      const dim3 block(kTx, kTy), grid((n + kTx - 1) / kTx, (n + kTy - 1) / kTy);
      jacobi_naive<<<grid, block>>>(n, rhs, u, u_new);
      break;
    }
    case Kernel::kSmem: {
      const dim3 block(kTx, kTy), grid((n + kTx - 1) / kTx, (n + kTy - 1) / kTy);
      jacobi_smem<<<grid, block>>>(n, rhs, u, u_new);
      break;
    }
    case Kernel::kRegs: {
      const dim3 grid((n + kStreamBlock - 1) / kStreamBlock, (n + kRowsPerThread - 1) / kRowsPerThread);
      jacobi_regs<<<grid, kStreamBlock>>>(n, rhs, u, u_new);
      break;
    }
  }
  CUDA_CHECK_LAUNCH();
}

// CPU reference / baseline with OpenMP.
void jacobi_cpu(int n, int sweeps, const std::vector<double>& rhs, std::vector<double>& u) {
  const int W = n + 2;
  std::vector<double> v(u);
  for (int s = 0; s < sweeps; ++s) {
#pragma omp parallel for schedule(static)
    for (int j = 1; j <= n; ++j)
      for (int i = 1; i <= n; ++i) {
        const long k = static_cast<long>(j) * W + i;
        v[k] = 0.25 * (rhs[k] + u[k - 1] + u[k + 1] + u[k - W] + u[k + W]);
      }
    u.swap(v);
  }
}

std::vector<double> padded_rhs(int n) {
  const std::vector<double> b = poisson::make_rhs(n);
  const int W = n + 2;
  std::vector<double> r(static_cast<size_t>(W) * W, 0.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) r[static_cast<size_t>(j + 1) * W + i + 1] = b[static_cast<size_t>(j) * n + i];
  return r;
}

const char* kHeader =
    "date,device,impl,kernel,n,sweeps,time_s,ms_per_sweep,mlups,model_bytes_per_point,eff_gbs,max_diff_vs_cpu";

}  // namespace

int main(int argc, char** argv) {
  const common::Args args(argc, argv);
  const int n = static_cast<int>(args.num("--n", 1024));
  const bool check = args.has("--check");
  const int sweeps = static_cast<int>(args.num("--sweeps", check ? 500 : 200));
  const int reps = static_cast<int>(args.num("--reps", 3));
  const bool cpu = args.has("--cpu");
  const std::string csv = args.str("--csv", "");

  const int W = n + 2;
  const size_t cells = static_cast<size_t>(W) * W;
  const std::vector<double> rhs = padded_rhs(n);
  const double points = static_cast<double>(n) * n;
  const double bytes_per_point = 24.0;  // read u, read rhs, write u_new (ideal reuse)

  std::vector<double> u_ref;
  if (check) {
    u_ref.assign(cells, 0.0);
    jacobi_cpu(n, sweeps, rhs, u_ref);
  }

  std::printf("%s\n", kHeader);
  if (cpu) {  // CPU OpenMP timing (same arithmetic)
    std::vector<double> u(cells, 0.0);
    jacobi_cpu(n, 2, rhs, u);
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
      std::fill(u.begin(), u.end(), 0.0);
      const double t0 = common::wall_seconds();
      jacobi_cpu(n, sweeps, rhs, u);
      best = std::min(best, common::wall_seconds() - t0);
    }
    char row[512];
    std::snprintf(row, sizeof row, "%s,AMD EPYC 7R13,cpu_openmp_%dt,cpu,%d,%d,%.6e,%.6e,%.2f,%.0f,%.3f,-1",
                  common::today_iso().c_str(), omp_get_max_threads(), n, sweeps, best,
                  1e3 * best / sweeps, points * sweeps / best / 1e6, bytes_per_point,
                  bytes_per_point * points * sweeps / best / 1e9);
    std::printf("%s\n", row);
    common::append_csv(csv, kHeader, row);
  }

  double *d_rhs, *d_u, *d_v;
  CUDA_CHECK(cudaMalloc(&d_rhs, cells * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_u, cells * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_v, cells * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_rhs, rhs.data(), cells * sizeof(double), cudaMemcpyHostToDevice));

  EventTimer timer;
  bool ok = true;
  for (const auto& kern : kKernels) {
    auto run = [&](int count) {
      CUDA_CHECK(cudaMemset(d_u, 0, cells * sizeof(double)));
      CUDA_CHECK(cudaMemset(d_v, 0, cells * sizeof(double)));  // halo of both buffers = 0
      double *a = d_u, *b = d_v;
      for (int s = 0; s < count; ++s) {
        launch(kern.kind, n, d_rhs, a, b);
        std::swap(a, b);
      }
      return a;  // buffer holding the latest iterate
    };
    run(2);  // warm-up
    float best_ms = 1e30f;
    double* result = nullptr;
    for (int r = 0; r < reps; ++r) {
      CUDA_CHECK(cudaDeviceSynchronize());
      timer.begin(0);
      result = run(sweeps);
      best_ms = std::min(best_ms, timer.end(0));
    }
    double max_diff = -1.0;
    if (check) {
      std::vector<double> u(cells);
      CUDA_CHECK(cudaMemcpy(u.data(), result, cells * sizeof(double), cudaMemcpyDeviceToHost));
      max_diff = poisson::max_abs_diff(u, u_ref);
      const bool pass = max_diff < 1e-12;
      ok &= pass;
      std::fprintf(stderr, "CHECK jacobi %-12s %s max_diff_vs_cpu=%.3e after %d sweeps (n=%d)\n",
                   kern.name, pass ? "PASS" : "FAIL", max_diff, sweeps, n);
    }
    const double t = best_ms * 1e-3;
    char row[512];
    std::snprintf(row, sizeof row, "%s,%s,cuda,%s,%d,%d,%.6e,%.6e,%.2f,%.0f,%.3f,%.3e",
                  common::today_iso().c_str(), device_name(), kern.name, n, sweeps, t,
                  1e3 * t / sweeps, points * sweeps / t / 1e6, bytes_per_point,
                  bytes_per_point * points * sweeps / t / 1e9, max_diff);
    std::printf("%s\n", row);
    common::append_csv(csv, kHeader, row);
  }
  cudaFree(d_rhs);
  cudaFree(d_u);
  cudaFree(d_v);
  return ok ? 0 : 1;
}
