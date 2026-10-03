// CUDA kernels for Conjugate Gradient on the 5-point Poisson problem.
//
// All kernels use a fixed-size grid with grid-stride loops (grid = SMs x
// resident blocks), so every reduction produces at most `gridDim.x` partial
// sums. Reductions finish inside the same kernel with the "last block done"
// pattern (threadfence + atomic ticket): the block that arrives last sums the
// partials and updates the CG scalars (alpha, beta, rr) in device memory. The
// host therefore never has to read a scalar back inside the iteration loop.
#pragma once

#include <cuda_runtime.h>

namespace cgk {

constexpr int kBlock = 256;

// CG scalars live on the device. `one` and `neg_alpha` exist only so cuBLAS
// can be driven in CUBLAS_POINTER_MODE_DEVICE by the library reference path.
struct Scalars {
  double rr;         // r.r of the current residual
  double bb;         // b.b (for the relative residual)
  double pAp;        // p.(A p)
  double alpha;      // rr / pAp
  double neg_alpha;  // -alpha
  double beta;       // rr_new / rr_old
  double rr_new;     // scratch for the library path
  double one;        // 1.0
};

struct GridReduce {
  double* partials;       // gridDim.x entries
  unsigned int* counter;  // must be 0 before every launch; the last block resets it
};

enum class Finalize { kInitRR, kAlpha, kBeta };

__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// Block-wide sum, result valid in thread 0. Safe to call more than once per kernel.
__device__ __forceinline__ double block_sum(double v) {
  __shared__ double warp_part[kBlock / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  __syncthreads();  // protect warp_part from a previous call
  v = warp_sum(v);
  if (lane == 0) warp_part[warp] = v;
  __syncthreads();
  v = (threadIdx.x < kBlock / 32) ? warp_part[lane] : 0.0;
  if (warp == 0) v = warp_sum(v);
  return v;
}

// Grid-wide reduction of `local` + scalar update, finished by the last block.
// Must be reached by every thread of every block (no early return before it).
// Works for 1D and 2D grids; g.partials needs gridDim.x * gridDim.y entries.
template <Finalize F>
__device__ __forceinline__ void grid_reduce(double local, GridReduce g, Scalars* s) {
  __shared__ bool is_last;
  const unsigned int nblocks = gridDim.x * gridDim.y;
  const double bsum = block_sum(local);
  if (threadIdx.x == 0) {
    g.partials[blockIdx.y * gridDim.x + blockIdx.x] = bsum;
    __threadfence();  // make the partial visible before taking a ticket
    const unsigned int ticket = atomicAdd(g.counter, 1u);
    is_last = (ticket == nblocks - 1);
  }
  __syncthreads();
  if (!is_last) return;

  double v = 0.0;
  for (unsigned int b = threadIdx.x; b < nblocks; b += blockDim.x) v += __ldcg(&g.partials[b]);
  v = block_sum(v);
  if (threadIdx.x == 0) {
    if constexpr (F == Finalize::kInitRR) {
      s->rr = v;
      s->bb = v;
      s->beta = 0.0;
      s->one = 1.0;
    } else if constexpr (F == Finalize::kAlpha) {
      s->pAp = v;
      s->alpha = s->rr / v;
      s->neg_alpha = -s->alpha;
    } else {
      s->beta = v / s->rr;
      s->rr = v;
    }
    *g.counter = 0;
  }
}

// ---------------------------------------------------------------- SpMV ----

// (a) CSR scalar: one thread per row. Neighbouring threads read neighbouring
// rows, so col/val loads are strided by the row length (5) -> poor coalescing.
__global__ void spmv_csr_scalar(int nrows, const int* __restrict__ row_ptr,
                                const int* __restrict__ col, const double* __restrict__ val,
                                const double* __restrict__ x, double* __restrict__ y) {
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < nrows;
       row += gridDim.x * blockDim.x) {
    double sum = 0.0;
    for (int e = row_ptr[row]; e < row_ptr[row + 1]; ++e) sum += val[e] * x[col[e]];
    y[row] = sum;
  }
}

// (b) CSR vector: TPR consecutive lanes share one row (TPR = 32 is the classic
// warp-per-row kernel). Lanes read consecutive nonzeros (coalesced) and
// combine with shuffles. With ~5 nonzeros per row, TPR = 32 leaves 27 of 32
// lanes idle, which is why TPR = 4 is also instantiated.
template <int TPR>
__global__ void spmv_csr_vector(int nrows, const int* __restrict__ row_ptr,
                                const int* __restrict__ col, const double* __restrict__ val,
                                const double* __restrict__ x, double* __restrict__ y) {
  static_assert(TPR >= 1 && TPR <= 32 && (TPR & (TPR - 1)) == 0, "TPR must be a power of 2");
  constexpr int kRowsPerWarp = 32 / TPR;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & (TPR - 1);
  const int stride = gridDim.x * blockDim.x / TPR;
  // The loop counter is uniform across the warp (first row of this warp), so
  // all 32 lanes always reach the full-mask shuffle together, even in the tail
  // where some sub-groups have no row left.
  for (int row0 = (tid / 32) * kRowsPerWarp; row0 < nrows; row0 += stride) {
    const int row = row0 + (threadIdx.x & 31) / TPR;
    double sum = 0.0;
    if (row < nrows)
      for (int e = row_ptr[row] + lane; e < row_ptr[row + 1]; e += TPR) sum += val[e] * x[col[e]];
#pragma unroll
    for (int off = TPR / 2; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off, TPR);
    if (lane == 0 && row < nrows) y[row] = sum;
  }
}

// Stencil value of A x at (i, j), zero Dirichlet halo handled by predicates.
__device__ __forceinline__ double apply_stencil(const double* __restrict__ x, int n, int i, int j,
                                                long k) {
  double s = 4.0 * x[k];
  if (i > 0) s -= x[k - 1];
  if (i < n - 1) s -= x[k + 1];
  if (j > 0) s -= x[k - n];
  if (j < n - 1) s -= x[k + n];
  return s;
}

// (c) Matrix-free SpMV: no row_ptr/col/val traffic, only x in and y out.
__global__ void spmv_stencil(int n, const double* __restrict__ x, double* __restrict__ y) {
  const long N = static_cast<long>(n) * n;
  for (long k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x) {
    const int j = static_cast<int>(k / n);
    const int i = static_cast<int>(k - static_cast<long>(j) * n);
    y[k] = apply_stencil(x, n, i, j, k);
  }
}

// ------------------------------------------------- BLAS-1 building blocks ----

// (d) Dot product with warp-shuffle block reduction + in-kernel finalisation.
template <Finalize F>
__global__ void dot_finalize(long N, const double* __restrict__ a, const double* __restrict__ b,
                             GridReduce g, Scalars* s) {
  double local = 0.0;
  for (long k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x)
    local += a[k] * b[k];
  grid_reduce<F>(local, g, s);
}

// y += sign * alpha * x  (alpha read from device memory)
__global__ void axpy_alpha(long N, double sign, const Scalars* __restrict__ s,
                           const double* __restrict__ x, double* __restrict__ y) {
  const double a = sign * s->alpha;
  for (long k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x)
    y[k] += a * x[k];
}

// p = r + beta * p
__global__ void update_p(long N, const Scalars* __restrict__ s, const double* __restrict__ r,
                         double* __restrict__ p) {
  const double beta = s->beta;
  for (long k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x)
    p[k] = r[k] + beta * p[k];
}

// ------------------------------------------------------ fused CG (2 kernels) ----

// (e1) p_new = r + beta p_old, Ap = A p_new, and p_new.Ap, in one pass.
// p_new at the 4 neighbours is recomputed from r and p_old (cheap FLOPs, the
// loads hit L1/L2), so the separate "p = r + beta p" pass disappears. p is
// double-buffered (p_old -> p_new) because neighbours must see the old value.
__global__ void fused_update_p_spmv_dot(int n, const double* __restrict__ r,
                                        const double* __restrict__ p_old,
                                        double* __restrict__ p_new, double* __restrict__ Ap,
                                        GridReduce g, Scalars* s) {
  const long N = static_cast<long>(n) * n;
  const double beta = s->beta;
  double local = 0.0;
  for (long k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x) {
    const int j = static_cast<int>(k / n);
    const int i = static_cast<int>(k - static_cast<long>(j) * n);
    const double pc = r[k] + beta * p_old[k];
    double ap = 4.0 * pc;
    if (i > 0) ap -= r[k - 1] + beta * p_old[k - 1];
    if (i < n - 1) ap -= r[k + 1] + beta * p_old[k + 1];
    if (j > 0) ap -= r[k - n] + beta * p_old[k - n];
    if (j < n - 1) ap -= r[k + n] + beta * p_old[k + n];
    p_new[k] = pc;
    Ap[k] = ap;
    local += pc * ap;
  }
  grid_reduce<Finalize::kAlpha>(local, g, s);
}

// (e1') Same maths as (e1), restructured after Nsight Compute showed (e1)
// reading ~2x its model bytes from DRAM (the k +- n neighbours were not
// reliably found in L2). Each thread owns one column of a kStripRows-high
// strip and walks down it: the vertical neighbours of p_new stay in registers
// (up / mid / down), the horizontal ones come from the neighbouring lanes via
// warp shuffles, so every r and p_old value is read from DRAM about once.
// Grid: (ceil(n / blockDim.x), ceil(n / strip_rows)); blockDim.x % 32 == 0.
// The host picks strip_rows (<= kMaxStripRows) so that small grids still
// launch enough blocks to fill the GPU.
constexpr int kMaxStripRows = 32;

__global__ void fused_update_p_spmv_dot_rows(int n, int strip_rows, const double* __restrict__ r,
                                             const double* __restrict__ p_old,
                                             double* __restrict__ p_new, double* __restrict__ Ap,
                                             GridReduce g, Scalars* s) {
  const double beta = s->beta;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int j0 = blockIdx.y * strip_rows;
  const int j1 = min(j0 + strip_rows, n);
  // p_new at (ii, jj); zero outside the domain (Dirichlet), incl. columns >= n.
  auto pn = [&](int ii, int jj) -> double {
    if (ii < 0 || ii >= n || jj < 0 || jj >= n) return 0.0;
    const long kk = static_cast<long>(jj) * n + ii;
    return r[kk] + beta * p_old[kk];
  };
  double up = pn(i, j0 - 1);
  double mid = pn(i, j0);
  double local = 0.0;
  for (int j = j0; j < j1; ++j) {  // block-uniform trip count: shuffles are safe
    const double down = pn(i, j + 1);
    double left = __shfl_up_sync(0xffffffffu, mid, 1);
    double right = __shfl_down_sync(0xffffffffu, mid, 1);
    if (lane == 0) left = pn(i - 1, j);
    if (lane == 31) right = pn(i + 1, j);
    if (i < n) {
      const long k = static_cast<long>(j) * n + i;
      const double ap = 4.0 * mid - left - right - up - down;
      p_new[k] = mid;
      Ap[k] = ap;
      local += mid * ap;
    }
    up = mid;
    mid = down;
  }
  grid_reduce<Finalize::kAlpha>(local, g, s);
}

// (e2) x += alpha p, r -= alpha Ap, and r.r, in one pass.
__global__ void fused_update_xr_dot(long N, const double* __restrict__ p,
                                    const double* __restrict__ Ap, double* __restrict__ x,
                                    double* __restrict__ r, GridReduce g, Scalars* s) {
  const double alpha = s->alpha;
  double local = 0.0;
  for (long k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += gridDim.x * blockDim.x) {
    x[k] += alpha * p[k];
    const double rk = r[k] - alpha * Ap[k];
    r[k] = rk;
    local += rk * rk;
  }
  grid_reduce<Finalize::kBeta>(local, g, s);
}

// Scalar helpers for the cuBLAS reference path (pointer mode device).
__global__ void lib_alpha(Scalars* s) {
  s->alpha = s->rr / s->pAp;
  s->neg_alpha = -s->alpha;
}
__global__ void lib_beta(Scalars* s) {
  s->beta = s->rr_new / s->rr;
  s->rr = s->rr_new;
}

}  // namespace cgk
