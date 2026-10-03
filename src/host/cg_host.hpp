// CPU Conjugate Gradient on a CSR matrix.
//
// The same source is compiled twice: without -fopenmp (the serial reference,
// pragmas are ignored) and with -fopenmp (the multithreaded baseline). Each
// building block is a separate loop, i.e. the textbook "unfused" CG:
//   q = A p; alpha = rr / (p.q); x += alpha p; r -= alpha q; rr' = r.r; p = r + beta p
#pragma once

#include <cmath>
#include <vector>

#include "poisson.hpp"

namespace cg_host {

inline void spmv(const poisson::Csr& a, const double* x, double* y) {
  const int* rp = a.row_ptr.data();
  const int* ci = a.col.data();
  const double* va = a.val.data();
#pragma omp parallel for schedule(static)
  for (int k = 0; k < a.nrows; ++k) {
    double s = 0.0;
    for (int e = rp[k]; e < rp[k + 1]; ++e) s += va[e] * x[ci[e]];
    y[k] = s;
  }
}

inline double dot(long n, const double* a, const double* b) {
  double s = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : s)
  for (long k = 0; k < n; ++k) s += a[k] * b[k];
  return s;
}

inline void axpy(long n, double alpha, const double* x, double* y) {  // y += alpha x
#pragma omp parallel for schedule(static)
  for (long k = 0; k < n; ++k) y[k] += alpha * x[k];
}

inline void xpay(long n, const double* x, double beta, double* y) {  // y = x + beta y
#pragma omp parallel for schedule(static)
  for (long k = 0; k < n; ++k) y[k] = x[k] + beta * y[k];
}

// Bytes moved per iteration by the loops above, assuming perfect reuse of the
// SpMV source vector (row_ptr 4 B/row, col 4 B/nz, val 8 B/nz, x and y 8 B/row):
//   SpMV 12 nnz + 20 N, p.q 16 N, two axpy 2*24 N, r.r 8 N, xpay 24 N.
inline double model_bytes_per_iter(double unknowns, double nnz) {
  return 12.0 * nnz + 20.0 * unknowns + 96.0 * unknowns;
}

struct Result {
  int iters = 0;
  double rel_res = 0.0;  // recurrence residual sqrt(rr / bb)
};

// Solves A x = b from x = 0. If fixed_iters is true it runs exactly maxit
// iterations (benchmark mode); otherwise it stops when sqrt(rr/bb) < tol.
inline Result solve(const poisson::Csr& a, const std::vector<double>& b, std::vector<double>& x,
                    int maxit, double tol, bool fixed_iters) {
  const long n = a.nrows;
  std::vector<double> r(n), p(n), q(n);
  x.assign(n, 0.0);
#pragma omp parallel for schedule(static)
  for (long k = 0; k < n; ++k) {  // parallel first touch
    r[k] = b[k];
    p[k] = b[k];
    q[k] = 0.0;
  }
  const double bb = dot(n, b.data(), b.data());
  double rr = bb;
  Result res;
  for (int it = 0; it < maxit; ++it) {
    if (!fixed_iters && std::sqrt(rr / bb) < tol) break;
    spmv(a, p.data(), q.data());
    const double alpha = rr / dot(n, p.data(), q.data());
    axpy(n, alpha, p.data(), x.data());
    axpy(n, -alpha, q.data(), r.data());
    const double rr_new = dot(n, r.data(), r.data());
    xpay(n, r.data(), rr_new / rr, p.data());
    rr = rr_new;
    res.iters = it + 1;
  }
  res.rel_res = std::sqrt(rr / bb);
  return res;
}

}  // namespace cg_host
