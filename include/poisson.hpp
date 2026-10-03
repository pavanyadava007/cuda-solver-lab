// Model problem shared by every solver in this repo.
//
//   -Laplace(u) = f   on the unit square, u = 0 on the boundary
//   u(x, y)     = g(x) g(y),  g(t) = t (1 - t) e^t          (analytic solution)
//   f(x, y)     = x (x + 3) e^x g(y) + g(x) y (y + 3) e^y   (since g'' = -t (t + 3) e^t)
//
// Note: the popular choice u = sin(pi x) sin(pi y) is an exact eigenvector of
// the discrete 5-point operator, so CG converges in ONE iteration and every
// benchmark becomes meaningless (see docs/DEVLOG.md). g(x) g(y) is not.
//
// Discretised with the 5-point stencil on an n x n grid of interior points,
// h = 1 / (n + 1). The linear system is scaled by h^2 so that
//
//   A = tridiag-block(4 on the diagonal, -1 for the 4 neighbours),  b = h^2 f.
//
// Unknown (i, j) (0-based, i = column / x, j = row / y) lives at index j * n + i.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace poisson {

struct Csr {
  int nrows = 0;
  std::vector<int> row_ptr;  // nrows + 1
  std::vector<int> col;      // nnz
  std::vector<double> val;   // nnz
  std::size_t nnz() const { return val.size(); }
};

inline double grid_h(int n) { return 1.0 / (n + 1); }

// Assemble the scaled 5-point Laplacian in CSR format (columns sorted per row).
inline Csr build_laplacian_csr(int n) {
  Csr a;
  const std::size_t N = static_cast<std::size_t>(n) * n;
  a.nrows = static_cast<int>(N);
  a.row_ptr.resize(N + 1);
  a.col.reserve(5 * N);
  a.val.reserve(5 * N);
  a.row_ptr[0] = 0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const int k = j * n + i;
      if (j > 0) { a.col.push_back(k - n); a.val.push_back(-1.0); }
      if (i > 0) { a.col.push_back(k - 1); a.val.push_back(-1.0); }
      a.col.push_back(k); a.val.push_back(4.0);
      if (i < n - 1) { a.col.push_back(k + 1); a.val.push_back(-1.0); }
      if (j < n - 1) { a.col.push_back(k + n); a.val.push_back(-1.0); }
      a.row_ptr[k + 1] = static_cast<int>(a.col.size());
    }
  }
  return a;
}

inline double g(double t) { return t * (1.0 - t) * std::exp(t); }
inline double minus_g2(double t) { return t * (t + 3.0) * std::exp(t); }  // -g''(t)

// Right-hand side b = h^2 f at the interior points.
inline std::vector<double> make_rhs(int n) {
  const double h = grid_h(n);
  std::vector<double> b(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 1) * h, y = (j + 1) * h;
      b[static_cast<std::size_t>(j) * n + i] = h * h * (minus_g2(x) * g(y) + g(x) * minus_g2(y));
    }
  return b;
}

inline std::vector<double> analytic_solution(int n) {
  const double h = grid_h(n);
  std::vector<double> u(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      u[static_cast<std::size_t>(j) * n + i] = g((i + 1) * h) * g((j + 1) * h);
  return u;
}

inline double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) m = std::fmax(m, std::fabs(a[k] - b[k]));
  return m;
}

// True relative residual ||b - A x||_2 / ||b||_2, recomputed on the host in
// long double so it does not inherit the rounding of the solver under test.
inline double true_rel_residual(const Csr& a, const std::vector<double>& b,
                                const std::vector<double>& x) {
  long double rr = 0.0L, bb = 0.0L;
  for (int k = 0; k < a.nrows; ++k) {
    long double ax = 0.0L;
    for (int e = a.row_ptr[k]; e < a.row_ptr[k + 1]; ++e)
      ax += static_cast<long double>(a.val[e]) * x[a.col[e]];
    const long double rk = b[k] - ax;
    rr += rk * rk;
    bb += static_cast<long double>(b[k]) * b[k];
  }
  return static_cast<double>(std::sqrt(rr / bb));
}

// A-priori bound on the discretisation error max |u_h - u|. The truncation
// error is tau = (h^2/12)(u_xxxx + u_yyyy) + O(h^4) and the discrete maximum
// principle gives ||A_h^-1||_inf <= 1/8 on the unit square, so
//   max |u_h - u| <= (h^2 / 96) sup |u_xxxx + u_yyyy|.
// With |g''''(t)| = (t^2 + 7t + 8) e^t <= 16 e and max g = g((sqrt5-1)/2) = 0.4380,
// sup |u_xxxx + u_yyyy| <= 2 * 16 e * 0.4380. A 10 % margin covers the O(h^4) term.
// The tests require every solver to land inside this bound (catches a wrong
// stencil, a wrong scaling or an unconverged solve).
inline double discretisation_error_bound(int n) {
  const double h = grid_h(n);
  const double sup_u4 = 2.0 * 16.0 * std::exp(1.0) * 0.4380;
  return 1.1 * sup_u4 / 96.0 * h * h;
}

}  // namespace poisson
