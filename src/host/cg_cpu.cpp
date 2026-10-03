// CPU CG driver. Built as cg_serial (no OpenMP) and cg_omp (-fopenmp).
//
//   cg_serial --n 1024 --mode fixed --iters 50 --csv results/raw/cg.csv
//   cg_omp    --n 512  --mode tol   --tol 1e-10
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cg_host.hpp"
#include "common.hpp"
#include "poisson.hpp"

int main(int argc, char** argv) {
  const common::Args args(argc, argv);
  const int n = static_cast<int>(args.num("--n", 512));
  const std::string mode = args.str("--mode", "tol");
  const int maxit = static_cast<int>(args.num("--iters", 100000));
  const double tol = args.real("--tol", 1e-10);
  const int reps = static_cast<int>(args.num("--reps", 3));
  const std::string csv = args.str("--csv", "");
  const bool fixed = (mode == "fixed");

#ifdef _OPENMP
  const int threads = omp_get_max_threads();
  const char* impl = "cpu_openmp";
#else
  const int threads = 1;
  const char* impl = "cpu_serial";
#endif

  const poisson::Csr a = poisson::build_laplacian_csr(n);
  const std::vector<double> b = poisson::make_rhs(n);
  std::vector<double> x;

  // Untimed warm-up (page faults, thread pool start-up), then best of `reps`.
  cg_host::solve(a, b, x, fixed ? 2 : maxit, tol, fixed);
  double best = 1e300;
  cg_host::Result res;
  for (int r = 0; r < (fixed ? reps : 1); ++r) {
    const double t0 = common::wall_seconds();
    res = cg_host::solve(a, b, x, maxit, tol, fixed);
    best = std::min(best, common::wall_seconds() - t0);
  }

  common::CgRecord rec;
  rec.device = "AMD EPYC 7R13";
  rec.impl = impl;
  rec.variant = "csr";
  rec.mode = mode;
  rec.n = n;
  rec.unknowns = a.nrows;
  rec.nnz = static_cast<long>(a.nnz());
  rec.threads = threads;
  rec.iters = res.iters;
  rec.time_s = best;
  rec.model_bytes_per_iter = cg_host::model_bytes_per_iter(a.nrows, a.nnz());
  rec.rel_res_true = poisson::true_rel_residual(a, b, x);
  rec.max_err_analytic = poisson::max_abs_diff(x, poisson::analytic_solution(n));
  rec.launches_per_iter = 0;
  std::printf("%s\n%s\n", common::kCgCsvHeader, rec.csv().c_str());
  common::append_csv(csv, common::kCgCsvHeader, rec.csv());

  if (!fixed) {
    const bool ok = rec.rel_res_true < 10 * tol &&
                    rec.max_err_analytic < poisson::discretisation_error_bound(n);
    std::printf("CHECK %s: rel_res_true=%.3e (tol %.1e), max_err=%.3e (bound %.3e)\n",
                ok ? "PASS" : "FAIL", rec.rel_res_true, tol, rec.max_err_analytic,
                poisson::discretisation_error_bound(n));
    return ok ? 0 : 1;
  }
  return 0;
}
