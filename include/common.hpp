// Small shared helpers: wall clock, command line flags, CSV output.
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

namespace common {

inline double wall_seconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Minimal "--key value" / "--flag" parser; enough for the benchmark drivers.
class Args {
 public:
  Args(int argc, char** argv) : argc_(argc), argv_(argv) {}
  bool has(const char* key) const { return find(key) >= 0; }
  std::string str(const char* key, const std::string& def) const {
    const int i = find(key);
    return (i >= 0 && i + 1 < argc_) ? std::string(argv_[i + 1]) : def;
  }
  long num(const char* key, long def) const {
    const int i = find(key);
    return (i >= 0 && i + 1 < argc_) ? std::strtol(argv_[i + 1], nullptr, 10) : def;
  }
  double real(const char* key, double def) const {
    const int i = find(key);
    return (i >= 0 && i + 1 < argc_) ? std::strtod(argv_[i + 1], nullptr) : def;
  }

 private:
  int find(const char* key) const {
    for (int i = 1; i < argc_; ++i)
      if (std::strcmp(argv_[i], key) == 0) return i;
    return -1;
  }
  int argc_;
  char** argv_;
};

inline std::string today_iso() {
  char buf[16];
  const std::time_t t = std::time(nullptr);
  std::strftime(buf, sizeof buf, "%Y-%m-%d", std::localtime(&t));
  return buf;
}

// Appends one CSV row to `path` (writes the header first if the file is new).
inline void append_csv(const std::string& path, const std::string& header,
                       const std::string& row) {
  if (path.empty()) return;
  FILE* probe = std::fopen(path.c_str(), "r");
  const bool fresh = (probe == nullptr);
  if (probe) std::fclose(probe);
  FILE* f = std::fopen(path.c_str(), "a");
  if (!f) { std::perror(path.c_str()); std::exit(2); }
  if (fresh) std::fprintf(f, "%s\n", header.c_str());
  std::fprintf(f, "%s\n", row.c_str());
  std::fclose(f);
}

}  // namespace common

namespace common {

// One schema for every CG implementation (C++ CPU, CUDA, Fortran OpenACC) so
// the report script can join them. "mode" is "fixed" (benchmark: exactly
// `iters` iterations, no convergence checks) or "tol" (solve to tolerance).
inline const char* kCgCsvHeader =
    "date,device,impl,variant,n,unknowns,nnz,threads,mode,iters,time_s,ms_per_iter,"
    "model_bytes_per_iter,eff_gbs,gflops,rel_res_true,max_err_analytic,max_err_vs_ref,"
    "launches_per_iter";

// Algorithmic flops of one CG iteration: SpMV (2 per nonzero), two dot
// products (2N each) and three axpy-type updates (2N each). Used for every
// variant, so GFLOP/s numbers are comparable even when a variant is matrix-free.
inline double cg_flops_per_iter(double unknowns, double nnz) { return 2.0 * nnz + 10.0 * unknowns; }

struct CgRecord {
  std::string device, impl, variant, mode;
  int n = 0;
  long unknowns = 0, nnz = 0;
  int threads = 0;
  int iters = 0;
  double time_s = 0.0;
  double model_bytes_per_iter = 0.0;  // minimum DRAM traffic of this variant
  double rel_res_true = -1.0;         // -1 = not computed
  double max_err_analytic = -1.0;
  double max_err_vs_ref = -1.0;
  int launches_per_iter = 0;

  std::string csv() const {
    const double ms = iters > 0 ? 1e3 * time_s / iters : 0.0;
    const double gbs = iters > 0 ? model_bytes_per_iter * iters / time_s / 1e9 : 0.0;
    const double gfl = iters > 0 ? cg_flops_per_iter(unknowns, nnz) * iters / time_s / 1e9 : 0.0;
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "%s,%s,%s,%s,%d,%ld,%ld,%d,%s,%d,%.6e,%.6e,%.6e,%.4f,%.4f,%.3e,%.3e,%.3e,%d",
                  today_iso().c_str(), device.c_str(), impl.c_str(), variant.c_str(), n,
                  unknowns, nnz, threads, mode.c_str(), iters, time_s, ms,
                  model_bytes_per_iter, gbs, gfl, rel_res_true, max_err_analytic,
                  max_err_vs_ref, launches_per_iter);
    return buf;
  }
};

}  // namespace common
