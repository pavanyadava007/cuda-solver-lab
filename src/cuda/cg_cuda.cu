// CUDA CG driver: runs one or all GPU variants on the 5-point Poisson problem.
//
//   cg_cuda --variant all --n 256 --mode tol --tol 1e-10 --ref      (correctness)
//   cg_cuda --variant fused_graph --n 4096 --mode fixed --iters 200  (benchmark)
//
// Variants
//   csr_scalar    CSR SpMV, one thread per row            + 5 BLAS-1 kernels
//   csr_vector32  CSR SpMV, one warp per row              + 5 BLAS-1 kernels
//   csr_vector4   CSR SpMV, 4 lanes per row               + 5 BLAS-1 kernels
//   stencil       matrix-free SpMV                        + 5 BLAS-1 kernels
//   fused         2 kernels per iteration (see cg_kernels.cuh, e1/e2)
//   fused_rows    as fused, first kernel walks row strips (e1', fixes DRAM re-reads)
//   *_graph       same, loop captured into a CUDA Graph
//   cusparse      cusparseSpMV (CSR) + cuBLAS ddot/daxpy/dscal (library reference)
#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cusparse.h>

#include "cg_host.hpp"
#include "cg_kernels.cuh"
#include "common.hpp"
#include "cuda_check.cuh"
#include "poisson.hpp"

#define CUBLAS_CHECK(call)                                                       \
  do {                                                                           \
    const cublasStatus_t st_ = (call);                                           \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                          \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", st_, __FILE__, __LINE__); \
      std::exit(3);                                                              \
    }                                                                            \
  } while (0)
#define CUSPARSE_CHECK(call)                                                       \
  do {                                                                             \
    const cusparseStatus_t st_ = (call);                                           \
    if (st_ != CUSPARSE_STATUS_SUCCESS) {                                          \
      std::fprintf(stderr, "cuSPARSE error %s at %s:%d\n", cusparseGetErrorString(st_), \
                   __FILE__, __LINE__);                                            \
      std::exit(3);                                                                \
    }                                                                              \
  } while (0)

namespace {

enum class Variant { kCsrScalar, kCsrVector32, kCsrVector4, kStencil, kFused, kFusedRows, kCusparse };

struct VariantSpec {
  const char* name;
  Variant kind;
  bool graph;
};

const VariantSpec kVariants[] = {
    {"csr_scalar", Variant::kCsrScalar, false},  {"csr_vector32", Variant::kCsrVector32, false},
    {"csr_vector4", Variant::kCsrVector4, false}, {"stencil", Variant::kStencil, false},
    {"stencil_graph", Variant::kStencil, true},   {"fused", Variant::kFused, false},
    {"fused_graph", Variant::kFused, true},       {"fused_rows", Variant::kFusedRows, false},
    {"fused_rows_graph", Variant::kFusedRows, true}, {"cusparse", Variant::kCusparse, false},
};

// Minimum DRAM bytes per CG iteration for each variant (perfect cache reuse of
// stencil / SpMV source vectors; see README "Traffic model").
double model_bytes(Variant v, double N, double nnz) {
  const double csr_spmv = 12.0 * nnz + 20.0 * N;  // row_ptr + col + val + x + y
  const double blas1 = 16.0 * N + 2 * 24.0 * N + 8.0 * N + 24.0 * N;  // p.q, 2 axpy, r.r, xpay
  switch (v) {
    case Variant::kCsrScalar:
    case Variant::kCsrVector32:
    case Variant::kCsrVector4: return csr_spmv + blas1;
    case Variant::kStencil: return 16.0 * N + blas1;
    case Variant::kFused:
    case Variant::kFusedRows: return 32.0 * N + 48.0 * N;
    case Variant::kCusparse: return csr_spmv + 16.0 * N + 48.0 * N + 8.0 * N + 16.0 * N + 24.0 * N;
  }
  return 0.0;
}

// Host-side API calls per iteration (kernel launches or library calls).
int launches_per_iter(Variant v) {
  switch (v) {
    case Variant::kFused:
    case Variant::kFusedRows: return 2;
    case Variant::kCusparse: return 9;  // spmv, ddot, alpha, 2 daxpy, ddot, beta, dscal, daxpy
    default: return 6;
  }
}

class GpuCg {
 public:
  GpuCg(int n, const poisson::Csr& a, const std::vector<double>& b) : n_(n), N_(a.nrows) {
    nnz_ = static_cast<long>(a.nnz());
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    grid_ = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / cgk::kBlock);
    // Row-strip kernel: up to 32 rows per strip, fewer when that would leave
    // less than ~8 blocks per SM (e.g. n = 256 gets 1-row strips).
    const int col_blocks = (n + cgk::kBlock - 1) / cgk::kBlock;
    strip_rows_ = std::clamp(n * col_blocks / (8 * prop.multiProcessorCount), 1, cgk::kMaxStripRows);
    rows_grid_ = dim3(col_blocks, (n + strip_rows_ - 1) / strip_rows_);
    const size_t max_blocks = std::max<size_t>(grid_, size_t(rows_grid_.x) * rows_grid_.y);

    const size_t vb = N_ * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_b_, vb));
    CUDA_CHECK(cudaMalloc(&d_x_, vb));
    CUDA_CHECK(cudaMalloc(&d_r_, vb));
    CUDA_CHECK(cudaMalloc(&d_p_, vb));
    CUDA_CHECK(cudaMalloc(&d_p2_, vb));
    CUDA_CHECK(cudaMalloc(&d_Ap_, vb));
    CUDA_CHECK(cudaMalloc(&d_row_ptr_, (N_ + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_, nnz_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val_, nnz_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partials_, max_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_counter_, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_s_, sizeof(cgk::Scalars)));
    CUDA_CHECK(cudaMemcpy(d_b_, b.data(), vb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr_, a.row_ptr.data(), (N_ + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_, a.col.data(), nnz_ * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val_, a.val.data(), nnz_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_counter_, 0, sizeof(unsigned int)));
    // scalars() copies the whole struct; zero it so no field is ever read
    // uninitialised (found by compute-sanitizer --tool initcheck).
    CUDA_CHECK(cudaMemset(d_s_, 0, sizeof(cgk::Scalars)));
    // cusparseSpMV with beta = 0 still runs a y = beta * y kernel that READS
    // y (= Ap) before it is ever written (initcheck finding), so Ap must hold
    // finite values; 0 * NaN would otherwise poison the first iteration.
    CUDA_CHECK(cudaMemset(d_Ap_, 0, vb));
    reduce_ = {d_partials_, d_counter_};

    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    CUBLAS_CHECK(cublasSetPointerMode(cublas_, CUBLAS_POINTER_MODE_DEVICE));
    CUSPARSE_CHECK(cusparseCreate(&cusparse_));
    CUSPARSE_CHECK(cusparseSetStream(cusparse_, stream_));
    CUSPARSE_CHECK(cusparseCreateCsr(&matA_, N_, N_, nnz_, d_row_ptr_, d_col_, d_val_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecP_, N_, d_p_, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecAp_, N_, d_Ap_, CUDA_R_64F));
    const double one = 1.0, zero = 0.0;
    size_t buf = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                           matA_, vecP_, &zero, vecAp_, CUDA_R_64F,
                                           CUSPARSE_SPMV_ALG_DEFAULT, &buf));
    CUDA_CHECK(cudaMalloc(&d_spmv_buf_, std::max<size_t>(buf, 16)));
  }

  ~GpuCg() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    cusparseDestroyDnVec(vecP_);
    cusparseDestroyDnVec(vecAp_);
    cusparseDestroySpMat(matA_);
    cusparseDestroy(cusparse_);
    cublasDestroy(cublas_);
    for (void* p : {(void*)d_b_, (void*)d_x_, (void*)d_r_, (void*)d_p_, (void*)d_p2_,
                    (void*)d_Ap_, (void*)d_row_ptr_, (void*)d_col_, (void*)d_val_,
                    (void*)d_partials_, (void*)d_counter_, (void*)d_s_, d_spmv_buf_})
      cudaFree(p);
    cudaStreamDestroy(stream_);
  }

  // x = 0, r = b, p = b (p_old = 0 and beta = 0 for the fused variant), rr = b.b
  void reset(Variant v) {
    const size_t vb = N_ * sizeof(double);
    p_cur_ = d_p_;
    p_other_ = d_p2_;
    CUDA_CHECK(cudaMemsetAsync(d_x_, 0, vb, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_r_, d_b_, vb, cudaMemcpyDeviceToDevice, stream_));
    if (v == Variant::kFused || v == Variant::kFusedRows)
      CUDA_CHECK(cudaMemsetAsync(p_cur_, 0, vb, stream_));  // p_old for iteration 1
    else
      CUDA_CHECK(cudaMemcpyAsync(p_cur_, d_b_, vb, cudaMemcpyDeviceToDevice, stream_));
    cgk::dot_finalize<cgk::Finalize::kInitRR>
        <<<grid_, cgk::kBlock, 0, stream_>>>(N_, d_r_, d_r_, reduce_, d_s_);
    CUDA_CHECK_LAUNCH();
  }

  // Enqueues exactly one CG iteration on stream_ (no host synchronisation).
  void step(Variant v) {
    const int G = grid_, B = cgk::kBlock;
    cudaStream_t s = stream_;
    switch (v) {
      case Variant::kCsrScalar:
        cgk::spmv_csr_scalar<<<G, B, 0, s>>>(N_, d_row_ptr_, d_col_, d_val_, p_cur_, d_Ap_);
        break;
      case Variant::kCsrVector32:
        cgk::spmv_csr_vector<32><<<G, B, 0, s>>>(N_, d_row_ptr_, d_col_, d_val_, p_cur_, d_Ap_);
        break;
      case Variant::kCsrVector4:
        cgk::spmv_csr_vector<4><<<G, B, 0, s>>>(N_, d_row_ptr_, d_col_, d_val_, p_cur_, d_Ap_);
        break;
      case Variant::kStencil:
        cgk::spmv_stencil<<<G, B, 0, s>>>(n_, p_cur_, d_Ap_);
        break;
      case Variant::kFused:
        cgk::fused_update_p_spmv_dot<<<G, B, 0, s>>>(n_, d_r_, p_cur_, p_other_, d_Ap_, reduce_,
                                                     d_s_);
        std::swap(p_cur_, p_other_);  // p_cur_ now holds this iteration's p
        cgk::fused_update_xr_dot<<<G, B, 0, s>>>(N_, p_cur_, d_Ap_, d_x_, d_r_, reduce_, d_s_);
        CUDA_CHECK_LAUNCH();
        return;
      case Variant::kFusedRows:
        cgk::fused_update_p_spmv_dot_rows<<<rows_grid_, B, 0, s>>>(n_, strip_rows_, d_r_, p_cur_,
                                                                   p_other_, d_Ap_, reduce_, d_s_);
        std::swap(p_cur_, p_other_);
        cgk::fused_update_xr_dot<<<G, B, 0, s>>>(N_, p_cur_, d_Ap_, d_x_, d_r_, reduce_, d_s_);
        CUDA_CHECK_LAUNCH();
        return;
      case Variant::kCusparse:
        step_library();
        return;
    }
    // Unfused BLAS-1 tail shared by the four SpMV variants.
    cgk::dot_finalize<cgk::Finalize::kAlpha><<<G, B, 0, s>>>(N_, p_cur_, d_Ap_, reduce_, d_s_);
    cgk::axpy_alpha<<<G, B, 0, s>>>(N_, +1.0, d_s_, p_cur_, d_x_);
    cgk::axpy_alpha<<<G, B, 0, s>>>(N_, -1.0, d_s_, d_Ap_, d_r_);
    cgk::dot_finalize<cgk::Finalize::kBeta><<<G, B, 0, s>>>(N_, d_r_, d_r_, reduce_, d_s_);
    cgk::update_p<<<G, B, 0, s>>>(N_, d_s_, d_r_, p_cur_);
    CUDA_CHECK_LAUNCH();
  }

  // Runs `iters` iterations. With use_graph, `chunk` iterations are captured
  // once into a CUDA Graph and replayed (iters must be a multiple of chunk).
  void run(Variant v, int iters, bool use_graph, int chunk) {
    if (!use_graph) {
      for (int it = 0; it < iters; ++it) step(v);
      return;
    }
    if (!graph_exec_) build_graph(v, chunk);
    for (int it = 0; it < iters; it += chunk) CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
  }

  void drop_graph() {
    if (graph_exec_) CUDA_CHECK(cudaGraphExecDestroy(graph_exec_));
    graph_exec_ = nullptr;
  }

  cgk::Scalars scalars() {
    cgk::Scalars h{};
    CUDA_CHECK(cudaMemcpyAsync(&h, d_s_, sizeof h, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return h;
  }

  std::vector<double> solution() {
    std::vector<double> x(N_);
    CUDA_CHECK(cudaMemcpy(x.data(), d_x_, N_ * sizeof(double), cudaMemcpyDeviceToHost));
    return x;
  }

  cudaStream_t stream() const { return stream_; }
  long unknowns() const { return N_; }
  long nnz() const { return nnz_; }

 private:
  void build_graph(Variant v, int chunk) {
    // Kernel arguments are baked into the graph. The fused variant swaps its
    // two p buffers every iteration, so an even chunk returns to the start state.
    if ((v == Variant::kFused || v == Variant::kFusedRows) && chunk % 2 != 0) {
      std::fprintf(stderr, "graph chunk must be even for the fused variant\n");
      std::exit(2);
    }
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    for (int it = 0; it < chunk; ++it) step(v);
    CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
  }

  void step_library() {
    const double one = 1.0, zero = 0.0;
    CUSPARSE_CHECK(cusparseDnVecSetValues(vecP_, p_cur_));
    CUSPARSE_CHECK(cusparseSpMV(cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA_, vecP_,
                                &zero, vecAp_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                d_spmv_buf_));
    cgk::Scalars* s = d_s_;
    CUBLAS_CHECK(cublasDdot(cublas_, N_, p_cur_, 1, d_Ap_, 1, &s->pAp));
    cgk::lib_alpha<<<1, 1, 0, stream_>>>(s);
    CUBLAS_CHECK(cublasDaxpy(cublas_, N_, &s->alpha, p_cur_, 1, d_x_, 1));
    CUBLAS_CHECK(cublasDaxpy(cublas_, N_, &s->neg_alpha, d_Ap_, 1, d_r_, 1));
    CUBLAS_CHECK(cublasDdot(cublas_, N_, d_r_, 1, d_r_, 1, &s->rr_new));
    cgk::lib_beta<<<1, 1, 0, stream_>>>(s);
    CUBLAS_CHECK(cublasDscal(cublas_, N_, &s->beta, p_cur_, 1));  // p = beta p
    CUBLAS_CHECK(cublasDaxpy(cublas_, N_, &s->one, d_r_, 1, p_cur_, 1));  // p += r
    CUDA_CHECK_LAUNCH();
  }

  int n_;
  long N_, nnz_ = 0;
  int grid_ = 0;
  dim3 rows_grid_;
  int strip_rows_ = 1;
  cudaStream_t stream_{};
  double *d_b_{}, *d_x_{}, *d_r_{}, *d_p_{}, *d_p2_{}, *d_Ap_{}, *d_val_{}, *d_partials_{};
  double *p_cur_{}, *p_other_{};
  int *d_row_ptr_{}, *d_col_{};
  unsigned int* d_counter_{};
  cgk::Scalars* d_s_{};
  cgk::GridReduce reduce_{};
  cudaGraphExec_t graph_exec_{};
  cublasHandle_t cublas_{};
  cusparseHandle_t cusparse_{};
  cusparseSpMatDescr_t matA_{};
  cusparseDnVecDescr_t vecP_{}, vecAp_{};
  void* d_spmv_buf_{};
};

struct Options {
  int n;
  std::string mode;
  int iters;
  double tol;
  int reps;
  int chunk;
  bool ref;
  std::string csv;
};

// Returns true when the variant passes its correctness checks (tol mode).
bool run_variant(GpuCg& cg, const VariantSpec& spec, const Options& o, const poisson::Csr& a,
                 const std::vector<double>& b, const std::vector<double>& exact,
                 const std::vector<double>* ref) {
  const bool fixed = (o.mode == "fixed");
  common::CgRecord rec;
  rec.device = device_name();
  rec.impl = "cuda";
  rec.variant = spec.name;
  rec.mode = o.mode;
  rec.n = o.n;
  rec.unknowns = cg.unknowns();
  rec.nnz = cg.nnz();
  rec.threads = 0;
  rec.model_bytes_per_iter = model_bytes(spec.kind, cg.unknowns(), cg.nnz());
  rec.launches_per_iter = spec.graph ? 0 : launches_per_iter(spec.kind);

  EventTimer timer;
  if (fixed) {
    // Warm-up (also builds the graph), then best of `reps` timed runs.
    cg.reset(spec.kind);
    cg.run(spec.kind, o.chunk, spec.graph, o.chunk);
    float best_ms = 1e30f;
    for (int r = 0; r < o.reps; ++r) {
      cg.reset(spec.kind);
      timer.begin(cg.stream());
      cg.run(spec.kind, o.iters, spec.graph, o.chunk);
      best_ms = std::min(best_ms, timer.end(cg.stream()));
    }
    rec.iters = o.iters;
    rec.time_s = best_ms * 1e-3;
  } else {
    // Converge, checking rr every iteration (or every graph chunk).
    const int every = spec.graph ? o.chunk : 1;
    cg.reset(spec.kind);
    timer.begin(cg.stream());
    int it = 0;
    for (; it < o.iters; it += every) {
      const cgk::Scalars s = cg.scalars();
      if (std::sqrt(s.rr / s.bb) < o.tol) break;
      cg.run(spec.kind, every, spec.graph, o.chunk);
    }
    rec.time_s = timer.end(cg.stream()) * 1e-3;
    rec.iters = it;
  }
  cg.drop_graph();

  const std::vector<double> x = cg.solution();
  rec.rel_res_true = poisson::true_rel_residual(a, b, x);
  rec.max_err_analytic = poisson::max_abs_diff(x, exact);
  if (ref) rec.max_err_vs_ref = poisson::max_abs_diff(x, *ref);
  std::printf("%s\n", rec.csv().c_str());
  common::append_csv(o.csv, common::kCgCsvHeader, rec.csv());
  if (fixed) return true;

  const double bound = poisson::discretisation_error_bound(o.n);
  bool ok = rec.rel_res_true < 10 * o.tol && rec.max_err_analytic < bound;
  if (ref) ok = ok && rec.max_err_vs_ref < 1e-3 * bound;
  std::printf("CHECK %-14s %s  rel_res_true=%.2e  max_err=%.3e (bound %.3e)  vs_cpu=%.2e  iters=%d\n",
              spec.name, ok ? "PASS" : "FAIL", rec.rel_res_true, rec.max_err_analytic, bound,
              rec.max_err_vs_ref, rec.iters);
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  const common::Args args(argc, argv);
  Options o;
  o.n = static_cast<int>(args.num("--n", 256));
  o.mode = args.str("--mode", "tol");
  o.iters = static_cast<int>(args.num("--iters", 100000));
  o.tol = args.real("--tol", 1e-10);
  o.reps = static_cast<int>(args.num("--reps", 3));
  o.chunk = static_cast<int>(args.num("--graph-chunk", 10));
  o.ref = args.has("--ref");
  o.csv = args.str("--csv", "");
  const std::string which = args.str("--variant", "all");

  const poisson::Csr a = poisson::build_laplacian_csr(o.n);
  const std::vector<double> b = poisson::make_rhs(o.n);
  const std::vector<double> exact = poisson::analytic_solution(o.n);

  std::vector<double> ref;
  if (o.ref) {  // CPU reference solution, same tolerance
    cg_host::solve(a, b, ref, o.iters, o.tol, false);
  }

  GpuCg cg(o.n, a, b);
  std::printf("%s\n", common::kCgCsvHeader);
  bool all_ok = true, found = false;
  for (const VariantSpec& spec : kVariants) {
    if (which != "all" && which != spec.name) continue;
    found = true;
    all_ok &= run_variant(cg, spec, o, a, b, exact, o.ref ? &ref : nullptr);
  }
  if (!found) {
    std::fprintf(stderr, "unknown variant '%s'\n", which.c_str());
    return 2;
  }
  return all_ok ? 0 : 1;
}
