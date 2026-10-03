# cuda-solver-lab

Conjugate Gradient and Jacobi solvers for the 2D Poisson equation, written as a
GPU performance study: a serial C++ reference, OpenMP, eight hand-written CUDA
variants (CSR scalar / vector SpMV, matrix-free stencil, warp-shuffle
reductions, fused kernels, CUDA Graphs), cuSPARSE + cuBLAS as the library
reference, three Jacobi stencil kernels, and a Fortran OpenACC version that runs
on the GPU through gfortran's nvptx offloading. Every variant is checked
against the CPU reference and an analytic solution, every timing is turned into
bandwidth and placed on a measured roofline, and the key kernels are profiled
with Nsight Compute.

All numbers below are produced by `./run_all.sh` on one machine: NVIDIA L4
(Ada, sm_89, 24 GB GDDR6 with ECC on, 48 MB L2), driver 580, CUDA 12.9,
AMD EPYC 7R13 (16 cores / 32 threads), gcc/gfortran 11.5. They are copied into
this file by `scripts/make_report.py` from the raw CSV files in `results/raw/`
and the Nsight exports in `results/ncu/` and `results/nsys/`; none are typed
by hand.

## Problem

-Laplace(u) = f on the unit square, u = 0 on the boundary, 5-point stencil on an
n x n interior grid (h = 1/(n+1)), system scaled by h^2 so the matrix has 4 on
the diagonal and -1 for the four neighbours. The manufactured solution is
u = g(x) g(y) with g(t) = t (1 - t) e^t (see `include/poisson.hpp`; the
textbook sin(pi x) sin(pi y) is a discrete eigenvector and makes CG converge in
one iteration, see `docs/DEVLOG.md`). FP64 throughout.

## What is in the repo

| Path | What |
|---|---|
| `include/poisson.hpp` | grid, CSR assembly, right-hand side, analytic solution, a-priori error bound |
| `src/host/cg_host.hpp`, `cg_cpu.cpp` | CPU CG (CSR SpMV + dot + axpy); built as `cg_serial` and `cg_omp` from one source |
| `src/cuda/cg_kernels.cuh` | all CG kernels: SpMV variants, warp-shuffle block reduction, in-kernel grid reduction ("last block done"), fused kernels |
| `src/cuda/cg_cuda.cu` | GPU CG driver: variants, CUDA Graph capture, cuSPARSE/cuBLAS path, timing, checks |
| `src/cuda/jacobi.cu` | Jacobi: naive, shared-memory tile, register streaming, CPU OpenMP baseline |
| `src/cuda/bw_probe.cu` | measured ceilings: copy / triad / read / write bandwidth, FP64 and FP32 FMA throughput |
| `src/acc/cg_acc.F90` | matrix-free CG in Fortran + OpenACC (`cg_acc_gpu`) and the same file serial (`cg_acc_host`) |
| `tests/` | ctest helpers: second-order convergence check, "OpenACC really ran on the GPU" check |
| `scripts/` | `build.sh`, `sanitize.sh`, `bench.sh`, `profile.sh`, `make_report.py` |
| `results/` | raw CSV, Nsight Compute / Systems exports, sanitizer logs, figures, `RESULTS.md` |
| `docs/DEVLOG.md` | the bugs hit while building this and how each was found |

### CG variants

Every GPU variant runs the same unpreconditioned CG with all scalars (alpha,
beta, r.r) kept in device memory: dot products finish inside the kernel with
the "last block done" pattern (block partials, `__threadfence`, atomic ticket,
last block reduces and updates the scalars), so the iteration loop never waits
for the host.

| Variant | Iteration = | Model DRAM bytes / iteration |
|---|---|---|
| `csr_scalar` | CSR SpMV one thread per row + 5 BLAS-1 kernels | 12 nnz + 20 N + 96 N |
| `csr_vector32` | CSR SpMV one warp per row (shuffle reduce) + 5 BLAS-1 | same |
| `csr_vector4` | CSR SpMV 4 lanes per row + 5 BLAS-1 | same |
| `stencil` | matrix-free SpMV + 5 BLAS-1 | 16 N + 96 N |
| `fused` | (1) p = r + beta p, Ap = A p, p.Ap in one pass; (2) x += alpha p, r -= alpha Ap, r.r in one pass | 32 N + 48 N |
| `fused_rows` | as `fused`, but kernel (1) walks row strips (up to 32 rows, fewer on small grids) with register / shuffle neighbour reuse | 32 N + 48 N |
| `*_graph` | the same iteration captured into a CUDA Graph (10 iterations per graph launch) | same |
| `cusparse` | `cusparseSpMV` (CSR) + `cublasDdot/Daxpy/Dscal` in device pointer mode | 12 nnz + 20 N + 112 N |
| CPU serial / OpenMP | CSR, same 6 steps as `csr_scalar` | 12 nnz + 20 N + 96 N |
| Fortran OpenACC | matrix-free, three `parallel loop` regions with reductions | 16 N + 48 N + 24 N |

N = unknowns, nnz = nonzeros (about 5 N). The model counts each array touched by
a kernel once (perfect cache reuse of the SpMV source vector) and is the
denominator for "effective bandwidth". Flops per iteration are counted the same
way for every variant (2 nnz + 10 N), so GFLOP/s are comparable.

Fusing removes the separate `p = r + beta p` pass by recomputing p at the four
neighbours inside the SpMV (p is double-buffered because neighbours must see the
old value), and merges the two axpys with the r.r reduction: 6 kernels and
112 N bytes become 2 kernels and 80 N bytes.

## Correctness

* Every CUDA variant, both CPU builds and both Fortran builds solve to a
  relative residual of 1e-10 in ctest; the true residual ||b - Ax|| / ||b|| is
  recomputed on the host in long double, the max error against the analytic
  solution must be inside the a-priori discretisation bound
  (h^2/96) sup|u_xxxx + u_yyyy| (discrete maximum principle), and each GPU
  solution must match the CPU reference solution.
* `second_order_convergence`: halving h must cut the error by about 4x.
* `jacobi_gpu_matches_cpu`: all three Jacobi kernels against the CPU after 500
  sweeps on a 257^2 grid (a size that is not a multiple of any tile).
* `fortran_openacc_runs_on_gpu`: runs with `GOMP_DEBUG=1` and requires nvptx
  kernel launches in the log, so a silent host fallback fails the test.
* compute-sanitizer memcheck, racecheck, synccheck and initcheck for every CG
  variant and the Jacobi binary (`scripts/sanitize.sh`, logs in
  `results/sanitizer/`).

## Reproduce

```bash
./run_all.sh        # or: make bench
```

This builds (CMake, `-arch=sm_89`), runs ctest, the sanitizers, all
benchmarks, the Nsight Compute / Systems profiles and regenerates the results
section below plus `results/RESULTS.md` and `results/figures/*.png`. About 15
minutes on an idle L4. Requirements: CUDA 12.x (`CUDA_HOME`, default
`/usr/local/cuda-12.9`), gcc/g++/gfortran with OpenMP, `gcc-offload-nvptx` for
the OpenACC build (`-DBUILD_OPENACC=OFF` to skip), `uv` for the plotting venv.
Nsight Compute needs GPU performance counter access; on this host that means
passwordless `sudo` (see results). `bench.sh` waits until no other process
uses the GPU and logs what it saw in `results/raw/gpu_contention.log`.

Single runs:

```bash
build/cg_cuda --variant all --n 512 --mode tol --tol 1e-10 --ref       # correctness, all variants
build/cg_cuda --variant fused_rows --n 4096 --mode fixed --iters 100   # timing
build/jacobi --n 4096 --sweeps 100 --cpu
build/cg_acc_gpu 1024 tol 100000 1e-8
```

## Results

![CG time per iteration](results/figures/cg_ms_per_iter.png)
![CG effective bandwidth](results/figures/cg_bandwidth.png)
![Roofline](results/figures/roofline.png)
![Jacobi bandwidth](results/figures/jacobi_bandwidth.png)

<!-- RESULTS:BEGIN -->
_All numbers: measured on NVIDIA L4, CUDA 12.9, 2026-10-03. Generated by `scripts/make_report.py` from `results/raw/`._

### Hardware ceilings (bw_probe)

| probe | best of 20 | % of 300 GB/s spec |
|---|---|---|
| copy_f64 | 232.4 GB/s | 77.5 % |
| copy_f64x2 | 233.5 GB/s | 77.8 % |
| triad_f64 | 237.5 GB/s | 79.2 % |
| read_f64 | 261.0 GB/s | 87.0 % |
| write_f64 | 240.6 GB/s | 80.2 % |
| memcpy_d2d | 232.7 GB/s | 77.6 % |
| fma_f64 | 398 GFLOP/s |  |
| fma_f32 | 25,463 GFLOP/s |  |

Best measured bandwidth: **261.0 GB/s** (87.0 % of the 300 GB/s datasheet value; ECC is enabled on this GPU). FP64 FMA peak: **398 GFLOP/s** (FP32/FP64 = 64). FP64 ridge point = 398 / 261.0 = **1.53 flop/byte**.

### CG: time per iteration (ms), fixed iteration count

| implementation | 256^2 | 512^2 | 1024^2 | 2048^2 | 4096^2 |
|---|---|---|---|---|---|
| CPU serial (CSR) | 0.354 | 1.473 | 6.913 | 32.888 | 149.812 |
| CPU OpenMP 16 thr (CSR) | 0.050 | 0.140 | 0.936 | 9.206 | 48.354 |
| CPU OpenMP 32 thr (CSR) | 0.065 | 0.137 | 0.957 | 10.112 | 49.648 |
| Fortran OpenACC gfortran, GPU | 2.688 | 3.630 | 3.626 | 4.526 | 10.824 |
| CUDA CSR scalar | 0.026 | 0.046 | 0.524 | 2.860 | 12.719 |
| CUDA CSR vector (32 lanes/row) | 0.089 | 0.297 | 1.372 | 6.494 | 28.005 |
| CUDA CSR vector (4 lanes/row) | 0.031 | 0.060 | 0.530 | 2.822 | 12.265 |
| CUDA matrix-free stencil | 0.025 | 0.039 | 0.117 | 1.601 | 8.021 |
| CUDA stencil + CUDA Graph | 0.021 | 0.035 | 0.116 | 1.594 | 8.013 |
| CUDA fused (2 kernels/iter) | 0.018 | 0.034 | 0.130 | 1.439 | 6.137 |
| CUDA fused + CUDA Graph | 0.017 | 0.033 | 0.131 | 1.436 | 6.137 |
| CUDA fused, row-strip SpMV (2 kernels/iter) | 0.017 | 0.032 | 0.111 | 1.471 | 5.838 |
| CUDA fused row-strip + CUDA Graph | 0.016 | 0.030 | 0.113 | 1.468 | 5.851 |
| cuSPARSE SpMV + cuBLAS | 0.045 | 0.082 | 0.395 | 3.178 | 13.977 |

### CG at 4096^2 (16,777,216 unknowns): bandwidth, roofline, speedups

| implementation | ms/iter | model MB/iter | eff. GB/s | % of 300 GB/s | GFLOP/s | flop/byte | vs CPU serial | vs cuSPARSE |
|---|---|---|---|---|---|---|---|---|
| CPU serial (CSR) | 149.812 | 2,953 | 19.7 | - | 2.2 | 0.114 | 1.0x | 0.09x |
| CPU OpenMP 16 thr (CSR) | 48.354 | 2,953 | 61.1 | - | 6.9 | 0.114 | 3.1x | 0.29x |
| CPU OpenMP 32 thr (CSR) | 49.648 | 2,953 | 59.5 | - | 6.8 | 0.114 | 3.0x | 0.28x |
| Fortran OpenACC gfortran, GPU | 10.824 | 1,476 | 136.4 | 45.5 | 31.0 | 0.227 | 13.8x | 1.29x |
| CUDA CSR scalar | 12.719 | 2,953 | 232.1 | 77.4 | 26.4 | 0.114 | 11.8x | 1.10x |
| CUDA CSR vector (32 lanes/row) | 28.005 | 2,953 | 105.4 | 35.1 | 12.0 | 0.114 | 5.3x | 0.50x |
| CUDA CSR vector (4 lanes/row) | 12.265 | 2,953 | 240.7 | 80.2 | 27.4 | 0.114 | 12.2x | 1.14x |
| CUDA matrix-free stencil | 8.021 | 1,879 | 234.3 | 78.1 | 41.8 | 0.179 | 18.7x | 1.74x |
| CUDA stencil + CUDA Graph | 8.013 | 1,879 | 234.5 | 78.2 | 41.9 | 0.179 | 18.7x | 1.74x |
| CUDA fused (2 kernels/iter) | 6.137 | 1,342 | 218.7 | 72.9 | 54.7 | 0.250 | 24.4x | 2.28x |
| CUDA fused + CUDA Graph | 6.137 | 1,342 | 218.7 | 72.9 | 54.7 | 0.250 | 24.4x | 2.28x |
| CUDA fused, row-strip SpMV (2 kernels/iter) | 5.838 | 1,342 | 229.9 | 76.6 | 57.5 | 0.250 | 25.7x | 2.39x |
| CUDA fused row-strip + CUDA Graph | 5.851 | 1,342 | 229.4 | 76.5 | 57.3 | 0.250 | 25.6x | 2.39x |
| cuSPARSE SpMV + cuBLAS | 13.977 | 3,221 | 230.5 | 76.8 | 24.0 | 0.104 | 10.7x | 1.00x |

Roofline: the fused CG iteration does 0.25 flop/byte against a ridge point of 1.53 flop/byte, so the attainable rate is bandwidth x intensity = 65 GFLOP/s, 16.4 % of FP64 peak: CG is memory bound by a factor of 6.1, and the only lever is bytes moved per iteration.

Speedups at 4096^2 (best GPU variant = CUDA fused, row-strip SpMV (2 kernels/iter)): **25.7x** vs CPU serial, **8.3x** vs best CPU OpenMP, **2.39x** vs cuSPARSE + cuBLAS; CPU OpenMP vs serial: 3.1x.

Launch-bound regime (256^2, everything L2 resident): stencil 24.8 us/iter -> 20.7 us with a CUDA Graph (1.19x); fused 18.0 -> 16.7 us (1.07x). At 4096^2 the graph changes stencil by 1.001x (launch cost is hidden behind ms-long kernels).

### CG time to solution, 1024^2, relative residual 1e-8

| implementation | iterations | seconds | true rel. residual | max error vs analytic | vs CPU serial |
|---|---|---|---|---|---|
| CUDA CSR scalar | 3152 | 1.683 | 9.95e-09 | 8.857e-08 | 12.7x |
| CUDA CSR vector (32 lanes/row) | 3152 | 4.523 | 9.95e-09 | 8.857e-08 | 4.7x |
| CUDA CSR vector (4 lanes/row) | 3152 | 1.701 | 9.95e-09 | 8.857e-08 | 12.6x |
| CUDA matrix-free stencil | 3152 | 0.446 | 9.95e-09 | 8.857e-08 | 47.9x |
| CUDA stencil + CUDA Graph | 3160 | 0.426 | 9.28e-09 | 8.857e-08 | 50.2x |
| CUDA fused (2 kernels/iter) | 3152 | 0.471 | 9.95e-09 | 8.857e-08 | 45.4x |
| CUDA fused + CUDA Graph | 3160 | 0.462 | 9.28e-09 | 8.857e-08 | 46.3x |
| CUDA fused, row-strip SpMV (2 kernels/iter) | 3152 | 0.409 | 9.95e-09 | 8.857e-08 | 52.2x |
| CUDA fused row-strip + CUDA Graph | 3160 | 0.399 | 9.28e-09 | 8.857e-08 | 53.6x |
| cuSPARSE SpMV + cuBLAS | 3152 | 1.306 | 9.95e-09 | 8.857e-08 | 16.4x |
| CPU serial (CSR) | 3152 | 21.372 | 9.95e-09 | 8.857e-08 | 1.0x |
| CPU OpenMP 16 thr (CSR) | 3152 | 2.548 | 9.95e-09 | 8.857e-08 | 8.4x |
| Fortran OpenACC gfortran, GPU | 3152 | 11.316 | 9.95e-09 | 8.857e-08 | 1.9x |

GPU solves check the residual every iteration (graph variants every 10), which adds a device-to-host copy per check; it is included in these times.

### Jacobi sweep: effective bandwidth (GB/s, 24 B per point update)

| kernel | 1024^2 | 2048^2 | 4096^2 | 8192^2 |
|---|---|---|---|---|
| CPU OpenMP 16 threads | 271.9 (0.093 ms) | 104.9 (0.960 ms) | 56.4 (7.142 ms) | 54.8 (29.396 ms) |
| CUDA naive (global loads) | 876.1 (0.029 ms) | 243.0 (0.414 ms) | 241.6 (1.667 ms) | 241.0 (6.682 ms) |
| CUDA shared-memory tile 32x8 | 823.9 (0.031 ms) | 245.3 (0.410 ms) | 243.1 (1.657 ms) | 243.2 (6.622 ms) |
| CUDA register streaming (16 rows/thread) | 851.3 (0.030 ms) | 225.2 (0.447 ms) | 227.6 (1.769 ms) | 227.6 (7.077 ms) |

### Nsight Compute, one launch per kernel at 4096^2 (`ncu --set full`)

| profile | duration ms | DRAM % of peak | DRAM GB/s | DRAM read MB | DRAM write MB | achieved occ. % | theor. occ. % | sectors/request (global ld) | L2 hit % | regs/thread |
|---|---|---|---|---|---|---|---|---|---|---|
| spmv_csr_scalar | 5.923 | 95.0 | 284.6 | 1,529 | 157 | 98.8 | 100 | 18.41 | 47.2 | 40 |
| spmv_csr_vector32 | 45.625 | 19.4 | 58.2 | 2,445 | 211 | 80.8 | 100 | 1.80 | 41.3 | 39 |
| spmv_csr_vector4 | 5.359 | 96.1 | 288.1 | 1,392 | 152 | 99.8 | 100 | 5.09 | 28.9 | 40 |
| cusparse_spmv | 5.443 | 95.7 | 286.9 | 1,423 | 138 | 72.8 | 75 | 5.92 | 28.0 | 56 |
| spmv_stencil | 1.360 | 82.7 | 247.9 | 199 | 138 | 84.9 | 100 | 8.40 | 69.1 | 34 |
| dot_finalize | 1.072 | 96.5 | 289.3 | 308 | 2 | 96.4 | 100 | 8.00 | 0.1 | 36 |
| fused_p_spmv_dot | 3.279 | 85.5 | 256.3 | 539 | 301 | 91.2 | 100 | 8.39 | 58.4 | 40 |
| fused_rows_p_spmv_dot | 2.459 | 85.9 | 257.5 | 336 | 297 | 98.5 | 100 | 3.43 | 52.3 | 36 |
| fused_xr_dot | 3.426 | 87.8 | 263.0 | 614 | 287 | 98.4 | 100 | 7.99 | 33.4 | 38 |

First fused kernel, model traffic 537 MB (read r, p_old; write p_new, Ap): the 1D grid-stride version moves 840 MB (reads 2.01x the model), the row-strip version 633 MB (reads 1.25x); kernel time 3.28 -> 2.46 ms under ncu.


ncu locks clocks to base during profiling, so durations differ slightly from the timed runs. Sectors/request is averaged over all global loads of the kernel: a fully coalesced warp load touches 8 sectors (32 B each) for 8-byte values and 4 for 4-byte values; higher means scattered accesses.

Unprivileged `ncu` on this host fails with:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

(RmProfilingAdminOnly=1), so the profiles were taken with passwordless `sudo`.

### Nsight Systems kernel census, cuSPARSE + cuBLAS path (n = 1024, 10 warm-up + 100 timed iterations)

| kernel | instances | per iteration |
|---|---|---|
| cusparse::csrmv_v3_kernel | 110 | 1 |
| axpy_kernel_ref | 330 | 3 |
| dot_kernel | 220 | 2 |
| scal_kernel_ref | 110 | 1 |
| cusparse::vector_scalar_multiply_kernel | 110 | 1 |
| reduce_1Block_kernel | 220 | 2 |
| cusparse::<unnamed>::csr_partition_kernel | 110 | 1 |
| cgk::lib_beta(cgk::Scalars *) | 110 | 1 |
| cgk::lib_alpha(cgk::Scalars *) | 110 | 1 |

**13 GPU kernels per CG iteration** for the library path vs 2 for the fused variant.

### compute-sanitizer

43 of 44 runs clean (memcheck, racecheck, synccheck, initcheck x every CG variant + Jacobi).
 Not clean: `cg_cuda cusparse racecheck`: RACECHECK SUMMARY: 100 hazards displayed (374 errors, 187 warnings).


<!-- RESULTS:END -->

## Reading the numbers

* **CG is memory bound everywhere.** Its arithmetic intensity sits far left of
  the FP64 ridge point even on the L4, whose FP64 rate is only 1/64 of FP32, so
  the work is about bytes per iteration, not flops. The variants are ranked by
  how few bytes they move and how close they get to the bandwidth they ask for.
* **CSR layout matters more than the SpMV kernel style.** One thread per row
  makes neighbouring threads read addresses about five elements apart
  (high sectors per request in ncu), yet the kernel still saturates DRAM because
  the stride is small and L1/L2 absorb it. One warp per row is the classic fix
  for long rows, but with about five nonzeros per row it leaves most lanes idle
  and is the slowest GPU variant; four lanes per row matches the row length.
  cuSPARSE's CSR kernel is in the same band as the hand-written CSR kernels.
* **The L2 cliff.** From 1024^2 to 2048^2 the matrix-free and fused variants
  slow down by far more than the 4x growth in work: below it their vectors live
  in the 48 MB L2, above it everything streams from DRAM. The CSR variants hit
  the same cliff one size earlier (512^2 to 1024^2) because the matrix itself
  stops fitting.
* **Dropping the matrix is the big step.** The matrix-free stencil removes
  row_ptr, col and val traffic, which is most of the CSR bytes.
* **Fusion pays exactly where the byte count says it should**, and Nsight
  Compute was needed to get the rest: the first fused kernel read about twice
  its model bytes from DRAM; walking row strips with neighbours in registers and
  warp shuffles brought it close to the model (see the ncu table and
  `docs/DEVLOG.md`).
* **CUDA Graphs only matter when kernels are tiny.** At 256^2 the whole working
  set is L2 resident and launch overhead is a visible fraction of an iteration;
  at 4096^2 the effect is within noise.
* **Effective bandwidth above the copy probe is not an error.** At 1024^2 and
  2048^2 the vectors (8 MB and 32 MB) fit in the 48 MB L2, so data written by one
  kernel is read by the next from L2; the model counts it as DRAM traffic. The
  4096^2 numbers (128 MB per vector) are the DRAM-bound ones. Also, read-heavy
  kernels reach higher DRAM throughput in ncu than the copy probe does in timed
  runs; copy has a 1:1 read/write mix.
* **Jacobi: shared-memory tiling does not beat the naive kernel on Ada.** The
  naive kernel's neighbour loads are already served by L1/L2, so both run at the
  same DRAM-bound rate; the register-streaming variant is slightly slower
  (not profiled; a likely cause is less memory-level parallelism with 16 rows
  per thread). The classic shared-memory speedup for this stencil was not
  reproduced on this GPU.

## What did not work / limits

* **OpenACC with gfortran 11 is slow.** It runs on the GPU (verified by the test
  above), and its time per iteration is dominated by libgomp's per-region launch
  and reduction overhead plus gang/vector mapping (vector length 32), not by
  memory traffic: at small sizes it is slower than the serial CPU. NVIDIA's
  `nvfortran` (HPC SDK) was not installed or tried; it would be the fair
  OpenACC comparison. Making gfortran 11 offload work at all needed
  `-foffload=nvptx-none=-Wa,--no-verify` (its PTX targets sm_35, which CUDA 12's
  ptxas refuses; the driver JIT-compiles the PTX for sm_89).
* **No preconditioner.** Unpreconditioned CG needs O(n) iterations; a real
  solver would use multigrid or at least Jacobi/IC preconditioning. This study
  is about the per-iteration kernel performance.
* **2D only, single GPU, FP64 only.** No 3D 7-point case, no multi-GPU halo
  exchange, no mixed precision.
* **cuSPARSE path is the plain API usage**: `CUSPARSE_SPMV_ALG_DEFAULT` without
  `cusparseSpMV_preprocess`, so a CSR partition kernel and a y = beta y kernel run
  every iteration (visible in the nsys census). The comparison is against the
  library as commonly called, not its best possible configuration.
* **compute-sanitizer racecheck reports hazards inside cuSPARSE's
  `csrmv_v3_kernel`** (library code, not investigated further; our own kernels
  are clean under every tool).
* **Nsight Compute needs elevated rights on this host** (`ERR_NVGPUCTRPERM`
  without sudo). ncu also locks clocks to base, so its kernel durations are not
  identical to the timed runs.
* **"% of 300 GB/s" uses the datasheet number.** ECC is enabled on this L4; the
  best timed probe reaches less than that, and DRAM byte counts in ncu are
  above the model even for pure streaming kernels. Both are reported as
  measured; the cause of the gap was not isolated.
* **Shared host.** Another workload used the same GPU at times during
  development; the benchmark script waits for an idle GPU and logs it, but one
  run per configuration (best of 3 to 5 repetitions) is not a statistical study.
* **CPU numbers are one socket of a cloud VM** (EPYC 7R13, 16 cores); the
  OpenMP code is the plain loop version without NUMA tuning or vectorisation
  work beyond `-O3 -march=native`.

## License

MIT, see `LICENSE`.
