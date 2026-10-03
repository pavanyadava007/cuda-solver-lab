# Development log

The real problems hit while building this, in the order they happened, and how
each one was found. Numbers quoted from final runs point at the file they come
from; numbers marked "dev run" were seen in terminal output during development
and are not stored in `results/`.

## 1. gfortran OpenACC offload would not link: `ptxas` rejects sm_35

**Symptom.** After installing `gcc-offload-nvptx` (it was missing; gfortran 11
was configured with `--enable-offload-targets=nvptx-none` but the target
compiler was not installed), `gfortran -fopenacc -foffload=nvptx-none` failed
at link time:

```
ptxas fatal   : Value 'sm_35' is not defined for option 'gpu-name'
nvptx-as: ptxas returned 255 exit status
mkoffload: fatal error: x86_64-amazon-linux-accel-nvptx-none-gcc returned 1 exit status
```

**Cause.** The GCC 11 nvptx backend emits PTX with `.target sm_35`. Its
assembler (`nvptx-as`) only calls `ptxas` to *verify* the PTX, and CUDA 12's
ptxas dropped Kepler. The PTX itself is still valid input for the driver JIT.

**Fix.** Pass `-Wa,--no-verify` to the offload assembler:
`-foffload=nvptx-none=-Wa,--no-verify` (GCC 11 syntax; the newer
`-foffload-options=` spelling is rejected). The driver JIT-compiles the PTX for
sm_89 at first launch.

**How I made sure it really ran on the GPU.** `GOMP_DEBUG=1` prints every
`nvptx_exec: kernel ...: launch gangs=..., vectors=32` line. The ctest
`fortran_openacc_runs_on_gpu` requires those lines, so a silent host fallback
(which OpenACC allows) fails the test instead of producing a plausible but wrong
"GPU" number.

## 2. CG converged in one iteration: the test problem was an eigenvector

**Symptom.** First Fortran run on 256^2: `CHECK PASS ... iters=1` (dev run).
A converged answer after one iteration of CG on a 65k-unknown Poisson problem is
not a fast solver, it is a broken experiment.

**Cause.** I had used the textbook manufactured solution
u = sin(pi x) sin(pi y). Its grid samples are an exact eigenvector of the
discrete 5-point Laplacian, so b = A u is parallel to u and the very first CG
step is exact. Every timing based on it would have measured one useful
iteration followed by divisions of rounding noise.

**Fix.** u = g(x) g(y) with g(t) = t (1 - t) e^t, which is smooth, zero on the
boundary and not an eigenvector. CG now needs the expected O(n) iterations
(874 at 255^2 in the ctest run, 3152 at 1024^2 to 1e-8 in `results/raw/cg.csv`). I also
re-derived the error bound the tests use: truncation error
(h^2/12)(u_xxxx + u_yyyy) times the discrete maximum principle bound
||A_h^-1||_inf <= 1/8. My first constant (for the sin solution) had been off by
a factor of four; deriving it properly instead of guessing is what caught that.
A separate test checks the observed order: halving h cuts the error by a factor
of 4.0 (dev run: 5.68e-6 at 127^2, 1.42e-6 at 255^2).

## 3. compute-sanitizer initcheck: copying a half-initialised struct

**Symptom.** `compute-sanitizer --tool initcheck` reported hundreds of
"Host API memory access error ... Uninitialized access ... by cudaMemcpy
source" from `GpuCg::scalars()`.

**Cause.** The CG scalars (rr, bb, pAp, alpha, ...) live in a device struct
that kernels fill field by field. The host convergence check copies the whole
64-byte struct, including fields no kernel had written yet.

**Fix.** `cudaMemset` the struct once at construction. Harmless in practice,
but the kind of thing that turns into a NaN the day someone reads the wrong
field.

## 4. initcheck again: cuSPARSE reads the output vector even with beta = 0

**Symptom.** After fix 3, initcheck still flagged the `cusparse` variant only:
"Uninitialized __global__ memory read of size 8 bytes at
`cusparse::vector_scalar_multiply_kernel`", called from `cusparseSpMV`.

**Cause.** With `CUSPARSE_SPMV_ALG_DEFAULT` the library runs a separate
`y = beta * y` kernel before the CSR kernel, even for beta = 0, and that kernel
reads y. y was our freshly allocated `Ap` buffer. If that memory had contained a
NaN bit pattern, 0 * NaN = NaN would have poisoned the first CG iteration.

**Fix.** Zero `Ap` at construction. The Nsight Systems census
(`results/nsys/cusparse_n1024_cuda_gpu_kern_sum.csv`) confirms the extra kernel:
the library path launches 13 GPU kernels per CG iteration, including this
scaling kernel and a CSR partition kernel, against 2 for the fused variant.

## 5. racecheck: 374 errors, none of them mine

**Symptom.** racecheck on `cg_cuda --variant all` printed 100 hazards (374
errors, 187 warnings) and exited non-zero.

**Finding.** Every displayed hazard was inside `cusparse::csrmv_v3_kernel`, and
the display limit hid whether any of our kernels were affected. I changed
`scripts/sanitize.sh` to run every tool once per variant. Result: all our
variants are clean under memcheck, racecheck, synccheck and initcheck; only the
cuSPARSE variant reports racecheck hazards, all in library code
(`results/sanitizer/summary.csv`). I did not investigate further and report it
as a library-side finding, not as a confirmed bug.

## 6. Warp-per-row SpMV: full-mask shuffle in a divergent tail (caught in review)

The first version of `spmv_csr_vector<TPR>` looped `for (row = my_row; row <
nrows; row += stride)` with `__shfl_down_sync(0xffffffff, ...)` inside. For
TPR < 32 several rows share a warp; in the last pass some sub-groups of a warp
have a row and others do not, so part of the warp leaves the loop while the rest
executes a full-mask shuffle: undefined behaviour. I caught this reading the
code before the first run, not with a tool. Fix: the loop counter is the first
row of the *warp* (uniform across all 32 lanes), each sub-group checks
`row < nrows` only around the loads and the store, and every lane reaches every
shuffle.

## 7. A 1e-12 tolerance failed the "true residual" check

**Symptom.** The convergence-order test first used tol = 1e-12. The fused
variant at 255^2 then failed its own check: the recursively updated residual
said converged, the true residual ||b - Ax|| / ||b|| recomputed on the host was
about 1.6e-11 (dev run), above the 10 x tol acceptance.

**Cause.** Not a kernel bug: the residual gap of CG in finite precision. The
recurrence r <- r - alpha A p drifts away from b - A x by accumulated rounding,
and the gap grows with problem size (at 1000^2 every variant, CPU included,
ends at the same true residual about 4.6x the tolerance; dev run). All variants
agree with the CPU reference to about 1e-15, which is what shows the kernels are
right.

**Fix.** The tests use tol = 1e-10, where the gap is irrelevant. A production
solver would recompute the true residual periodically (residual replacement).

## 8. ncu: ERR_NVGPUCTRPERM

`ncu` as a normal user fails on this host
(`results/ncu/permission_error.txt`): the driver runs with
`RmProfilingAdminOnly=1`. Passwordless `sudo` is available here, so
`scripts/profile.sh` records the unprivileged error and then profiles through
`sudo -n`, giving the report files back to the user. If sudo is not available
the script skips profiling instead of inventing metrics.

## 9. Nsight Compute showed the fused kernel reading twice its model bytes

This is the optimisation that came out of profiling rather than intuition.

**Observation.** At 4096^2 the first fused kernel (p update + stencil SpMV +
p.Ap) should move 32 bytes per unknown: read r and p_old, write p_new and Ap.
ncu reported DRAM reads of about 2.0x that model, while the purely streaming
kernels (`dot_finalize`, `fused_xr_dot`) read only 1.1x to 1.15x their model
(`results/ncu/*.raw.csv`, summarised in `results/RESULTS.md`). So the vertical
neighbours (index k +- n) were mostly *not* coming from L2: with a 1D
grid-stride loop, the rows above and below are touched by different blocks at
different times.

**Change.** `fused_update_p_spmv_dot_rows`: one thread per column walks a strip
of up to 32 rows and keeps p_new for rows j-1, j, j+1 in registers; the left and
right neighbours come from the adjacent lanes with `__shfl_up_sync` /
`__shfl_down_sync` (lanes 0 and 31 load their missing neighbour). Every r and
p_old value is now read from DRAM about once per strip.

**Result.** DRAM reads went from 2.01x to 1.25x of the model and the kernel from
3.28 to 2.46 ms under ncu; the whole CG iteration from 6.137 to 5.838 ms at
4096^2 (`results/raw/cg.csv`).

**Follow-up bug.** The first version used fixed 32-row strips. At 256^2 that
is a grid of 1 x 8 blocks on a 58-SM GPU, and the "optimised" variant was
almost 2x *slower* than the plain fused one at that size (dev run: 0.032 vs
0.018 ms per iteration). The strip height is now chosen per problem size so the
grid has at least about 8 blocks per SM (1-row strips at 256^2).

## 10. Effective bandwidth above the measured peak

Two separate effects made some "GB/s" numbers look impossible:

* At 1024^2 and 2048^2 the stencil variants report more than the best timed
  bandwidth probe. The vectors are 8 MB and 32 MB and the L4 has 48 MB of L2,
  so `Ap` written by one kernel is read by the next one from L2. The traffic
  model counts it as DRAM traffic. Only the 4096^2 numbers are DRAM bound, so
  the analysis uses those.
* ncu reports read-heavy kernels (CSR SpMV, dot) at about 95 % of peak DRAM
  throughput (about 285 GB/s), more than any timed probe (best: the read probe).
  Copy-type probes mix reads and writes 1:1, which costs bus turnarounds; the
  SpMV is mostly reads. ECC is enabled on this L4, and ncu's DRAM byte counts
  are 10 to 15 % above the model even for pure streaming kernels; I did not
  isolate how much of that is ECC. The README reports "% of the 300 GB/s
  datasheet value" and the measured probe values side by side rather than
  picking the flattering denominator.

## 11. Shared GPU

When I started, another workload (a PyTorch evaluation from a different
project) was using the same L4 at about 40 % utilisation and the CPU load
average was above 20. Benchmarks taken then would have been wrong in a way
nobody could detect later. `scripts/env.sh` now has `wait_for_idle_gpu`, which
blocks until no other compute process is on the GPU and logs what it saw to
`results/raw/gpu_contention.log` before every benchmark block. For the final
run every entry shows no other GPU process.
