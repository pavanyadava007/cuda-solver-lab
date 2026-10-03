#!/usr/bin/env bash
# All timing runs. Writes results/raw/{peaks,cg,jacobi}.csv and env.txt.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$RAW"
rm -f "$RAW"/peaks.csv "$RAW"/cg.csv "$RAW"/jacobi.csv "$RAW"/gpu_contention.log

{
  echo "date: $(date -Iseconds)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.memory,power.limit --format=csv
  nvcc --version | tail -2
  g++ --version | head -1
  gfortran --version | head -1
  if [ -x "$BUILD/cg_acc_nvf" ]; then
    nvf=$(grep -m1 '^NVFORTRAN:' "$BUILD/CMakeCache.txt" | cut -d= -f2)
    echo "nvfortran: $nvf"
    "$nvf" --version | grep -m1 nvfortran
  fi
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core'
} > "$RAW/env.txt"

# 1. Practical ceilings (bandwidth + FP64/FP32 FMA throughput).
wait_for_idle_gpu peaks
"$BUILD/bw_probe" --reps 20 --csv "$RAW/peaks.csv"

# 2. CG fixed-iteration benchmarks. Iteration counts keep each run ~0.1-1.5 s.
declare -A GPU_IT=([256]=2000 [512]=1000 [1024]=500 [2048]=200 [4096]=100)
declare -A CPU_IT=([256]=200 [512]=100 [1024]=50 [2048]=20 [4096]=10)
for n in 256 512 1024 2048 4096; do
  wait_for_idle_gpu "cg_cuda_n$n"
  "$BUILD/cg_cuda" --variant all --n "$n" --mode fixed --iters "${GPU_IT[$n]}" --reps 5 --csv "$RAW/cg.csv"
  "$BUILD/cg_serial" --n "$n" --mode fixed --iters "${CPU_IT[$n]}" --reps 3 --csv "$RAW/cg.csv"
  for t in 16 32; do  # 16 = one thread per physical core, 32 = with SMT
    OMP_NUM_THREADS=$t OMP_PROC_BIND=spread OMP_PLACES=threads \
      "$BUILD/cg_omp" --n "$n" --mode fixed --iters $((2 * CPU_IT[$n])) --reps 3 --csv "$RAW/cg.csv"
  done
  # Fortran GPU builds use the CUDA iteration counts: the timed region includes
  # the host->device copies of the data region (managed-memory migration for
  # do concurrent), which short runs would not amortise (docs/DEVLOG.md).
  wait_for_idle_gpu "cg_acc_n$n"
  "$BUILD/cg_acc_gpu" "$n" fixed "${GPU_IT[$n]}" 0 "$RAW/cg.csv"
  for b in cg_acc_nvf cg_dc_nvf; do  # optional nvfortran builds
    [ -x "$BUILD/$b" ] && "$BUILD/$b" "$n" fixed "${GPU_IT[$n]}" 0 "$RAW/cg.csv"
  done
done
"$BUILD/cg_acc_host" 1024 fixed 50 0 "$RAW/cg.csv"

# 3. Time to solution (rel. residual 1e-8) at n = 1024 for every implementation.
wait_for_idle_gpu cg_tol
"$BUILD/cg_cuda" --variant all --n 1024 --mode tol --tol 1e-8 --csv "$RAW/cg.csv"
"$BUILD/cg_serial" --n 1024 --mode tol --tol 1e-8 --csv "$RAW/cg.csv"
OMP_NUM_THREADS=16 OMP_PROC_BIND=spread OMP_PLACES=threads \
  "$BUILD/cg_omp" --n 1024 --mode tol --tol 1e-8 --csv "$RAW/cg.csv"
"$BUILD/cg_acc_gpu" 1024 tol 100000 1e-8 "$RAW/cg.csv"
for b in cg_acc_nvf cg_dc_nvf; do
  [ -x "$BUILD/$b" ] && "$BUILD/$b" 1024 tol 100000 1e-8 "$RAW/cg.csv"
done
# nvfortran compiler feedback (-Minfo) as an artefact.
mkdir -p "$ROOT/results/minfo"
for b in cg_acc_nvf cg_dc_nvf; do
  [ -f "$BUILD/$b.minfo.txt" ] && cp "$BUILD/$b.minfo.txt" "$ROOT/results/minfo/"
done

# 4. Jacobi kernels (+ CPU OpenMP baseline with 16 threads).
for n in 1024 2048 4096 8192; do
  wait_for_idle_gpu "jacobi_n$n"
  OMP_NUM_THREADS=16 OMP_PROC_BIND=spread OMP_PLACES=threads \
    "$BUILD/jacobi" --n "$n" --sweeps 100 --reps 3 --cpu --csv "$RAW/jacobi.csv"
done
echo "bench done"
