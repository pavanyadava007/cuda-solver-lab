#!/usr/bin/env bash
# Nsight Systems kernel census of the Fortran GPU builds (gfortran OpenACC and,
# if built, nvfortran OpenACC / do concurrent): which kernels run per CG
# iteration, how long each takes, and the CUDA API calls around them.
# n = 1024, 100 fixed iterations, no warm-up (same as the timed runs).
set -uo pipefail
source "$(dirname "$0")/env.sh"
out="$ROOT/results/nsys"
mkdir -p "$out"
for b in cg_acc_gpu cg_acc_nvf cg_dc_nvf; do
  [ -x "$BUILD/$b" ] || continue
  wait_for_idle_gpu "nsys_$b"
  nsys profile --trace=cuda --force-overwrite=true -o "$out/${b}_n1024" \
    "$BUILD/$b" 1024 fixed 100 0 > /dev/null 2>&1
  nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv --force-export=true \
    -o "$out/${b}_n1024" "$out/${b}_n1024.nsys-rep" > /dev/null 2>&1
done
ls "$out"
