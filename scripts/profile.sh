#!/usr/bin/env bash
# Nsight Compute profiles of the key kernels at n = 4096 (DRAM-bound regime)
# and an Nsight Systems kernel census of the cuSPARSE/cuBLAS path.
#
# This host sets RmProfilingAdminOnly=1, so ncu as a normal user fails with
# ERR_NVGPUCTRPERM (the exact error is saved to results/ncu/permission_error.txt).
# ncu is therefore run through passwordless sudo when available.
set -uo pipefail
source "$(dirname "$0")/env.sh"
out="$ROOT/results/ncu"
rm -rf "$out"; mkdir -p "$out" "$ROOT/results/nsys"
NCU="$CUDA_HOME/bin/ncu"
N=4096
common=(--n $N --mode fixed --iters 10 --reps 1 --graph-chunk 10)

# Record the unprivileged attempt verbatim.
"$NCU" --section SpeedOfLight -k regex:spmv_stencil -c 1 "$BUILD/cg_cuda" --variant stencil "${common[@]}" \
  2>&1 | grep -E "==ERROR==|==WARNING==" > "$out/permission_error.txt" || true

if ! sudo -n true 2>/dev/null; then
  echo "no passwordless sudo: ncu profiles skipped (see $out/permission_error.txt)"
  exit 0
fi

profile() {  # name variant kernel-regex
  local name=$1 variant=$2 regex=$3
  wait_for_idle_gpu "ncu_$name"
  sudo -n "$NCU" --set full -k "regex:$regex" --launch-skip 3 -c 1 -f -o "$out/$name" \
    "$BUILD/cg_cuda" --variant "$variant" "${common[@]}" > "$out/$name.log" 2>&1
  sudo -n chown "$(id -u):$(id -g)" "$out/$name.ncu-rep"
  "$NCU" -i "$out/$name.ncu-rep" --csv --page raw > "$out/$name.raw.csv"
  "$NCU" -i "$out/$name.ncu-rep" --page details > "$out/$name.details.txt"
}
profile spmv_csr_scalar   csr_scalar   '^spmv_csr_scalar'
profile spmv_csr_vector32 csr_vector32 '^spmv_csr_vector'
profile spmv_csr_vector4  csr_vector4  '^spmv_csr_vector'
profile spmv_stencil      stencil      '^spmv_stencil'
profile dot_finalize      stencil      '^dot_finalize'
profile fused_p_spmv_dot  fused        '^fused_update_p_spmv_dot$'
profile fused_rows_p_spmv_dot fused_rows '^fused_update_p_spmv_dot_rows'
profile fused_xr_dot      fused        '^fused_update_xr_dot'
profile cusparse_spmv     cusparse     'csrmv'

# Kernel census of the library path: how many GPU kernels per CG iteration.
wait_for_idle_gpu nsys_cusparse
nsys profile --trace=cuda --force-overwrite=true -o "$ROOT/results/nsys/cusparse_n1024" \
  "$BUILD/cg_cuda" --variant cusparse --n 1024 --mode fixed --iters 100 --reps 1 --graph-chunk 10 \
  > /dev/null 2>&1
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
  -o "$ROOT/results/nsys/cusparse_n1024" "$ROOT/results/nsys/cusparse_n1024.nsys-rep" > /dev/null 2>&1
ls "$out" "$ROOT/results/nsys"
