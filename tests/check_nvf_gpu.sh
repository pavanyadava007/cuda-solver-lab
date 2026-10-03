#!/usr/bin/env bash
# nvfortran counterpart of check_acc_gpu.sh: runs an nvfortran build (OpenACC
# or do concurrent with -stdpar=gpu) with NV_ACC_NOTIFY=1, which makes the
# NVIDIA runtime print one line per CUDA kernel launch, and requires that it
# (1) converges and (2) really launched kernels on the GPU.
set -euo pipefail
bin="$1"
log=$(mktemp)
trap 'rm -f "$log"' EXIT
NV_ACC_NOTIFY=1 "$bin" 255 tol 100000 1e-10 >"$log" 2>&1
grep -m1 "CHECK" "$log"
launches=$(grep -c "launch CUDA kernel" "$log" || true)
echo "CUDA kernel launches: $launches"
grep -q "CHECK PASS" "$log" && [ "$launches" -gt 100 ]
