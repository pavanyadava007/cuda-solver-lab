#!/usr/bin/env bash
# compute-sanitizer memcheck / racecheck / synccheck / initcheck on small runs.
# The CG binary is checked per variant so library (cuSPARSE) findings do not
# hide findings in our own kernels.
set -uo pipefail
source "$(dirname "$0")/env.sh"
out="$ROOT/results/sanitizer"
rm -rf "$out"; mkdir -p "$out"
summary="$out/summary.csv"
echo "binary,variant,tool,exit_code,summary" > "$summary"
for tool in memcheck racecheck synccheck initcheck; do
  for v in csr_scalar csr_vector32 csr_vector4 stencil stencil_graph fused fused_graph fused_rows fused_rows_graph cusparse; do
    log="$out/cg_cuda_${v}_${tool}.txt"
    compute-sanitizer --tool "$tool" --error-exitcode 9 "$BUILD/cg_cuda" --variant "$v" --n 63 \
      --mode tol --tol 1e-8 >"$log" 2>&1
    rc=$?
    echo "cg_cuda,$v,$tool,$rc,\"$(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' "$log" | tail -1 | sed 's/=* //')\"" >> "$summary"
  done
  log="$out/jacobi_${tool}.txt"
  compute-sanitizer --tool "$tool" --error-exitcode 9 "$BUILD/jacobi" --n 67 --check --sweeps 20 \
    --reps 1 >"$log" 2>&1
  rc=$?
  echo "jacobi,all,$tool,$rc,\"$(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' "$log" | tail -1 | sed 's/=* //')\"" >> "$summary"
done
cat "$summary"
