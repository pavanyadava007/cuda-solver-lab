#!/usr/bin/env bash
# Runs the Fortran OpenACC solver with libgomp debug output and checks that
# (1) it converges and (2) its kernels were really launched on the nvptx
# (NVIDIA GPU) device rather than silently falling back to the host.
set -euo pipefail
bin="$1"
log=$(mktemp)
trap 'rm -f "$log"' EXIT
GOMP_DEBUG=1 "$bin" 255 tol 100000 1e-10 >"$log" 2>&1
grep -m1 "CHECK" "$log"
launches=$(grep -c "nvptx_exec: kernel .*: launch" "$log" || true)
echo "nvptx kernel launches: $launches"
grep -q "CHECK PASS" "$log" && [ "$launches" -gt 100 ]
