#!/usr/bin/env bash
# Verifies second-order convergence of the discretisation: solving on n = 127
# and n = 255 (h halves) must reduce max |u_h - u| by ~4x. Uses the fused
# CUDA variant, so it also checks that variant against an independent truth.
set -euo pipefail
bin="$1"
err() { "$bin" --variant fused --n "$1" --mode tol --tol 1e-10 | awk -F, '$3=="cuda"{print $17}'; }
e1=$(err 127)
e2=$(err 255)
ratio=$(awk -v a="$e1" -v b="$e2" 'BEGIN{printf "%.3f", a/b}')
echo "max_err n=127: $e1  n=255: $e2  ratio: $ratio (expect ~4 for O(h^2))"
awk -v r="$ratio" 'BEGIN{exit !(r > 3.6 && r < 4.4)}'
