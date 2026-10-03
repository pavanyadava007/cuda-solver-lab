#!/usr/bin/env bash
# Configure + build everything (Release, sm_89).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME" >/dev/null
cmake --build "$BUILD" -j "$(nproc)"
