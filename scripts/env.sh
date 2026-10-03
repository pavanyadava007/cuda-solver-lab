# Sourced by the other scripts: tool paths and shared helpers.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
RAW="$ROOT/results/raw"

# The host is shared: wait (up to 30 min) until no other process uses the GPU,
# then log what was running so contention is visible next to the numbers.
wait_for_idle_gpu() {
  local waited=0
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ $waited -lt 1800 ]; do
    sleep 10; waited=$((waited + 10))
  done
  local others
  others=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')
  echo "$(date -Iseconds),$1,waited_s=$waited,other_gpu_procs=[${others}],util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)" \
    >> "$RAW/gpu_contention.log"
}
