#!/usr/bin/env bash
# One command: build, test, sanitize, benchmark, profile, report.
#   ./run_all.sh            (about 10 minutes on an idle L4)
set -euo pipefail
cd "$(dirname "$0")"
source scripts/env.sh
./scripts/build.sh
(cd build && ctest --output-on-failure) | tee results/ctest.log
./scripts/sanitize.sh
./scripts/bench.sh
./scripts/profile.sh
./scripts/profile_fortran.sh
[ -x .venv/bin/python ] || { uv venv -q --python 3.10 .venv && uv pip install -q --python .venv/bin/python "numpy<2" matplotlib; }
.venv/bin/python scripts/make_report.py > /dev/null
echo "done: see results/RESULTS.md and README.md"
