#!/usr/bin/env bash
#
# Reproduce every benchmark in the project and regenerate all charts.
#
# Requires a Python environment with tenseal, openfhe, numpy and matplotlib
# importable (see README.md — OpenFHE ships Linux-only wheels).
#
# Override the repetition count for a quick smoke run:
#   REPEATS=5 ./run_all_benchmarks.sh
#
# Skip the unit tests (not recommended — they catch a broken risk score before
# an hour of benchmarking is spent measuring it):
#   SKIP_TESTS=1 ./run_all_benchmarks.sh
#
# Layout: sources in src/, JSON results in results/, charts in figures/. Scripts
# resolve those directories from their own location (see src/project_paths.py),
# so this works from any working directory.
#
set -euo pipefail

REPEATS="${REPEATS:-1000}"
PYTHON="${PYTHON:-python}"
SKIP_TESTS="${SKIP_TESTS:-0}"

cd "$(dirname "$0")"

echo "############################################################"
echo "# Repetitions per measurement: ${REPEATS}"
echo "# Python: $(${PYTHON} -V 2>&1)"
echo "# Host:   $(hostname)"
echo "# Load:   $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo n/a)"
echo "############################################################"

if [ "${SKIP_TESTS}" != "1" ]; then
  echo
  echo "=== 0/6  Unit tests ==="
  "${PYTHON}" -m unittest discover -s tests
fi

echo
echo "=== 1/4  Primitive operations (plaintext vs TenSEAL vs OpenFHE) ==="
"${PYTHON}" src/run_comparison.py --repeats "${REPEATS}"

echo
echo "=== 2/4  Encrypted synthetic medical risk score ==="
"${PYTHON}" src/risk_score_benchmark.py --repeats "${REPEATS}"

# The matched-parameter run evaluates a full pipeline per repetition and exists
# to isolate a single effect, so it uses a lower default repetition count.
echo
echo "=== 3/4  Matched-parameter comparison (OpenFHE at TenSEAL's parameters) ==="
"${PYTHON}" src/matched_comparison.py --repeats "$(( REPEATS < 200 ? REPEATS : 200 ))"

echo
echo "=== 4/4  Charts ==="
"${PYTHON}" src/plot_results.py

echo
echo "All benchmarks complete."
