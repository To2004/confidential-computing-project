#!/usr/bin/env bash
#
# Reproduce every benchmark in the project and regenerate all charts.
#
# Requires a Python environment with tenseal, openfhe, numpy and matplotlib
# importable (see README.md — OpenFHE ships Python 3.8 Linux-only wheels).
#
# Override the repetition count for a quick smoke run:
#   REPEATS=5 ./run_all_benchmarks.sh
#
# Skip the unit tests (not recommended — they catch a broken risk score before
# an hour of benchmarking is spent measuring it):
#   SKIP_TESTS=1 ./run_all_benchmarks.sh
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
  "${PYTHON}" -m unittest discover -p 'test_*.py'
fi

echo
echo "=== 1/6  Primitive operations (plaintext vs TenSEAL vs OpenFHE) ==="
"${PYTHON}" run_comparison.py --repeats "${REPEATS}"

echo
echo "=== 2/6  Scaling with vector size ==="
"${PYTHON}" benchmark_scaling.py --repeats "${REPEATS}"

echo
echo "=== 3/6  Encrypted synthetic medical risk score ==="
"${PYTHON}" risk_score_benchmark.py --repeats "${REPEATS}"

# The two experiments below run a full pipeline per repetition and exist to
# isolate a single effect, so they use a lower default repetition count.
echo
echo "=== 4/6  Matched-parameter comparison (OpenFHE at TenSEAL's parameters) ==="
"${PYTHON}" matched_comparison.py --repeats "$(( REPEATS < 200 ? REPEATS : 200 ))"

echo
echo "=== 5/6  IND-CPA^D noise flooding cost ==="
"${PYTHON}" ind_cpad_flooding.py --repeats "$(( REPEATS < 200 ? REPEATS : 200 ))"

echo
echo "=== 6/6  Charts ==="
"${PYTHON}" plot_results.py

echo
echo "All benchmarks complete."
