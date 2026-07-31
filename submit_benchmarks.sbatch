#!/usr/bin/env bash
#SBATCH --job-name=he-benchmark
#SBATCH --partition=cpu
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-he-benchmark-%j.out
#SBATCH --error=slurm-he-benchmark-%j.out
#
# Run the full benchmark suite on an EXCLUSIVE compute node.
#
# Why this exists
# ---------------
# The first full run was done on a shared login node carrying a load average of
# 10.4 on 8 cores. The harness's drift check caught the consequence: the first
# and second halves of the samples for several operations differed by 10-30%,
# meaning the samples were not drawn from one distribution and the confidence
# intervals computed from them were meaningless — narrow, but meaningless.
#
# Competing load cannot be averaged away, so the fix is to not share the machine.
# --exclusive gives the whole node, and pinning the thread count removes the
# remaining source of run-to-run variation.
#
# Usage:
#   sbatch submit_benchmarks.sbatch            # 1000 repetitions
#   REPEATS=50 sbatch submit_benchmarks.sbatch # quick check
#
set -euo pipefail

REPEATS="${REPEATS:-1000}"

# Single-threaded, deliberately. Both libraries use multithreaded native code, so
# on a shared machine their measured cost depends on how many cores they happened
# to get — which is not a property of the library. Pinning to one thread makes
# the comparison reproducible and puts both libraries on identical footing. It
# raises absolute times relative to a multithreaded run; the ratios are the
# transferable result.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONDA_SH="/storage/modules/packages/anaconda/etc/profile.d/conda.sh"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate he38

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "############################################################"
echo "# SLURM job     : ${SLURM_JOB_ID:-n/a}"
echo "# Node          : $(hostname)"
echo "# Exclusive     : ${SLURM_JOB_CPUS_PER_NODE:-?} CPUs allocated"
echo "# Load at start : $(cut -d' ' -f1-3 /proc/loadavg)"
echo "# Threads       : OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "# Repetitions   : ${REPEATS}"
echo "# Python        : $(python -V 2>&1)"
echo "############################################################"

python -m unittest discover -p 'test_*.py'

REPEATS="${REPEATS}" SKIP_TESTS=1 ./run_all_benchmarks.sh

echo
echo "# Load at end   : $(cut -d' ' -f1-3 /proc/loadavg)"
echo "Done."
