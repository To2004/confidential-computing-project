#!/usr/bin/env bash
#SBATCH --job-name=he-confirm
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-he-confirm-%j.out
#SBATCH --error=slurm-he-confirm-%j.out
#
# Re-run only the experiment whose numbers changed with the review fixes:
#   * risk_score_benchmark  -- the plaintext stages are now disjoint, so the
#                              overhead denominator changed
#
# Why this is NOT the exclusive job. `submit_benchmarks.sbatch` asks for a whole
# node, which on this cluster queues indefinitely because the site scheduler
# routes every non-GPU job into a `cpu` partition that has no fully idle nodes.
# This job asks for a share instead, so it schedules — at the cost of sharing the
# machine with other work.
#
# That trade is only acceptable because the harness can tell us whether it
# mattered: every measurement carries drift and autocorrelation diagnostics, and
# the numbers are usable only if those come back clean. Check them before
# believing anything this job produces. The primitive-operation and
# matched-parameter figures in the report come from a genuinely exclusive run
# (job 19649703) and are not touched here.
#
set -euo pipefail

REPEATS="${REPEATS:-1000}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate he38

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

echo "############################################################"
echo "# SLURM job   : ${SLURM_JOB_ID:-n/a}   node: $(hostname)"
echo "# CPUs        : ${SLURM_CPUS_PER_TASK:-?} (shared node, NOT exclusive)"
echo "# Load start  : $(cut -d' ' -f1-3 /proc/loadavg)"
echo "# Repetitions : ${REPEATS}"
echo "############################################################"

python -m unittest discover -s tests

echo
echo "=== Encrypted composite score (disjoint plaintext stages) ==="
python src/risk_score_benchmark.py --repeats "${REPEATS}"

echo
echo "# Load end    : $(cut -d' ' -f1-3 /proc/loadavg)"
echo "Done."
