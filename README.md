# In-Depth Analysis of Homomorphic Encryption Libraries

Benchmark comparison of **TenSEAL** and **OpenFHE** — both using the CKKS scheme —
against an unencrypted baseline, for the Confidential Computing course at
Ben-Gurion University of the Negev.

The project measures what homomorphic encryption actually costs: runtime, memory,
approximation error, and how all three behave on a computation more realistic than
a single average. The write-up is in [`final_report.md`](final_report.md).

## What is measured

| Experiment | Script | Output |
|---|---|---|
| Primitive operations (encrypt, add, multiply, sum, average, dot product, decrypt) + key generation | `run_comparison.py` | `results.json` |
| Scaling with vector size (5 → 500 elements) | `benchmark_scaling.py` | `scaling_results.json` |
| Encrypted synthetic medical risk score over 1000 patients | `risk_score_benchmark.py` | `risk_score_results.json` |
| OpenFHE forced onto TenSEAL's parameters (like-for-like) | `matched_comparison.py` | `matched_comparison_results.json` |
| IND-CPA^D: is decryption randomized, and what does flooding cost | `ind_cpad_flooding.py` | `ind_cpad_results.json` |
| All charts | `plot_results.py` | `chart_*.png` |

Every timed figure is a **mean over 1000 repetitions**, taken after a warm-up phase
that is discarded. See [Methodology](#methodology) below.

## Environment

OpenFHE is the constraint. Its PyPI wheels are **Linux-only** and are built for a
**specific CPython version** — the current wheel ships a `cpython-38` shared
object, so it imports only under **Python 3.8**. Installing it under 3.9+ appears
to succeed and then fails at import with `No module named 'openfhe.openfhe'`.

```bash
conda create -y -n he38 python=3.8
conda activate he38
pip install tenseal openfhe numpy matplotlib
```

Verify both libraries load before running anything:

```bash
python -c "import tenseal, openfhe; print('ok')"
```

TenSEAL alone installs cleanly on Windows, macOS and Linux across Python
versions. On Windows, run the OpenFHE half under WSL or on a Linux host; the
scripts degrade gracefully and report the plaintext and TenSEAL results only.

## Running the benchmarks

Everything, in order, then the charts:

```bash
./run_all_benchmarks.sh
```

A fast smoke run — same code path, far fewer repetitions, numbers not
report-quality:

```bash
REPEATS=5 ./run_all_benchmarks.sh
```

Individual experiments, with the repetition count under your control:

```bash
python run_comparison.py       --repeats 1000 --warmup 50
python benchmark_scaling.py    --repeats 1000 --warmup 20
python risk_score_benchmark.py --repeats 1000 --warmup 20
python plot_results.py
```

Useful flags: `--patients` and `--seed` on the risk-score benchmark,
`--skip-openfhe` to run without OpenFHE, `--keygen-repeats` on the comparison
(key generation is a one-off setup cost, so it is timed separately and with fewer
repetitions).

At 1000 repetitions the full suite takes roughly 1.5 hours, dominated by OpenFHE.

### Run it on an exclusive node, not a shared one

**This matters more than it sounds.** Our first full run was done on a shared login
node carrying a load average of 10.4 on 8 cores. The harness's drift check caught
the consequence: for several operations the first and second halves of the samples
differed by 10-30% (73% in one case), meaning the samples were not drawn from one
distribution and the confidence intervals computed from them were meaningless —
narrow, but meaningless.

Competing load cannot be averaged away. On a SLURM cluster:

```bash
sbatch submit_benchmarks.sbatch
```

That requests an exclusive node and pins threading (`OMP_NUM_THREADS=1`) so both
libraries are measured on identical footing. Worst-case drift dropped from 30.9%
to 1.1%. If you run on a shared machine anyway, check the `drift_pct` and
`drift_flagged` fields in the results JSON before believing any number.

### Tests

```bash
python -m unittest discover -p 'test_*.py'
```

40 tests covering the statistics (against hand-computed values), the risk score
(against a hand-computed patient), and the ciphertext-padding invariant — the last
of these guards an error that would otherwise produce plausible-looking wrong
results.

### Charts produced

| File | Content |
|---|---|
| `chart_operations.png` | Mean time per operation, and slowdown vs plaintext |
| `chart_memory.png` | Peak Python-side memory per operation |
| `chart_errors.png` | CKKS approximation error per operation |
| `chart_scaling.png` | Pipeline time vs vector size |
| `chart_risk_score.png` | Risk-score stage costs, cohort distribution, accuracy |
| `chart_matched_comparison.png` | OpenFHE at TenSEAL's parameters: speed and precision |
| `chart_ind_cpad.png` | Cost of the IND-CPA^D defence, and which libraries randomize decryption |

Bars and markers carry 95% confidence intervals. Series colours are a
colourblind-safe categorical set, checked for deuteranopia, protanopia and
tritanopia separation.

### Regenerating the report PDF

`final_report.pdf` is built from `final_report.md`. It is **not** produced by
`run_all_benchmarks.sh`, because it needs a TeX toolchain that the benchmark
environment does not otherwise require:

```bash
pandoc final_report.md -o final_report.pdf \
  --pdf-engine=tectonic -V geometry:margin=2.2cm -V colorlinks=true
```

Install those with `conda install -c conda-forge pandoc tectonic`. If you edit
`final_report.md`, re-run this or the PDF will go stale.

## Methodology

**Warm-up.** The first calls into either library pay one-off costs that are not
part of the steady-state cost of an operation: loading the native library, lazily
building NTT tables and evaluation-key caches, first-touch page faults, and CPU
frequency ramp-up. Including them would inflate the mean and make it depend on how
many repetitions were run, so a warm-up phase runs first and is discarded.

**Repetitions.** Each operation is then timed individually 1000 times and the mean
reported, alongside the standard deviation, robust statistics (median, 10% trimmed
mean, MAD, IQR, p95, p99, bootstrap interval for the median) and a 95% confidence
interval on the mean.

**A confidence interval is not a certificate.** It measures how consistent the
samples are *with each other*, assuming they are independent draws from one
distribution. If the machine changes underneath the benchmark, that assumption
fails and the interval does not notice — our contaminated run had narrow intervals
*and* 73% drift. The harness therefore also compares the first half of the samples
against the second half and flags a measurement when they disagree by more than
5%. Read `drift_flagged` before trusting `ci95_half_width_ms`.

**Timing and memory are measured separately.** `tracemalloc` hooks every Python
allocation and measurably distorts short timings, so it is never active while the
clock is running. Garbage collection is disabled inside the timed loop, as
`timeit` does, so an unrelated collection is not charged to the operation under
test.

**Fresh operands.** Every repetition operates on the same freshly encrypted
inputs, so no repetition inherits accumulated noise or a consumed multiplicative
level from the one before it.

All of this lives in [`benchmark_harness.py`](benchmark_harness.py).

## The medical use case

`synthetic_patients.py` generates a synthetic cohort of 1000 patients with five
features (age, systolic blood pressure, BMI, cholesterol, glucose) and defines a
**synthetic risk score** over them. `risk_score_benchmark.py` evaluates that score
homomorphically and compares it against the plaintext reference.

> The risk score is a synthetic construct built for demonstration purposes only.
> It is **not** a validated clinical model, it is not derived from medical
> literature, and it carries no medical meaning. The cohort is likewise
> synthetic — values are drawn from plausible-looking distributions, not from
> patients.

Its purpose is to give the benchmark a computation with real multiplicative
depth. Beyond a weighted average of normalized features, it includes two
interaction terms between distinct features and one quadratic term, so it
requires ciphertext × ciphertext multiplication rather than only multiplication by
public constants. The exact definition is in the `synthetic_patients` module
docstring.

## Files

| File | Purpose |
|---|---|
| `benchmark_harness.py` | Warm-up, repetition, and statistics shared by every benchmark |
| `plaintext_baseline.py` | Unencrypted reference implementation |
| `tenseal_benchmark.py` | TenSEAL (CKKS) primitive operations |
| `openfhe_benchmark.py` | OpenFHE (CKKS) primitive operations |
| `run_comparison.py` | Runs all three, prints the comparison, writes `results.json` |
| `benchmark_scaling.py` | Pipeline time at vector sizes 5–500 |
| `synthetic_patients.py` | Synthetic cohort + synthetic risk score definition |
| `risk_score_benchmark.py` | Encrypted evaluation of the risk score |
| `matched_comparison.py` | OpenFHE forced onto TenSEAL's ring dimension and scaling factor |
| `ind_cpad_flooding.py` | IND-CPA^D: is decryption randomized, and what does flooding cost |
| `test_benchmark_harness.py` | Unit tests for the statistics (mean, median, drift, memory) |
| `test_synthetic_patients.py` | Unit tests for the risk score and the padding invariant |
| `submit_benchmarks.sbatch` | Run the suite on an exclusive SLURM node (see below) |
| `plot_results.py` | All charts |
| `run_all_benchmarks.sh` | Reproduce everything end to end |
| `colab_benchmark.ipynb` | Self-contained notebook version (Colab is Linux, so both libraries work there) |
| `final_report.md` | The report |

## Authors

Itai Zloclzower, Jacob Shemesh, Tomer Ovadia, Yoni Glickstein
