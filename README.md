# In-Depth Analysis of Homomorphic Encryption Libraries

Benchmark comparison of **TenSEAL** and **OpenFHE** — both using the CKKS scheme —
against an unencrypted baseline, for the Confidential Computing course at
Ben-Gurion University of the Negev.

The project measures what homomorphic encryption actually costs: runtime, memory,
approximation error, and how all three behave on a computation more realistic than
a single average. The write-up is in [`report/final_report.md`](report/final_report.md).

## What is measured

| Experiment | Script | Output |
|---|---|---|
| Primitive operations (encrypt, add, multiply, sum, average, dot product, decrypt) + key generation | `src/run_comparison.py` | `results/results.json` |
| Scaling with vector size (5 → 500 elements) | `src/benchmark_scaling.py` | `results/scaling_results.json` |
| Encrypted synthetic medical risk score over 1000 patients | `src/risk_score_benchmark.py` | `results/risk_score_results.json` |
| OpenFHE forced onto TenSEAL's parameters (like-for-like) | `src/matched_comparison.py` | `results/matched_comparison_results.json` |
| IND-CPA^D: is decryption randomized, and what does flooding cost | `src/ind_cpad_flooding.py` | `results/ind_cpad_results.json` |
| All charts | `src/plot_results.py` | `figures/chart_*.png` |

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
python src/run_comparison.py       --repeats 1000 --warmup 50
python src/benchmark_scaling.py    --repeats 1000 --warmup 20
python src/risk_score_benchmark.py --repeats 1000 --warmup 20
python src/matched_comparison.py   --repeats 200
python src/ind_cpad_flooding.py    --repeats 200
python src/plot_results.py
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
libraries are measured on identical footing. Worst-case drift on the encrypted
measurements dropped from 30.9% to 2.9%. If you run on a shared machine anyway, check the `drift_pct` and
`drift_flagged` fields in the results JSON before believing any number.

### Tests

```bash
python -m unittest discover -s tests
```

40 tests covering the statistics (against hand-computed values), the risk score
(against a hand-computed patient), and the ciphertext-padding invariant — the last
of these guards an error that would otherwise produce plausible-looking wrong
results.

### Charts produced

| File | Content |
|---|---|
| `figures/chart_operations.png` | Mean time per operation, and slowdown vs plaintext |
| `figures/chart_memory.png` | Peak Python-side memory per operation |
| `figures/chart_errors.png` | CKKS approximation error per operation |
| `figures/chart_scaling.png` | Pipeline time vs vector size |
| `figures/chart_risk_score.png` | Risk-score stage costs, cohort distribution, accuracy |
| `figures/chart_matched_comparison.png` | OpenFHE at TenSEAL's parameters: speed and precision |
| `figures/chart_ind_cpad.png` | Cost of the IND-CPA^D defence, and which libraries randomize decryption |

Bars and markers carry 95% confidence intervals. Series colours are a
colourblind-safe categorical set, checked for deuteranopia, protanopia and
tritanopia separation.

### Regenerating the report PDF

`final_report.pdf` is built from `final_report.md`. It is **not** produced by
`run_all_benchmarks.sh`, because it needs a TeX toolchain that the benchmark
environment does not otherwise require:

```bash
cd report
pandoc final_report.md -o final_report.pdf \
  --pdf-engine=tectonic -V geometry:margin=2.2cm -V colorlinks=true -V fontsize=10pt
```

Run it from `report/` so the `../figures/…` image paths resolve.

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

All of this lives in [`src/benchmark_harness.py`](src/benchmark_harness.py).

## The medical use case

`src/synthetic_patients.py` generates a synthetic cohort of 1000 patients with five
features (age, systolic blood pressure, BMI, cholesterol, glucose) and defines a
**synthetic risk score** over them. `src/risk_score_benchmark.py` evaluates that score
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

## Repository layout

```
.
├── run_all_benchmarks.sh     reproduce every experiment, then the charts
├── submit_benchmarks.sbatch  the same, on an exclusive SLURM node
├── src/                      benchmark and experiment code
├── tests/                    unit tests (40, plain unittest)
├── results/                  benchmark output, JSON
├── figures/                  generated charts, PNG
├── report/                   final_report.md, its PDF, and the proposal
└── notebooks/                self-contained Colab notebook
```

Each directory has its own `README.md` describing what is in it and how it is
produced. The scripts locate `results/` and `figures/` from their own location
(see [`src/project_paths.py`](src/project_paths.py)), so they behave identically
whether you run them from the repository root or from inside `src/`.

Start here:

| I want to… | Go to |
|---|---|
| Read the findings | [`report/final_report.md`](report/final_report.md) |
| Understand the measurement method | [`src/benchmark_harness.py`](src/benchmark_harness.py) |
| See the risk score definition | [`src/synthetic_patients.py`](src/synthetic_patients.py) |
| Reproduce the numbers | `./run_all_benchmarks.sh` |
| Check what the numbers mean | [`results/README.md`](results/README.md) |

## Authors

Itai Zloclzower, Jacob Shemesh, Tomer Ovadia, Yoni Glickstein
