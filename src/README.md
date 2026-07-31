# src — benchmark and experiment code

All Python for the project. Every script is runnable on its own and resolves the
`results/` and `figures/` directories from its own location via
[`project_paths.py`](project_paths.py), so it behaves the same whether you run
`python src/run_comparison.py` from the repository root or `python
run_comparison.py` from inside this directory.

## Shared modules

| File | Purpose |
|---|---|
| `project_paths.py` | Locates `results/` and `figures/` relative to the repository root |
| `benchmark_harness.py` | Warm-up, repetitions, robust statistics, drift detection, memory measurement, accuracy gates — used by every experiment |
| `synthetic_patients.py` | The synthetic cohort and the synthetic risk score definition |

## Benchmarks

| File | Writes |
|---|---|
| `plaintext_baseline.py` | (library module; unencrypted reference) |
| `tenseal_benchmark.py` | (library module; TenSEAL CKKS primitives) |
| `openfhe_benchmark.py` | (library module; OpenFHE CKKS primitives) |
| `run_comparison.py` | `results/results.json` |
| `benchmark_scaling.py` | `results/scaling_results.json` |
| `risk_score_benchmark.py` | `results/risk_score_results.json` |
| `matched_comparison.py` | `results/matched_comparison_results.json` |
| `ind_cpad_flooding.py` | `results/ind_cpad_results.json` |
| `plot_results.py` | `figures/chart_*.png` |

The three `*_benchmark.py` files double as importable modules and as standalone
scripts; `run_comparison.py` imports all three and drives them together.

## Dependency direction

```
project_paths ──┐
                ├──> run_comparison ──> plaintext_baseline
benchmark_harness┤                      tenseal_benchmark
                 │                      openfhe_benchmark
                 ├──> benchmark_scaling
                 ├──> matched_comparison
                 └──> risk_score_benchmark <── ind_cpad_flooding
                             │
                      synthetic_patients
```

`ind_cpad_flooding.py` deliberately imports `risk_score_benchmark` and evaluates
*its* circuit under noise-flooding parameters rather than reimplementing it, so
the security experiment and the performance experiment cannot drift apart.

## Running

See the [project README](../README.md). Everything in order:

```bash
./run_all_benchmarks.sh          # from the repository root
```
