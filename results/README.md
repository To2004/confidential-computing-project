# results — benchmark output (JSON)

Written by the scripts in [`../src`](../src). Regenerate everything with
`./run_all_benchmarks.sh` from the repository root, or a single experiment with
its own script.

| File | Produced by | Contents |
|---|---|---|
| `results.json` | `run_comparison.py` | Primitive operations: plaintext vs TenSEAL vs OpenFHE, plus key generation |
| `scaling_results.json` | `benchmark_scaling.py` | Pipeline time at vector sizes 5–500 |
| `risk_score_results.json` | `risk_score_benchmark.py` | Encrypted synthetic risk score over 1000 patients |
| `matched_comparison_results.json` | `matched_comparison.py` | OpenFHE pinned to TenSEAL's ring dimension and scaling factor |
| `ind_cpad_results.json` | `ind_cpad_flooding.py` | Cost of the IND-CPA^D noise-flooding defence, and decryption-randomization measurements |

## Reading a timing entry

Each timed operation stores a `<op>_time_stats` object holding the mean, median,
10% trimmed mean, standard deviation, MAD, IQR, p95, p99, a 95% confidence
interval on the mean, a bootstrap interval on the median, and the **raw per-call
samples** in `samples_ms`. The samples are kept so a later question about the
distribution can be answered without re-running a benchmark that takes over an
hour — which is why these files are large.

**Check `drift_flagged` before trusting `ci95_half_width_ms`.** A confidence
interval says the samples agree with each other; it cannot detect the machine
changing underneath the benchmark. `drift_pct` compares the first half of the
samples against the second half, and anything above 5% means the measurement was
contaminated by competing load rather than being steady state.

The current files were produced on an exclusive SLURM node with threading pinned
(`submit_benchmarks.sbatch`); worst-case drift on any encrypted measurement is
2.9%. One plaintext operation is still flagged at 5.1% — it runs in about a
microsecond, where timer noise is comparable to the measurement itself.
