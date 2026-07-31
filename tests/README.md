# tests — unit tests

Run from the repository root:

```bash
python -m unittest discover -s tests
```

40 tests, no external test framework required — plain `unittest`. Each file
inserts `../src` onto `sys.path` at import time, so the tests work regardless of
how they are invoked.

| File | Covers |
|---|---|
| `test_benchmark_harness.py` | Summary statistics against hand-computed values; that the median and trimmed mean resist an outlier the mean does not; percentile interpolation; confidence-interval shrinkage (including Bessel's correction); bootstrap reproducibility; drift detection; warm-up exclusion; memory measurement |
| `test_synthetic_patients.py` | Normalization endpoints; the affine form used by the encrypted path matching direct normalization; the risk score against a hand-computed patient (all features at range midpoint must score exactly 45.0); cohort reproducibility and range clipping; **the padding invariant** |

## Why the padding test matters

`test_padding_with_zeros_would_corrupt_the_sum` pins down a bug that would
otherwise produce plausible-looking wrong numbers. The encrypted risk score pads
each feature vector to a power of two; those slots must be filled with the
feature's range *minimum* so they normalize to exactly 0 and drop out of the
cohort sum. Padding with zeros instead gives them a nonzero normalized value of
`-min/(max-min)` and silently corrupts the cohort mean.
