# figures — generated charts

All produced by [`../src/plot_results.py`](../src/plot_results.py) from the JSON
in [`../results`](../results). Do not edit by hand; regenerate with:

```bash
python src/plot_results.py
```

| File | Content |
|---|---|
| `chart_operations.png` | Mean time per operation, and slowdown versus plaintext |
| `chart_memory.png` | What `tracemalloc` sees beside what a ciphertext actually costs |
| `chart_errors.png` | CKKS approximation error per operation |
| `chart_risk_score.png` | Risk-score stage costs, cohort distribution, accuracy |
| `chart_matched_comparison.png` | OpenFHE at TenSEAL's parameters: speed and precision |

Bars and markers carry 95% confidence intervals. Series colours are a
colourblind-safe categorical set, checked for deuteranopia, protanopia and
tritanopia separation against a light surface.

These files are referenced by [`../report/final_report.md`](../report/final_report.md)
as `../figures/…`, which resolves correctly both on GitHub and for the PDF build.
