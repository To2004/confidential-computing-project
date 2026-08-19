# report — the write-up

| File | Notes |
|---|---|
| `final_report.tex` | The report. Source of truth — edit this |
| [`../final_report.pdf`](../final_report.pdf) | Built from the LaTeX source. Kept at the repository root so it is the first thing a reader sees; regenerate after any edit |

## Rebuilding the PDF

Run from **this directory**, so the `../figures/…` image paths resolve:

```bash
cd report
tectonic final_report.tex && mv final_report.pdf ..
```

`pdflatex` works too, but needs two passes so that `\ref` cross-references and the
bibliography numbering resolve:

```bash
pdflatex final_report.tex && pdflatex final_report.tex
mv final_report.pdf ..
```

Install the toolchain with `conda install -c conda-forge tectonic`, or use any TeX
distribution that provides `pdflatex`.

The PDF is **not** rebuilt by `run_all_benchmarks.sh`, because it needs a TeX
toolchain the benchmarks do not otherwise require. If you edit the source and skip
this step, the PDF goes stale.

## Note on notation

Numbers in the text use LaTeX math (`$10^{-6}$`, `$2^{40}$`) rather than Unicode
superscripts, because the default LaTeX fonts have no glyphs for `⁻⁶` and drop them
silently from the PDF.
