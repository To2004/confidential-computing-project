# report — the write-up

| File | Notes |
|---|---|
| `final_report.md` | The report. Source of truth — edit this |
| [`../final_report.pdf`](../final_report.pdf) | Built from the markdown. Kept at the repository root so it is the first thing a reader sees; regenerate after any edit |
| `project-proposal 5 - IN_DEPTH_ANALYSIS_OF_HOMOMORPHIC_ENCRYPTION_LIBRARIES.pdf` | The original project proposal |

## Rebuilding the PDF

Run from **this directory**, so the `../figures/…` image paths resolve:

```bash
cd report
pandoc final_report.md -o ../final_report.pdf \
  --pdf-engine=tectonic -V geometry:margin=2.2cm -V colorlinks=true -V fontsize=10pt
```

The output goes to the repository root while pandoc still runs from here, so the
`../figures/…` image paths resolve.

Install the toolchain with `conda install -c conda-forge pandoc tectonic`.

The PDF is **not** rebuilt by `run_all_benchmarks.sh`, because it needs a TeX
toolchain the benchmarks do not otherwise require. If you edit the markdown and
skip this step, the PDF goes stale.

## Note on notation

Numbers in the text use LaTeX math (`$10^{-6}$`, `$2^{40}$`) rather than Unicode
superscripts, because the default LaTeX fonts have no glyphs for `⁻⁶` and drop
them silently from the PDF. The math form renders correctly both on GitHub and
in the PDF.
