# notebooks

One notebook per experiment. Each installs the libraries, fetches the project,
runs one experiment, and shows its results.

| Notebook | Experiment | In the report |
|---|---|---|
| `01_basic_operations.ipynb` | Cost of encrypt, add, multiply, sum, average, dot product, decrypt | Experiment 1 |
| `02_matched_parameters.ipynb` | OpenFHE pinned to TenSEAL's parameters | Experiment 2 |
| `03_healthcare_risk_score.ipynb` | Encrypted risk score over 1000 synthetic patients | Experiment 3 |

Each notebook walks the same path: install and load the project, check which
libraries are available, read the code that does the work, run the experiment, and
show the results. Notebook 01 adds a sixth section that turns the harness's own
diagnostics on the run that just happened.

## What each notebook shows, beyond the numbers

A notebook that only shells out to a script shows a command and a table, not the
code a reader would want to interrogate. Each notebook therefore displays the
relevant source inline, read out of `../src` at run time by
[`notebook_tools.py`](../src/notebook_tools.py), so a snippet on screen cannot
drift from the implementation that produced the numbers.

| Notebook | Code it puts on screen | The question it answers |
|---|---|---|
| `01` | `time_operation`, `assert_accuracy`, and the `operations` dict from all three backends | How was this timed, and are the three columns measured the same way? |
| `01` | `_batch_means_half_width_ms`, `_drift_diagnostics`, `_mann_kendall_z`, then applied to the run's own samples | How do you know the numbers are steady state and the intervals honest? |
| `02` | `openfhe_arm`, `probe_library_chosen_ring` | What exactly was matched, and what did matching it cost? |
| `03` | `pad_features`, `normalization_affine_terms`, `scaled_score_weights`, `multiplicative_depth_required` | Why does the padding *value* change the answer, and why did the ring have to double? |

Notebook 03 also computes the padding failure in the clear, with no encryption
involved: padding 1000 patients to 1024 slots with each feature's minimum leaves the
cohort mean unchanged, while padding with zeros shifts it by about one full score
point — large enough to matter, small enough to look plausible.

## Running them

Open a notebook in Colab, or run it locally in the `he38` environment described
in the [project README](../README.md). No manual setup is needed either way —
the first cells install the dependencies, and clone the repository if the
notebook is not already inside a checkout.

## Two things to know

**OpenFHE may not import.** Every OpenFHE distribution on PyPI is tagged
`py3-none-any`, but the payload is Linux ELF binaries built against one specific
CPython — `1.4.1.0.20.4` for 3.8, `1.4.1.0.22.4` for 3.10, `1.4.1.0.24.4` for 3.12.
There has never been a Windows wheel. Because those bounds are `>=`, an unpinned
install on, say, CPython 3.11 silently resolves to the 3.10 build and then fails at
*import*, so the first cell pins the build to the running interpreter or skips it
with a reason. Notebooks 01 and 03 detect the absence and run with the plaintext
and TenSEAL columns only; notebook 02 measures OpenFHE itself and needs it present.

**The notebooks use fewer repetitions than the report.** Each has a `REPEATS`
constant near the top, set low enough to finish in a couple of minutes. The report
uses 1000 repetitions on a reserved compute node, so notebook numbers are noisier
and will not match the report exactly. Notebook 01 uses 200 rather than something
smaller because the batch-means confidence interval needs at least 100 samples;
below that the harness falls back to the naive i.i.d. formula and section 6 has
nothing to compare. Expect drift flags on a laptop or a hosted runtime — that is
the check working, not a defect.

Output goes to `notebook_output/`, which is git-ignored, so running a notebook
never overwrites the `results/` and `figures/` the report was built from.

The notebooks import from [`../src`](../src) rather than copying it, so they run
the same code the report's numbers came from.
