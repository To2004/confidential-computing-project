# notebooks

One notebook per experiment. Each installs the libraries, fetches the project,
runs one experiment, and shows its results.

| Notebook | Experiment | In the report |
|---|---|---|
| `01_basic_operations.ipynb` | Cost of encrypt, add, multiply, sum, average, dot product, decrypt | Experiment 1 |
| `02_input_size.ipynb` | Runtime at vector sizes 5 to 500 | no — repository only |
| `03_matched_parameters.ipynb` | OpenFHE pinned to TenSEAL's parameters | Experiment 2 |
| `04_healthcare_risk_score.ipynb` | Encrypted risk score over 1000 synthetic patients | Experiment 3 |
| `05_noise_flooding.ipynb` | Cost of OpenFHE's noise-flooding decryption | no — beyond the proposal |

Every notebook has the same four steps: install and load the project, check
which libraries are available, run the experiment, show the results.

## Running them

Open a notebook in Colab, or run it locally in the `he38` environment described
in the [project README](../README.md). No manual setup is needed either way —
the first cells install the dependencies, and clone the repository if the
notebook is not already inside a checkout.

## Two things to know

**OpenFHE needs Python 3.8.** Its wheel ships a `cpython-38` binary, so on a
current Colab runtime it installs and then fails to import. Notebooks 01, 02 and
04 detect this and run with the plaintext and TenSEAL columns only. Notebooks 03
and 05 measure OpenFHE itself and cannot run without it.

**The notebooks use fewer repetitions than the report.** Each has a `REPEATS`
constant near the top, set low enough to finish in a few minutes. The report
uses 1000 repetitions on a reserved compute node, so notebook numbers are
noisier and will not match the report exactly.

Output goes to `notebook_output/`, which is git-ignored, so running a notebook
never overwrites the `results/` and `figures/` the report was built from.

The notebooks import from [`../src`](../src) rather than copying it, so they run
the same code the report's numbers came from.
