# notebooks

| File | Notes |
|---|---|
| `colab_benchmark.ipynb` | Self-contained Colab version of the whole benchmark |

The notebook duplicates the harness and the benchmarks deliberately: it installs
its own dependencies and imports nothing from [`../src`](../src), so it runs on a
fresh Colab runtime with no repository checkout. The trade-off is that changes to
`src/` do not propagate here automatically.

**OpenFHE will not import on a current Colab runtime.** Its wheel ships a
`cpython-38` shared object, and Colab runs a newer Python. The notebook detects
this and runs the plaintext and TenSEAL sections only, reporting the reason. To
exercise the OpenFHE half, use a Python 3.8 environment — see the
[project README](../README.md).
