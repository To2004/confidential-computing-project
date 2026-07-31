"""
Where the project keeps its inputs and outputs.

Every script resolves its paths through this module rather than through the
current working directory, so `python src/run_comparison.py` and
`cd src && python run_comparison.py` write to the same place. Paths are derived
from this file's own location, which is the only fixed point available.
"""

import os

# src/ -> repository root
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)

RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
REPORT_DIR = os.path.join(REPO_ROOT, "report")


def result_path(filename):
    """Absolute path to a results JSON file, creating the directory if needed."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, filename)


def figure_path(filename):
    """Absolute path to a chart image, creating the directory if needed."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, filename)


def relative_to_root(path):
    """Path shown to the user, relative to the repository root when possible."""
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:  # different drive on Windows
        return path
