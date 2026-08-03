"""Project input/output directories.

Paths come from this file's location, not the working directory, so scripts
write to the same place however they are launched.
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
    """Path relative to the repository root, or unchanged if that is impossible."""
    try:
        return os.path.relpath(path, REPO_ROOT)
    except ValueError:  # different drive on Windows
        return path
