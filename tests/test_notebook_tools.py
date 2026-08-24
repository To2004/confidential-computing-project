"""Tests for the notebook display helpers.

The point of `notebook_tools` is that a snippet shown in a notebook is read from
`src/` at run time and therefore cannot drift from the implementation. These
tests pin that property: the extracted text must be exactly what the file holds,
and the names the notebooks ask for must still exist.
"""

import ast
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import notebook_tools as nt
import project_paths


def read_src(filename):
    return io.open(os.path.join(project_paths.SRC_DIR, filename), encoding="utf-8").read()


class TestSourceOf(unittest.TestCase):

    def test_extracted_text_appears_verbatim_in_the_file(self):
        code = nt.source_of("benchmark_harness.py", "next_power_of_two")
        self.assertIn(code, read_src("benchmark_harness.py"))

    def test_extraction_starts_at_the_def_and_is_parseable(self):
        code = nt.source_of("benchmark_harness.py", "time_operation")
        self.assertTrue(code.lstrip().startswith("def time_operation("))
        ast.parse(code)  # a partial slice would raise

    def test_extraction_stops_at_the_end_of_the_definition(self):
        """The next top-level definition must not be swept in."""
        code = nt.source_of("benchmark_harness.py", "summarize")
        self.assertNotIn("def time_operation(", code)

    def test_a_class_method_can_be_addressed_with_a_dotted_name(self):
        code = nt.source_of("risk_score_benchmark.py", "TenSEALRiskScore.score")
        self.assertTrue(code.lstrip().startswith("def score("))

    def test_a_class_can_be_extracted_whole(self):
        code = nt.source_of("risk_score_benchmark.py", "TenSEALRiskScore")
        self.assertTrue(code.lstrip().startswith("class TenSEALRiskScore"))
        self.assertIn("def score(", code)

    def test_a_missing_name_raises_rather_than_returning_nothing(self):
        with self.assertRaises(NameError):
            nt.source_of("benchmark_harness.py", "no_such_function")


class TestAssignmentIn(unittest.TestCase):

    def test_operations_dict_is_extracted_from_inside_a_function(self):
        code = nt.assignment_in("tenseal_benchmark.py", "run_benchmark", "operations")
        self.assertTrue(code.lstrip().startswith("operations = {"))
        self.assertIn('"decrypt"', code)
        self.assertIn(code, read_src("tenseal_benchmark.py"))

    def test_a_missing_assignment_raises(self):
        with self.assertRaises(NameError):
            nt.assignment_in("tenseal_benchmark.py", "run_benchmark", "no_such_name")


class TestNamesTheNotebooksShow(unittest.TestCase):
    """Guards against a rename in src/ silently breaking a notebook cell."""

    SHOWN = [
        ("benchmark_harness.py", "time_operation"),
        ("benchmark_harness.py", "assert_accuracy"),
        ("benchmark_harness.py", "_batch_means_half_width_ms"),
        ("benchmark_harness.py", "_drift_diagnostics"),
        ("benchmark_harness.py", "_mann_kendall_z"),
        ("matched_comparison.py", "openfhe_arm"),
        ("matched_comparison.py", "probe_library_chosen_ring"),
        ("risk_score_benchmark.py", "pad_features"),
        ("risk_score_benchmark.py", "scaled_score_weights"),
        ("synthetic_patients.py", "normalization_affine_terms"),
        ("synthetic_patients.py", "multiplicative_depth_required"),
    ]

    def test_every_definition_a_notebook_displays_still_exists(self):
        for filename, name in self.SHOWN:
            with self.subTest(filename=filename, name=name):
                self.assertTrue(nt.source_of(filename, name).strip())

    def test_every_operations_dict_a_notebook_displays_still_exists(self):
        for filename in ("plaintext_baseline.py", "tenseal_benchmark.py",
                         "openfhe_benchmark.py"):
            with self.subTest(filename=filename):
                code = nt.assignment_in(filename, "run_benchmark", "operations")
                self.assertTrue(code.lstrip().startswith("operations = {"))


if __name__ == "__main__":
    unittest.main()
