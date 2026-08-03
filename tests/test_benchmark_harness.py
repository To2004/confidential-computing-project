"""
Tests for the statistics in `benchmark_harness`, checked against hand-computed
values. Run with: python -m unittest discover -p 'test_*.py'
"""

import os
import sys
import unittest

# Modules under test live in src/; add it to the path so the tests run under
# unittest discover, pytest, or directly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))


import benchmark_harness as harness


def stats_for_ms(values_ms, **kwargs):
    """Summarize values given in milliseconds (the harness takes seconds)."""
    return harness.summarize([v / 1000.0 for v in values_ms], **kwargs)


class TestSummarize(unittest.TestCase):

    def test_known_values(self):
        # 1..9 ms: mean 5, median 5, sample sd 2.7386.
        st = stats_for_ms(list(range(1, 10)), bootstrap=False)
        self.assertAlmostEqual(st["mean_ms"], 5.0, places=9)
        self.assertAlmostEqual(st["median_ms"], 5.0, places=9)
        self.assertAlmostEqual(st["std_ms"], 2.7386127875258306, places=9)
        self.assertAlmostEqual(st["min_ms"], 1.0, places=9)
        self.assertAlmostEqual(st["max_ms"], 9.0, places=9)
        self.assertEqual(st["repeats"], 9)

    def test_single_sample_has_no_spread(self):
        st = stats_for_ms([4.2], bootstrap=False)
        self.assertAlmostEqual(st["mean_ms"], 4.2, places=9)
        self.assertEqual(st["std_ms"], 0.0)
        self.assertEqual(st["ci95_half_width_ms"], 0.0)
        self.assertEqual(st["mad_ms"], 0.0)

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            harness.summarize([])

    def test_median_resists_an_outlier_but_mean_does_not(self):
        clean = [10.0] * 100
        with_outlier = clean[:-1] + [1000.0]

        st = stats_for_ms(with_outlier, bootstrap=False)
        # One sample of 100 at 100x moves the mean from 10 to 19.9.
        self.assertAlmostEqual(st["mean_ms"], 19.9, places=6)
        self.assertAlmostEqual(st["median_ms"], 10.0, places=9)
        self.assertAlmostEqual(st["trimmed_mean_ms"], 10.0, places=9)
        self.assertAlmostEqual(st["mad_ms"], 0.0, places=9)
        self.assertGreater(st["mean_over_median_pct"], 90.0)

    def test_percentiles_interpolate(self):
        st = stats_for_ms([float(v) for v in range(1, 101)], bootstrap=False)
        # 1..100 with linear interpolation on ranks: p95 = 95.05, p99 = 99.01.
        self.assertAlmostEqual(st["p95_ms"], 95.05, places=6)
        self.assertAlmostEqual(st["p99_ms"], 99.01, places=6)
        # IQR = 75.25 - 25.75 = 49.5
        self.assertAlmostEqual(st["iqr_ms"], 49.5, places=6)

    def test_outlier_fraction_counts_only_the_upper_tail(self):
        values = [10.0] * 95 + [500.0] * 5
        st = stats_for_ms(values, bootstrap=False)
        self.assertAlmostEqual(st["outlier_fraction_pct"], 5.0, places=6)

        # Symmetric spread, no extreme upper tail.
        symmetric = [float(v) for v in range(1, 101)]
        self.assertEqual(stats_for_ms(symmetric, bootstrap=False)["outlier_fraction_pct"], 0.0)

    def test_confidence_interval_shrinks_with_sample_count(self):
        few = stats_for_ms([9.0, 10.0, 11.0] * 5, bootstrap=False)      # n = 15
        many = stats_for_ms([9.0, 10.0, 11.0] * 500, bootstrap=False)   # n = 1500
        self.assertLess(many["ci95_half_width_ms"], few["ci95_half_width_ms"])

        # Ratio is (sd_15 / sqrt(15)) / (sd_1500 / sqrt(1500)) with
        # sd_n = sqrt(2 * (n/3) / (n-1)). Slightly above sqrt(100) because
        # Bessel's correction inflates sd more at n=15.
        ratio = few["ci95_half_width_ms"] / many["ci95_half_width_ms"]
        expected = ((2 * (15 / 3) / 14) ** 0.5 / 15 ** 0.5) / \
                   ((2 * (1500 / 3) / 1499) ** 0.5 / 1500 ** 0.5)
        self.assertAlmostEqual(ratio, expected, places=6)
        self.assertAlmostEqual(ratio, 10.35, delta=0.05)

    def test_bootstrap_median_interval_brackets_the_median(self):
        values = [float(v) for v in range(1, 101)]
        st = stats_for_ms(values)
        self.assertIn("median_ci95_low_ms", st)
        self.assertLessEqual(st["median_ci95_low_ms"], st["median_ms"])
        self.assertGreaterEqual(st["median_ci95_high_ms"], st["median_ms"])

    def test_bootstrap_is_reproducible(self):
        values = [float(v) for v in range(1, 51)]
        first = stats_for_ms(values)
        second = stats_for_ms(values)
        self.assertEqual(first["median_ci95_low_ms"], second["median_ci95_low_ms"])
        self.assertEqual(first["median_ci95_high_ms"], second["median_ci95_high_ms"])


class TestDriftDetection(unittest.TestCase):

    def test_steady_samples_are_not_flagged(self):
        samples = [0.010, 0.0101, 0.0099, 0.010, 0.0102, 0.0098, 0.010, 0.0101]
        d = harness._drift_diagnostics(samples)
        self.assertLess(abs(d["drift_pct"]), harness.DRIFT_WARN_PCT)
        self.assertFalse(d["drift_flagged"])

    def test_a_slowdown_midway_is_flagged(self):
        # Second half 50% slower.
        samples = [0.010] * 50 + [0.015] * 50
        d = harness._drift_diagnostics(samples)
        self.assertAlmostEqual(d["drift_pct"], 50.0, places=6)
        self.assertTrue(d["drift_flagged"])

    def test_drift_survives_sorting_because_order_is_preserved(self):
        # summarize() sorts internally; drift must use the raw sample order, so
        # an in-place sort bug shows up here as zero drift.
        samples = [0.010] * 50 + [0.020] * 50
        d = harness._drift_diagnostics(samples)
        self.assertGreater(d["drift_pct"], 90.0)

    def test_too_few_samples_reports_no_drift(self):
        d = harness._drift_diagnostics([0.01, 0.02])
        self.assertIsNone(d["drift_pct"])
        self.assertFalse(d["drift_flagged"])

    def test_monotonic_ramp_is_caught_by_the_trend_test(self):
        """A linear ramp shows up as half its size in the halves comparison, so
        the Mann-Kendall trend test is what catches it."""
        ramp = [0.010 * (1 + 0.0001 * i) for i in range(1000)]  # +10% end to end
        d = harness._drift_diagnostics(ramp)
        self.assertLess(abs(d["drift_pct"]), 6.0)
        self.assertGreater(abs(d["trend_z"]), 1.96)
        self.assertTrue(d["drift_flagged"])

    def test_v_shape_cancels_in_the_halves_comparison(self):
        """Known blind spot: a disturbance symmetric about the midpoint cancels
        in the halves statistic and is non-monotonic, so neither check fires."""
        n = 1000
        v = [0.010 * (1 + 0.5 * abs(i - n / 2) / (n / 2)) for i in range(n)]
        d = harness._drift_diagnostics(v)
        self.assertLess(abs(d["drift_pct"]), 1.0)
        self.assertFalse(d["halves_flagged"])

    def test_autocorrelation_is_detected(self):
        import random

        # Independent draws give ACF near zero. An alternating series would not
        # work as the independent case: it is strongly negatively autocorrelated.
        rng = random.Random(11)
        independent = [rng.gauss(0.010, 2e-4) for _ in range(1000)]
        self.assertLess(abs(harness._lag1_autocorrelation(independent)), 0.15)

        # Dependent series: each sample close to the previous one.
        dependent, value = [], 0.010
        for i in range(1000):
            value += 1e-6 if i % 200 < 100 else -1e-6
            dependent.append(value)
        self.assertGreater(harness._lag1_autocorrelation(dependent), 0.9)


class TestDependenceAwareInterval(unittest.TestCase):

    def test_batch_means_interval_exceeds_iid_under_dependence(self):
        """Back-to-back timings are autocorrelated, so the iid s/sqrt(n)
        interval is too narrow; batch means should be wider."""
        import random

        rng = random.Random(20260730)
        series, value = [], 0.010
        for _ in range(1000):
            value = 0.9 * value + 0.1 * 0.010 + rng.gauss(0, 2e-4)
            series.append(value)

        iid = harness.summarize(series)["ci95_iid_half_width_ms"]
        batched = harness._batch_means_half_width_ms(series)
        self.assertIsNotNone(batched)
        self.assertGreater(batched, iid * 2)

    def test_independent_samples_give_comparable_intervals(self):
        import random

        rng = random.Random(7)
        series = [rng.gauss(0.010, 2e-4) for _ in range(1000)]
        iid = harness.summarize(series)["ci95_iid_half_width_ms"]
        batched = harness._batch_means_half_width_ms(series)
        # No dependence to correct for, so the two agree to within ~2x.
        self.assertLess(batched / iid, 2.0)

    def test_time_operation_reports_which_method_it_used(self):
        stats = harness.time_operation(lambda: None, repeats=200, warmup=5,
                                       keep_samples=False)
        self.assertIn("ci95_method", stats)
        self.assertIn("batch means", stats["ci95_method"])

    def test_short_run_falls_back_to_iid_and_says_so(self):
        stats = harness.time_operation(lambda: None, repeats=10, warmup=1,
                                       keep_samples=False)
        self.assertIn("iid", stats["ci95_method"])


class TestTimeOperation(unittest.TestCase):

    def test_warmup_calls_are_excluded_from_the_samples(self):
        calls = []
        stats = harness.time_operation(lambda: calls.append(1), repeats=10, warmup=7)
        self.assertEqual(len(calls), 17)          # 7 warm-up + 10 measured
        self.assertEqual(stats["repeats"], 10)
        self.assertEqual(len(stats["samples_ms"]), 10)
        self.assertEqual(stats["warmup"], 7)

    def test_samples_can_be_omitted(self):
        stats = harness.time_operation(lambda: None, repeats=5, warmup=0,
                                       keep_samples=False)
        self.assertNotIn("samples_ms", stats)

    def test_garbage_collection_is_restored_afterwards(self):
        self.assertTrue(gc_enabled := __import__("gc").isenabled())
        harness.time_operation(lambda: None, repeats=3, warmup=0)
        self.assertEqual(__import__("gc").isenabled(), gc_enabled)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            harness.time_operation(lambda: None, repeats=0)
        with self.assertRaises(ValueError):
            harness.time_operation(lambda: None, warmup=-1)

    def test_a_slower_function_measures_slower(self):
        def spin():
            total = 0
            for i in range(20000):
                total += i
            return total

        fast = harness.time_operation(lambda: None, repeats=50, warmup=5)
        slow = harness.time_operation(spin, repeats=50, warmup=5)
        self.assertGreater(slow["median_ms"], fast["median_ms"])


class TestMemoryMeasurement(unittest.TestCase):

    def test_tracemalloc_sees_a_python_allocation(self):
        kb = harness.measure_peak_memory_kb(lambda: [0] * 100_000, repeats=2)
        # A list of 100k ints is ~800 KB of pointers; check order of magnitude.
        self.assertGreater(kb, 100.0)

    def test_retained_bytes_tracks_a_known_allocation(self):
        # RSS growth per 1 MiB bytearray should be near 1 MiB.
        per_object = harness.measure_retained_bytes(lambda: bytearray(1024 * 1024),
                                                    count=32)
        if per_object is None:
            self.skipTest("/proc unavailable on this platform")
        self.assertGreater(per_object, 0.5 * 1024 * 1024)
        self.assertLess(per_object, 2.0 * 1024 * 1024)

    def test_retained_bytes_is_small_for_tiny_objects(self):
        per_object = harness.measure_retained_bytes(lambda: object(), count=50)
        if per_object is None:
            self.skipTest("/proc unavailable on this platform")
        self.assertLess(per_object, 100 * 1024)


class TestEnvironmentInfo(unittest.TestCase):

    def test_records_the_fields_the_report_cites(self):
        info = harness.environment_info()
        self.assertIn("python_version", info)
        self.assertIn("cpu_count", info)
        self.assertIn("thread_env", info)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            self.assertIn(var, info["thread_env"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
