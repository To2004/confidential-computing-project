"""
Tests for the measurement harness statistics.

The statistics in `benchmark_harness` decide what the report claims, so they are
checked against values worked out by hand rather than trusted. Run with:

    python -m unittest discover -p 'test_*.py'
"""

import unittest

import benchmark_harness as harness


def stats_for_ms(values_ms, **kwargs):
    """Summarize a list expressed in milliseconds (the harness takes seconds)."""
    return harness.summarize([v / 1000.0 for v in values_ms], **kwargs)


class TestSummarize(unittest.TestCase):

    def test_known_values(self):
        # 1..9 ms: mean 5, median 5, population-corrected sd of 1..9 is 2.7386.
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
        # One sample in a hundred at 100x drags the mean up by ~10%.
        self.assertAlmostEqual(st["mean_ms"], 19.9, places=6)
        # The median and the trimmed mean do not move at all.
        self.assertAlmostEqual(st["median_ms"], 10.0, places=9)
        self.assertAlmostEqual(st["trimmed_mean_ms"], 10.0, places=9)
        self.assertAlmostEqual(st["mad_ms"], 0.0, places=9)
        # And the skew is reported rather than hidden.
        self.assertGreater(st["mean_over_median_pct"], 90.0)

    def test_percentiles_interpolate(self):
        st = stats_for_ms([float(v) for v in range(1, 101)], bootstrap=False)
        # For 1..100 the p95 by linear interpolation on ranks is 95.05.
        self.assertAlmostEqual(st["p95_ms"], 95.05, places=6)
        self.assertAlmostEqual(st["p99_ms"], 99.01, places=6)
        # IQR of 1..100 is 75.25 - 25.75 = 49.5
        self.assertAlmostEqual(st["iqr_ms"], 49.5, places=6)

    def test_outlier_fraction_counts_only_the_upper_tail(self):
        # A tight body plus five very large samples.
        values = [10.0] * 95 + [500.0] * 5
        st = stats_for_ms(values, bootstrap=False)
        self.assertAlmostEqual(st["outlier_fraction_pct"], 5.0, places=6)

        # A symmetric spread with no extreme tail flags nothing.
        symmetric = [float(v) for v in range(1, 101)]
        self.assertEqual(stats_for_ms(symmetric, bootstrap=False)["outlier_fraction_pct"], 0.0)

    def test_confidence_interval_shrinks_with_sample_count(self):
        few = stats_for_ms([9.0, 10.0, 11.0] * 5, bootstrap=False)      # n = 15
        many = stats_for_ms([9.0, 10.0, 11.0] * 500, bootstrap=False)   # n = 1500
        self.assertLess(many["ci95_half_width_ms"], few["ci95_half_width_ms"])

        # Same underlying spread, 100x the samples, so the interval narrows by
        # roughly sqrt(100) = 10. It is slightly more than 10 because the sample
        # standard deviation carries Bessel's correction: dividing by n-1 inflates
        # the estimate more at n=15 than at n=1500, so the numerator shrinks too.
        # The exact expected value is
        #     (sd_15 / sqrt(15)) / (sd_1500 / sqrt(1500))
        # with sd_n = sqrt(2 * (n/3) / (n-1)), which comes to 10.3475.
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
        # Second half 50% slower: exactly the shape of competing machine load.
        samples = [0.010] * 50 + [0.015] * 50
        d = harness._drift_diagnostics(samples)
        self.assertAlmostEqual(d["drift_pct"], 50.0, places=6)
        self.assertTrue(d["drift_flagged"])

    def test_drift_survives_sorting_because_order_is_preserved(self):
        # summarize() sorts internally; drift must be computed on the raw order,
        # so a sorted-in-place bug would show up as zero drift here.
        samples = [0.010] * 50 + [0.020] * 50
        d = harness._drift_diagnostics(samples)
        self.assertGreater(d["drift_pct"], 90.0)

    def test_too_few_samples_reports_no_drift(self):
        d = harness._drift_diagnostics([0.01, 0.02])
        self.assertIsNone(d["drift_pct"])
        self.assertFalse(d["drift_flagged"])


class TestTimeOperation(unittest.TestCase):

    def test_warmup_calls_are_excluded_from_the_samples(self):
        calls = []
        stats = harness.time_operation(lambda: calls.append(1), repeats=10, warmup=7)
        self.assertEqual(len(calls), 17)          # warm-up really ran
        self.assertEqual(stats["repeats"], 10)    # but is not in the statistics
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
        # A list of 100k ints is ~800 KB of pointers; assert the order of magnitude.
        self.assertGreater(kb, 100.0)

    def test_retained_bytes_tracks_a_known_allocation(self):
        # 1 MiB bytearrays: RSS growth per object should be near 1 MiB.
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
