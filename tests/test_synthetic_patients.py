"""
Tests for the synthetic cohort and the plaintext risk score.

Run with:  python -m unittest discover -p 'test_*.py'
"""

import os
import sys
import unittest

# Modules under test live in src/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))


import numpy as np

import synthetic_patients as patients


class TestNormalization(unittest.TestCase):

    def test_range_endpoints_map_to_zero_and_one(self):
        for name in patients.FEATURES:
            low, high = patients.RANGES[name]
            cohort = {n: np.array([patients.RANGES[n][0]]) for n in patients.FEATURES}
            cohort[name] = np.array([low, high])
            for other in patients.FEATURES:
                if other != name:
                    cohort[other] = np.array([patients.RANGES[other][0]] * 2)
            norm = patients.normalize_cohort(cohort)
            self.assertAlmostEqual(norm[name][0], 0.0, places=12, msg=name)
            self.assertAlmostEqual(norm[name][1], 1.0, places=12, msg=name)

    def test_affine_form_matches_direct_normalization(self):
        """scale/offset form must agree with (x-min)/(max-min)."""
        rng = np.random.default_rng(1)
        for name in patients.FEATURES:
            low, high = patients.RANGES[name]
            values = rng.uniform(low, high, size=200)
            scale, offset = patients.normalization_affine_terms(name)
            affine = values * scale + offset
            direct = (values - low) / (high - low)
            np.testing.assert_allclose(affine, direct, rtol=0, atol=1e-12)

    def test_weights_sum_to_one(self):
        total = (sum(patients.LINEAR_WEIGHTS.values())
                 + sum(patients.INTERACTION_WEIGHTS.values())
                 + sum(patients.SQUARE_WEIGHTS.values()))
        self.assertAlmostEqual(total, 1.0, places=12)


class TestRiskScore(unittest.TestCase):

    def test_hand_computed_patient(self):
        """Midpoint of every range: all normalized features are 0.5, so
        100 * (0.80*0.5 + 0.10*0.25 + 0.05*0.25 + 0.05*0.25) = 45.0.
        """
        cohort = {}
        for name in patients.FEATURES:
            low, high = patients.RANGES[name]
            cohort[name] = np.array([(low + high) / 2.0])
        score = patients.plaintext_risk_scores(cohort)[0]
        self.assertAlmostEqual(score, 45.0, places=10)

    def test_patient_at_the_top_of_every_range_scores_100(self):
        cohort = {n: np.array([patients.RANGES[n][1]]) for n in patients.FEATURES}
        self.assertAlmostEqual(patients.plaintext_risk_scores(cohort)[0], 100.0,
                               places=10)

    def test_patient_at_the_bottom_of_every_range_scores_zero(self):
        cohort = {n: np.array([patients.RANGES[n][0]]) for n in patients.FEATURES}
        self.assertAlmostEqual(patients.plaintext_risk_scores(cohort)[0], 0.0,
                               places=10)

    def test_score_is_term_by_term_what_the_formula_says(self):
        """Re-derive the score with the weights written out explicitly."""
        rng = np.random.default_rng(7)
        cohort = {}
        for name in patients.FEATURES:
            low, high = patients.RANGES[name]
            cohort[name] = rng.uniform(low, high, size=50)
        norm = patients.normalize_cohort(cohort)

        expected = (
            0.15 * norm["age"]
            + 0.20 * norm["systolic_bp"]
            + 0.15 * norm["bmi"]
            + 0.15 * norm["cholesterol"]
            + 0.15 * norm["glucose"]
            + 0.10 * norm["systolic_bp"] * norm["glucose"]
            + 0.05 * norm["bmi"] * norm["cholesterol"]
            + 0.05 * norm["systolic_bp"] ** 2
        ) * 100.0
        np.testing.assert_allclose(patients.plaintext_risk_scores(cohort), expected,
                                   rtol=0, atol=1e-12)

    def test_interaction_terms_actually_matter(self):
        """Two patients differing only in the product terms must score differently."""
        norm_a = {"age": 0.5, "systolic_bp": 0.9, "bmi": 0.5,
                  "cholesterol": 0.5, "glucose": 0.1}
        norm_b = {"age": 0.5, "systolic_bp": 0.1, "bmi": 0.5,
                  "cholesterol": 0.5, "glucose": 0.9}

        def to_raw(norm):
            return {n: np.array([patients.RANGES[n][0]
                                 + norm[n] * (patients.RANGES[n][1] - patients.RANGES[n][0])])
                    for n in patients.FEATURES}

        # bp and glucose have different linear weights (0.20 vs 0.15); the gap
        # must exceed that difference alone, thanks to the bp^2 term.
        score_a = patients.plaintext_risk_scores(to_raw(norm_a))[0]
        score_b = patients.plaintext_risk_scores(to_raw(norm_b))[0]
        linear_only_gap = 100 * (0.20 * 0.9 + 0.15 * 0.1) - 100 * (0.20 * 0.1 + 0.15 * 0.9)
        self.assertGreater(abs(score_a - score_b) - abs(linear_only_gap), 1.0)


class TestCohortGeneration(unittest.TestCase):

    def test_is_reproducible_for_a_given_seed(self):
        a = patients.generate_cohort(200, seed=123)
        b = patients.generate_cohort(200, seed=123)
        for name in patients.FEATURES:
            np.testing.assert_array_equal(a[name], b[name])

    def test_different_seeds_give_different_cohorts(self):
        a = patients.generate_cohort(200, seed=1)
        b = patients.generate_cohort(200, seed=2)
        self.assertFalse(np.array_equal(a["age"], b["age"]))

    def test_every_value_is_inside_its_public_range(self):
        """Values must be clipped, otherwise a normalized feature can leave [0, 1]."""
        cohort = patients.generate_cohort(5000, seed=99)
        for name in patients.FEATURES:
            low, high = patients.RANGES[name]
            self.assertGreaterEqual(cohort[name].min(), low, msg=name)
            self.assertLessEqual(cohort[name].max(), high, msg=name)

    def test_scores_stay_within_zero_to_one_hundred(self):
        cohort = patients.generate_cohort(5000, seed=99)
        scores = patients.plaintext_risk_scores(cohort)
        self.assertGreaterEqual(scores.min(), 0.0)
        self.assertLessEqual(scores.max(), 100.0)

    def test_requested_size_is_honoured(self):
        self.assertEqual(len(patients.generate_cohort(37, seed=5)["age"]), 37)

    def test_rejects_an_empty_cohort(self):
        with self.assertRaises(ValueError):
            patients.generate_cohort(0)

    def test_declared_depth_matches_the_term_structure(self):
        """The declared depth sets the crypto parameters, so keep it in sync."""
        depth = patients.multiplicative_depth_required()
        self.assertEqual(depth["total"], sum(v for k, v in depth.items() if k != "total"))
        # Ciphertext-by-ciphertext products need depth beyond constant scaling.
        self.assertTrue(patients.INTERACTION_WEIGHTS or patients.SQUARE_WEIGHTS)
        self.assertGreaterEqual(depth["total"], 3)


class TestPaddingInvariant(unittest.TestCase):

    def test_padding_with_the_range_minimum_contributes_nothing(self):
        """Padding to a power of two uses each feature's minimum, so the padded
        slots normalize to 0 and score 0 instead of corrupting the cohort sum.
        """
        import risk_score_benchmark as rsb

        cohort = patients.generate_cohort(100, seed=11)
        padded = rsb.pad_features(cohort, 128)

        scores = patients.plaintext_risk_scores(padded)
        np.testing.assert_allclose(scores[100:], 0.0, rtol=0, atol=1e-12)

        np.testing.assert_allclose(scores[:100],
                                   patients.plaintext_risk_scores(cohort),
                                   rtol=0, atol=1e-12)

    def test_padding_with_zeros_would_corrupt_the_sum(self):
        """Why the pad value is the range minimum and not zero."""
        cohort = patients.generate_cohort(100, seed=11)
        zero_padded = {n: np.concatenate([cohort[n], np.zeros(28)])
                       for n in patients.FEATURES}
        bad_scores = patients.plaintext_risk_scores(zero_padded)
        # Zero-padded slots normalize to -min/(max-min) and score far from zero.
        self.assertGreater(abs(bad_scores[100:]).max(), 1.0)

    def test_no_padding_needed_when_already_a_power_of_two(self):
        import risk_score_benchmark as rsb

        cohort = patients.generate_cohort(64, seed=3)
        padded = rsb.pad_features(cohort, 64)
        for name in patients.FEATURES:
            np.testing.assert_array_equal(padded[name], cohort[name])


if __name__ == "__main__":
    unittest.main(verbosity=2)
