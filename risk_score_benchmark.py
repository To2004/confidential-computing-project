"""
Encrypted evaluation of the synthetic medical risk score.

The score itself is defined in `synthetic_patients.py`, which also carries the
disclaimer: it is a synthetic construct for demonstration only and has no
clinical meaning.

Scenario
--------
A hospital holds a cohort of patient records with five measures per patient. It
wants a cloud provider to compute a risk score per patient and the cohort mean
score, without ever exposing an individual value. The hospital encrypts one
ciphertext per feature, ships them to the cloud, and the cloud evaluates the
whole score circuit on ciphertexts. Only the hospital can decrypt.

Packing
-------
One feature per ciphertext, one CKKS slot per patient. A single homomorphic
multiplication therefore advances all patients at once, which is what makes the
per-patient cost of this scenario tolerable. Vectors are padded to a power of two
because OpenFHE requires a power-of-two batch size; padding slots are filled with
the *low end of each feature's public range*, so their normalized value is
exactly 0 and they contribute nothing to the cohort sum. Masking them out
afterwards would otherwise cost an extra multiplicative level.

Circuit and depth
-----------------
    normalize        ct * scale + offset            (mult by constant)
    linear terms     norm * (100 * weight)          (mult by constant)
    interactions     norm_a * norm_b                (ct x ct)  <-- needs depth
    quadratic        norm_bp * norm_bp              (ct x ct)  <-- needs depth
    cohort mean      EvalSum(score) * (1 / N)       (rotations + constant)

The parameters used for the primitive-operation benchmark (TenSEAL
`[60, 40, 40, 60]`, OpenFHE depth 3) cannot run this circuit — see
`CRYPTO_PARAMETERS` below for what it takes and why.
"""

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import benchmark_harness as harness
import synthetic_patients as patients

# Matches the 1000 repetitions used for the primitive operations. A pipeline call
# costs far more than a single operation, so the warm-up is shorter here: the
# caches it exists to fill are already full after a handful of calls.
DEFAULT_PIPELINE_REPEATS = 1000
DEFAULT_PIPELINE_WARMUP = 20

# Deeper circuit than the primitive benchmark, so both libraries need a longer
# modulus chain, which in turn forces a larger ring dimension.
#
# These are the *tightest* parameters that run the whole circuit, established by
# sweeping both libraries (see section 7 of the report):
#   TenSEAL  3 usable levels -> "scale out of bounds"; 4 works. The 4-level chain
#            cannot fit in poly_modulus_degree 8192 at a 128-bit security level, so
#            the ring dimension has to double to 16384.
#   OpenFHE  depth 3 evaluates the score but fails on the cohort mean; depth 4
#            completes at ring dimension 16384. Depth 5 also works but pushes the
#            ring to 32768, roughly doubling the cost of every operation for
#            nothing — so depth 4 is deliberate, not incidental.
CRYPTO_PARAMETERS = {
    "tenseal": {
        "poly_modulus_degree": 16384,
        "coeff_mod_bit_sizes": [60, 40, 40, 40, 40, 60],
        "global_scale_bits": 40,
    },
    "openfhe": {
        "multiplicative_depth": 4,
        "scaling_mod_size": 50,
    },
}


def next_power_of_two(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def pad_features(cohort, n_slots):
    """Pad every feature column to `n_slots` with that feature's range minimum.

    A padded slot normalizes to 0, so it drops out of every score term and out
    of the cohort sum.
    """
    padded = {}
    for name in patients.FEATURES:
        values = np.asarray(cohort[name], dtype=float)
        low = patients.RANGES[name][0]
        pad_len = n_slots - len(values)
        padded[name] = np.concatenate([values, np.full(pad_len, low)]) if pad_len > 0 else values
    return padded


def scaled_score_weights():
    """Score weights with `SCORE_SCALE` folded in.

    Folding the factor of 100 into each term's weight avoids a final
    multiplication of the whole score by a constant, saving one level.
    """
    scale = patients.SCORE_SCALE
    return (
        {k: scale * w for k, w in patients.LINEAR_WEIGHTS.items()},
        {k: scale * w for k, w in patients.INTERACTION_WEIGHTS.items()},
        {k: scale * w for k, w in patients.SQUARE_WEIGHTS.items()},
    )


# ── TenSEAL backend ──────────────────────────────────────────────────────────

class TenSEALRiskScore:
    """Encrypted risk score using TenSEAL (CKKS)."""

    name = "TenSEAL"

    def __init__(self, cohort, n_patients):
        import tenseal as ts

        self.ts = ts
        self.n_patients = n_patients
        self.n_slots = next_power_of_two(n_patients)
        self.padded = pad_features(cohort, self.n_slots)
        cfg = CRYPTO_PARAMETERS["tenseal"]

        self.ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=cfg["poly_modulus_degree"],
            coeff_mod_bit_sizes=cfg["coeff_mod_bit_sizes"],
        )
        self.ctx.generate_galois_keys()
        self.ctx.global_scale = 2 ** cfg["global_scale_bits"]

    def encrypt(self):
        return {
            name: self.ts.ckks_vector(self.ctx, self.padded[name].tolist())
            for name in patients.FEATURES
        }

    def score(self, encrypted):
        """Cloud-side evaluation: normalization + all eight score terms."""
        linear_w, interaction_w, square_w = scaled_score_weights()

        # Normalization, as an affine map with public constants.
        norm = {}
        for name in patients.FEATURES:
            scale, offset = patients.normalization_affine_terms(name)
            norm[name] = encrypted[name] * scale + offset

        total = None
        for name, weight in linear_w.items():
            term = norm[name] * weight
            total = term if total is None else total + term
        for (first, second), weight in interaction_w.items():
            total = total + (norm[first] * norm[second]) * weight
        for name, weight in square_w.items():
            total = total + (norm[name] * norm[name]) * weight
        return total

    def cohort_mean(self, encrypted_score):
        return encrypted_score.sum() * (1.0 / self.n_patients)

    def decrypt_scores(self, encrypted_score):
        return np.asarray(encrypted_score.decrypt()[: self.n_patients])

    def decrypt_mean(self, encrypted_mean):
        return encrypted_mean.decrypt()[0]

    def ciphertext_bytes(self, encrypted):
        return len(encrypted[patients.FEATURES[0]].serialize())

    def parameter_summary(self):
        cfg = CRYPTO_PARAMETERS["tenseal"]
        return {
            "poly_modulus_degree": cfg["poly_modulus_degree"],
            "coeff_mod_bit_sizes": cfg["coeff_mod_bit_sizes"],
            "global_scale": f"2^{cfg['global_scale_bits']}",
            "usable_levels": len(cfg["coeff_mod_bit_sizes"]) - 2,
        }


# ── OpenFHE backend ──────────────────────────────────────────────────────────

class OpenFHERiskScore:
    """Encrypted risk score using OpenFHE (CKKS)."""

    name = "OpenFHE"

    def __init__(self, cohort, n_patients, context=None):
        """Build the backend, optionally over a caller-supplied crypto context.

        `context` is a `(crypto_context, key_pair)` pair. It exists so that
        `ind_cpad_flooding.py` can evaluate *this* circuit under noise-flooding
        parameters instead of reimplementing it — the security experiment and
        the performance experiment must not be allowed to drift apart.
        """
        self.n_patients = n_patients
        self.n_slots = next_power_of_two(n_patients)
        self.padded = pad_features(cohort, self.n_slots)

        if context is not None:
            self.cc, self.keys = context
        else:
            self.cc, self.keys = self.build_default_context(self.n_slots)

    @staticmethod
    def build_default_context(n_slots):
        """The crypto context used by the main risk-score benchmark."""
        from openfhe import (CCParamsCKKSRNS, GenCryptoContext,
                             HEStd_128_classic, PKESchemeFeature)

        cfg = CRYPTO_PARAMETERS["openfhe"]
        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(cfg["multiplicative_depth"])
        params.SetScalingModSize(cfg["scaling_mod_size"])
        params.SetBatchSize(n_slots)
        params.SetSecurityLevel(HEStd_128_classic)

        cc = GenCryptoContext(params)
        for feature in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH,
                        PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE):
            cc.Enable(feature)
        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalSumKeyGen(keys.secretKey)
        return cc, keys

    def encrypt(self):
        return {
            name: self.cc.Encrypt(
                self.keys.publicKey,
                self.cc.MakeCKKSPackedPlaintext(self.padded[name].tolist()),
            )
            for name in patients.FEATURES
        }

    def score(self, encrypted):
        linear_w, interaction_w, square_w = scaled_score_weights()
        cc = self.cc

        norm = {}
        for name in patients.FEATURES:
            scale, offset = patients.normalization_affine_terms(name)
            norm[name] = cc.EvalAdd(cc.EvalMult(encrypted[name], scale), offset)

        total = None
        for name, weight in linear_w.items():
            term = cc.EvalMult(norm[name], weight)
            total = term if total is None else cc.EvalAdd(total, term)
        for (first, second), weight in interaction_w.items():
            product = cc.EvalMult(norm[first], norm[second])
            total = cc.EvalAdd(total, cc.EvalMult(product, weight))
        for name, weight in square_w.items():
            product = cc.EvalMult(norm[name], norm[name])
            total = cc.EvalAdd(total, cc.EvalMult(product, weight))
        return total

    def cohort_mean(self, encrypted_score):
        summed = self.cc.EvalSum(encrypted_score, self.n_slots)
        return self.cc.EvalMult(summed, 1.0 / self.n_patients)

    def _decrypt(self, ciphertext, length):
        plain = self.cc.Decrypt(self.keys.secretKey, ciphertext)
        plain.SetLength(length)
        return plain.GetRealPackedValue()

    def decrypt_scores(self, encrypted_score):
        return np.asarray(self._decrypt(encrypted_score, self.n_patients))

    def decrypt_mean(self, encrypted_mean):
        return self._decrypt(encrypted_mean, 1)[0]

    def ciphertext_bytes(self, encrypted):
        """Serialized size of one encrypted feature.

        OpenFHE serializes to a file rather than returning bytes, so this writes
        to a temporary file and measures it.
        """
        import os
        import tempfile

        from openfhe import BINARY, SerializeToFile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "feature.bin")
            first = encrypted[patients.FEATURES[0]]
            return os.path.getsize(path) if SerializeToFile(path, first, BINARY) else None

    def parameter_summary(self):
        cfg = CRYPTO_PARAMETERS["openfhe"]
        return {
            "multiplicative_depth": cfg["multiplicative_depth"],
            "scaling_mod_size": cfg["scaling_mod_size"],
            "batch_size": self.n_slots,
            "ring_dimension": self.cc.GetRingDimension(),
        }


# ── Benchmark driver ─────────────────────────────────────────────────────────

def benchmark_backend(backend, reference_scores, repeats, warmup, memory_repeats):
    """Time each stage of the pipeline and check accuracy against plaintext."""
    print(f"\n[{backend.name}] parameters: {backend.parameter_summary()}")

    encrypted = backend.encrypt()
    encrypted_score = backend.score(encrypted)
    encrypted_mean = backend.cohort_mean(encrypted_score)

    stages = {
        "encrypt_features": lambda: backend.encrypt(),
        "score_evaluation": lambda: backend.score(encrypted),
        "cohort_mean": lambda: backend.cohort_mean(encrypted_score),
        "decrypt_scores": lambda: backend.decrypt_scores(encrypted_score),
    }
    results = harness.benchmark_operations(
        stages, repeats=repeats, warmup=warmup,
        memory_repeats=memory_repeats, label=backend.name,
    )

    # Accuracy of the decrypted result against the plaintext reference.
    decrypted_scores = backend.decrypt_scores(encrypted_score)
    decrypted_mean = backend.decrypt_mean(encrypted_mean)
    reference_mean = float(np.mean(reference_scores))
    errors = np.abs(decrypted_scores - reference_scores)

    results["score_max_abs_error"] = float(errors.max())
    results["score_mean_abs_error"] = float(errors.mean())
    results["score_max_rel_error"] = float(
        (errors / np.maximum(np.abs(reference_scores), 1e-12)).max()
    )
    results["cohort_mean_encrypted"] = float(decrypted_mean)
    results["cohort_mean_plaintext"] = reference_mean
    results["cohort_mean_abs_error"] = float(abs(decrypted_mean - reference_mean))
    results["pipeline_total_ms"] = sum(
        results[f"{stage}_time_stats"]["mean_ms"] for stage in stages
    )
    results["parameters"] = backend.parameter_summary()

    ct_bytes = backend.ciphertext_bytes(encrypted)
    if ct_bytes is not None:
        results["ciphertext_bytes_per_feature"] = ct_bytes

    # Fail loudly rather than reporting timings for a circuit that silently ran
    # out of levels. A relative error above 0.1% on a 0-100 score would mean the
    # parameters are wrong, not that CKKS was noisy.
    harness.assert_accuracy(
        results,
        {"score_max_rel_error": 1e-3, "cohort_mean_abs_error": 1e-1},
        label=backend.name,
    )

    return results


def plaintext_pipeline_stats(cohort, repeats, warmup, memory_repeats):
    """The same score computed in the clear, as the overhead reference."""
    stages = {
        "score_evaluation": lambda: patients.plaintext_risk_scores(cohort),
        "cohort_mean": lambda: float(np.mean(patients.plaintext_risk_scores(cohort))),
    }
    results = harness.benchmark_operations(
        stages, repeats=repeats, warmup=warmup,
        memory_repeats=memory_repeats, label="Plaintext",
    )
    results["pipeline_total_ms"] = results["score_evaluation_time_stats"]["mean_ms"]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--patients", type=int, default=patients.DEFAULT_N_PATIENTS,
                        help=f"cohort size (default {patients.DEFAULT_N_PATIENTS})")
    parser.add_argument("--seed", type=int, default=patients.DEFAULT_SEED,
                        help="RNG seed for the synthetic cohort")
    parser.add_argument("--repeats", type=int, default=DEFAULT_PIPELINE_REPEATS,
                        help=f"timed repetitions per stage (default {DEFAULT_PIPELINE_REPEATS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_PIPELINE_WARMUP,
                        help=f"discarded warm-up calls (default {DEFAULT_PIPELINE_WARMUP})")
    parser.add_argument("--memory-repeats", type=int, default=3,
                        help="repetitions for the memory pass (default 3)")
    parser.add_argument("--skip-openfhe", action="store_true",
                        help="run only the plaintext and TenSEAL versions")
    parser.add_argument("--output", default="risk_score_results.json",
                        help="where to write the JSON results")
    args = parser.parse_args()

    print("=" * 88)
    print(f"{'Encrypted Synthetic Medical Risk Score':^88}")
    print("=" * 88)
    print(patients.DISCLAIMER)
    print()

    cohort = patients.generate_cohort(args.patients, seed=args.seed)
    reference_scores = patients.plaintext_risk_scores(cohort)
    print(patients.describe_cohort(cohort, reference_scores))

    print(f"\nTiming: {args.repeats} repetitions per stage, "
          f"{args.warmup} discarded warm-up calls.")

    print("\nPlaintext reference:")
    plaintext = plaintext_pipeline_stats(cohort, args.repeats, args.warmup,
                                        args.memory_repeats)

    backends = [TenSEALRiskScore(cohort, args.patients)]
    if not args.skip_openfhe:
        backends.append(OpenFHERiskScore(cohort, args.patients))

    measured = {}
    for backend in backends:
        measured[backend.name.lower()] = benchmark_backend(
            backend, reference_scores, args.repeats, args.warmup, args.memory_repeats
        )

    print_summary(plaintext, measured, args.patients)

    output = {
        "n_patients": args.patients,
        "seed": args.seed,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "disclaimer": patients.DISCLAIMER,
        "plaintext_mean_risk_score": float(np.mean(reference_scores)),
        "depth_required": patients.multiplicative_depth_required(),
        "plaintext": plaintext,
        **measured,
    }
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"\nResults saved to {args.output}")


def print_summary(plaintext, measured, n_patients):
    stages = [
        ("Encrypt features", "encrypt_features"),
        ("Score evaluation", "score_evaluation"),
        ("Cohort mean", "cohort_mean"),
        ("Decrypt scores", "decrypt_scores"),
    ]
    names = list(measured.keys())

    print()
    print("=" * 88)
    print(f"{'Risk Score Pipeline — mean time per stage (ms)':^88}")
    print("=" * 88)
    header = f"{'Stage':<20}{'Plaintext':>14}" + "".join(f"{n.capitalize():>16}" for n in names)
    print(header)
    print("-" * 88)
    for label, key in stages:
        cells = f"{label:<20}"
        pt_stats = plaintext.get(f"{key}_time_stats")
        cells += f"{pt_stats['mean_ms']:>14.4f}" if pt_stats else f"{'—':>14}"
        for n in names:
            st = measured[n].get(f"{key}_time_stats")
            cells += f"{st['mean_ms']:>16.3f}" if st else f"{'—':>16}"
        print(cells)
    print("-" * 88)

    pt_total = plaintext["pipeline_total_ms"]
    totals = f"{'Pipeline total':<20}{pt_total:>14.4f}"
    for n in names:
        totals += f"{measured[n]['pipeline_total_ms']:>16.3f}"
    print(totals)

    overhead = f"{'Overhead vs plain':<20}{'1x':>14}"
    for n in names:
        overhead += f"{measured[n]['pipeline_total_ms'] / pt_total:>15,.0f}x"
    print(overhead)

    per_patient = f"{'Per patient (ms)':<20}{pt_total / n_patients:>14.6f}"
    for n in names:
        per_patient += f"{measured[n]['pipeline_total_ms'] / n_patients:>16.4f}"
    print(per_patient)

    print()
    print("=" * 88)
    print(f"{'Risk Score Accuracy vs Plaintext Reference':^88}")
    print("=" * 88)
    print(f"{'Metric':<34}" + "".join(f"{n.capitalize():>18}" for n in names))
    print("-" * 88)
    for label, key, fmt in [
        ("Max abs error (score points)", "score_max_abs_error", "{:>18.4e}"),
        ("Mean abs error (score points)", "score_mean_abs_error", "{:>18.4e}"),
        ("Max relative error", "score_max_rel_error", "{:>18.4e}"),
        ("Cohort mean, encrypted", "cohort_mean_encrypted", "{:>18.6f}"),
        ("Cohort mean, plaintext", "cohort_mean_plaintext", "{:>18.6f}"),
        ("Cohort mean abs error", "cohort_mean_abs_error", "{:>18.4e}"),
    ]:
        row = f"{label:<34}"
        for n in names:
            row += fmt.format(measured[n][key])
        print(row)
    print("-" * 88)


if __name__ == "__main__":
    main()
