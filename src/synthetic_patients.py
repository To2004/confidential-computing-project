"""
Synthetic patient cohort and the synthetic medical risk score.

IMPORTANT — this risk score is a synthetic construct built for demonstration
purposes only. It is NOT a validated clinical model, it is not derived from any
medical literature, and it must not be interpreted as a real assessment of
patient risk. Its only purpose is to give the homomorphic-encryption benchmark a
computation that is richer than a single average: it mixes additions,
multiplications by public constants, multiplications between two encrypted
values, and a squared term, so it exercises real multiplicative depth.

The cohort is likewise synthetic: values are drawn from plausible-looking
distributions, not sampled from patients.

Score definition
----------------
Each of the five features is first mapped to roughly [0, 1] by min-max
normalization against a *public* reference range:

    x_norm = (x - min_value) / (max_value - min_value)

The score then combines the normalized features:

    risk_score = 100 * ( 0.15 * age_norm
                       + 0.20 * bp_norm
                       + 0.15 * bmi_norm
                       + 0.15 * cholesterol_norm
                       + 0.15 * glucose_norm
                       + 0.10 * bp_norm * glucose_norm
                       + 0.05 * bmi_norm * cholesterol_norm
                       + 0.05 * bp_norm ** 2 )

The first five terms are a weighted average. The last three are what make the
score interesting for HE: two interaction terms between distinct features and
one quadratic term, all of which require ciphertext x ciphertext multiplication
rather than only multiplication by a plaintext constant. The eight weights sum
to 1.0, so a patient at the top of every reference range scores 100.

Why the reference ranges are public
-----------------------------------
The min and max used for normalization are fixed published ranges, not
statistics computed from the cohort. This matters for the encrypted version:
deriving min/max from the data would require comparisons between ciphertexts,
which CKKS does not support natively (it would need an expensive polynomial
approximation of a step function). Using public ranges keeps normalization an
affine transformation, which CKKS performs cheaply.
"""

import numpy as np

# Feature order is fixed: it defines the order of ciphertexts everywhere.
FEATURES = ("age", "systolic_bp", "bmi", "cholesterol", "glucose")

# Public reference ranges (min, max) used for min-max normalization.
RANGES = {
    "age": (18.0, 90.0),
    "systolic_bp": (90.0, 180.0),
    "bmi": (18.0, 40.0),
    "cholesterol": (120.0, 280.0),
    "glucose": (70.0, 200.0),
}

# Distribution used by the generator: age is uniform over its range, the four
# clinical measures are normal around a plausible centre and then clipped.
DISTRIBUTIONS = {
    "age": {"kind": "uniform"},
    "systolic_bp": {"kind": "normal", "mean": 125.0, "sd": 15.0},
    "bmi": {"kind": "normal", "mean": 27.0, "sd": 4.5},
    "cholesterol": {"kind": "normal", "mean": 200.0, "sd": 35.0},
    "glucose": {"kind": "normal", "mean": 100.0, "sd": 20.0},
}

# Weights of the eight score terms. Linear terms first, then the two
# interactions, then the quadratic term. They sum to 1.0.
LINEAR_WEIGHTS = {
    "age": 0.15,
    "systolic_bp": 0.20,
    "bmi": 0.15,
    "cholesterol": 0.15,
    "glucose": 0.15,
}
INTERACTION_WEIGHTS = {
    ("systolic_bp", "glucose"): 0.10,
    ("bmi", "cholesterol"): 0.05,
}
SQUARE_WEIGHTS = {"systolic_bp": 0.05}

SCORE_SCALE = 100.0

DEFAULT_N_PATIENTS = 1000
DEFAULT_SEED = 20260730

DISCLAIMER = (
    "The risk score below is SYNTHETIC and for demonstration only. "
    "It is not a validated clinical model and carries no medical meaning."
)


def generate_cohort(n_patients=DEFAULT_N_PATIENTS, seed=DEFAULT_SEED):
    """Generate a synthetic cohort as a column-per-feature dict of arrays.

    Column-major layout is deliberate: the encrypted version packs one feature
    into one ciphertext, with one CKKS slot per patient, so that a single
    homomorphic operation advances all patients at once (SIMD).
    """
    if n_patients < 1:
        raise ValueError(f"n_patients must be >= 1, got {n_patients}")

    rng = np.random.default_rng(seed)
    cohort = {}
    for name in FEATURES:
        low, high = RANGES[name]
        spec = DISTRIBUTIONS[name]
        if spec["kind"] == "uniform":
            values = rng.uniform(low, high, size=n_patients)
        else:
            values = rng.normal(spec["mean"], spec["sd"], size=n_patients)
        # Clip so every value stays inside the public reference range, which
        # keeps every normalized feature inside [0, 1].
        cohort[name] = np.clip(values, low, high)
    return cohort


def normalize_cohort(cohort):
    """Min-max normalize every feature against its public reference range."""
    normalized = {}
    for name in FEATURES:
        low, high = RANGES[name]
        normalized[name] = (np.asarray(cohort[name], dtype=float) - low) / (high - low)
    return normalized


def normalization_affine_terms(name):
    """Return (scale, offset) such that x_norm == x * scale + offset.

    The encrypted implementation applies normalization in this affine form,
    because CKKS multiplies by a plaintext constant and adds a plaintext
    constant far more cheaply than it divides.
    """
    low, high = RANGES[name]
    scale = 1.0 / (high - low)
    return scale, -low / (high - low)


def plaintext_risk_scores(cohort):
    """Reference risk score per patient, computed in the clear with NumPy."""
    norm = normalize_cohort(cohort)

    total = np.zeros_like(norm[FEATURES[0]])
    for name, weight in LINEAR_WEIGHTS.items():
        total = total + weight * norm[name]
    for (first, second), weight in INTERACTION_WEIGHTS.items():
        total = total + weight * norm[first] * norm[second]
    for name, weight in SQUARE_WEIGHTS.items():
        total = total + weight * norm[name] * norm[name]

    return SCORE_SCALE * total


def multiplicative_depth_required():
    """Multiplicative depth consumed by the encrypted score, per stage.

    Both libraries must be configured for at least this depth or the ciphertext
    runs out of levels partway through the circuit.
    """
    return {
        "normalization (x * scale + offset)": 1,
        "interaction / square (ct x ct)": 1,
        "term weight (x plaintext constant)": 1,
        "cohort mean (x 1/N after summation)": 1,
        "total": 4,
    }


def describe_cohort(cohort, scores=None):
    """Human-readable summary of the generated cohort."""
    lines = [
        "Synthetic patient cohort",
        "-" * 72,
        f"!! {DISCLAIMER}",
        "-" * 72,
        f"Patients: {len(cohort[FEATURES[0]])}",
        "",
        f"{'Feature':<14}{'Range (public)':>18}{'Mean':>10}{'SD':>10}{'Min':>10}{'Max':>10}",
    ]
    for name in FEATURES:
        values = np.asarray(cohort[name], dtype=float)
        low, high = RANGES[name]
        lines.append(
            f"{name:<14}{f'[{low:g}, {high:g}]':>18}{values.mean():>10.2f}"
            f"{values.std():>10.2f}{values.min():>10.2f}{values.max():>10.2f}"
        )

    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        lines += [
            "",
            f"{'Synthetic risk score':<14} mean {scores.mean():.4f}  sd {scores.std():.4f}  "
            f"min {scores.min():.4f}  max {scores.max():.4f}",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    cohort = generate_cohort()
    scores = plaintext_risk_scores(cohort)
    print(describe_cohort(cohort, scores))
    print()
    print("Multiplicative depth required by the encrypted score:")
    for stage, depth in multiplicative_depth_required().items():
        print(f"  {stage:<38} {depth}")
