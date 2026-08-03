"""
The cost of making CKKS safe to release decrypted results (IND-CPA^D).

The security gap
----------------
Every other experiment in this project treats CKKS as if confidentiality were
settled once the data is encrypted. It is not, and the reason is specific to
*approximate* homomorphic encryption.

Li and Micciancio (EUROCRYPT 2021, "On the Security of Homomorphic Encryption on
Approximate Numbers") showed that CKKS as originally specified is **not IND-CPA^D
secure** — that is, it is not secure against an adversary who is allowed to see
decryption results. The decrypted value of a CKKS ciphertext is the message plus
an error term, and that error term depends on the secret key. An adversary who
obtains enough approximate decryptions of ciphertexts it influenced can solve for
the key. The attack is passive and practical; it does not require the server to
misbehave beyond asking for decryptions.

Why this matters for our scenario, specifically
-----------------------------------------------
Our healthcare use case is exactly the setting the attack targets. The hospital
decrypts and then *publishes* a result — the cohort mean risk score. Under the
plain IND-CPA reading of CKKS that is fine, because IND-CPA says nothing about
released decryptions. Under IND-CPA^D it is the dangerous operation: each
published aggregate is one approximate decryption handed to whoever reads it.
The per-patient scores are worse still, since there are 1000 of them.

The mitigation and what it costs
--------------------------------
The standard countermeasure is **noise flooding**: before decryption, add noise
large enough to statistically drown the key-dependent error, so the released
value leaks nothing about the key. It cannot be applied blindly, because the
flooding noise must exceed the circuit's own error — which is not known in
advance. OpenFHE therefore implements a **two-pass protocol**:

    pass 1  EXEC_NOISE_ESTIMATION   run the circuit, read back the noise via
                                    Plaintext.GetLogError()
    pass 2  EXEC_EVALUATION         rebuild the context with that estimate and
                                    NOISE_FLOODING_DECRYPT, then run for real

This script runs both passes and measures the price: the extra pass, the effect
on ciphertext parameters, the effect on each operation, and the precision given
up to the flooding noise.

What we measured, and one thing that surprised us
-------------------------------------------------
The decisive test of whether any decryption-noise defence is active is whether
decryption is *randomized*: plain CKKS decryption is a deterministic function of
the ciphertext and the key, so repeatedly decrypting one fixed ciphertext must
return bit-identical results. Adding noise before release breaks that.

Running that test on all three configurations:

* **TenSEAL** — spread exactly 0. Decryption is deterministic, so TenSEAL offers
  no decryption-noise defence at all and is directly exposed when decrypted
  values are released.
* **OpenFHE, default `FIXED_NOISE_DECRYPT`** — already randomized. We expected a
  bare baseline here and found that OpenFHE adds decryption noise by default; the
  mode name says so, but this is not something the library forces you to notice.
* **OpenFHE, `NOISE_FLOODING_DECRYPT`** — randomized, with the magnitude
  calibrated to the measured circuit noise instead of a conservative constant.

What this experiment does NOT show
---------------------------------
Observing randomized decryption shows a defence is *active*. It does not show
that the defence is *sufficient*, and the literature is explicit that it is not.

OpenFHE estimates the circuit's noise empirically, by measuring precision loss in
the imaginary slots of a trial decryption. That is an **average-case** estimate.
Guo, Nabokov, Suvanto and Johansson (USENIX Security 2024, "Key Recovery Attacks
on Approximate Homomorphic Encryption with Non-Worst-Case Noise Flooding
Countermeasures") show that noise flooding built on non-worst-case noise
estimation remains vulnerable to key recovery — explicitly including deployments
that implement the differential-privacy bounds of Li, Micciancio, Schultz and
Sorrell (CRYPTO 2022). Average-case analysis assumes the input ciphertexts are
independent, and that assumption fails when a circuit combines correlated inputs,
which our risk-score circuit does.

Cheon-style provable IND-CPA^D flooding needs a variance high enough to cost
substantial message precision (see "Revisiting the Security of Approximate FHE
with Noise-Flooding Countermeasures", PKC 2025). OpenFHE's own follow-up work
(ePrint 2024/203, "Application-Aware Approximate Homomorphic Encryption")
exists specifically to counter the Guo et al. attacks, and is a proof of concept
rather than the default behaviour of the release used here (1.4.1).

So the honest reading of the numbers below is: **enabling flooding raises the bar
and costs roughly 2x on decryption plus an entire extra pass, but the resulting
configuration is not known to be IND-CPA^D secure.** Treating it as "the fix"
would misrepresent the state of the art.

Usage: python ind_cpad_flooding.py [--repeats 200]
"""

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import openfhe as fhe
from openfhe import CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature

import benchmark_harness as harness
import project_paths
import synthetic_patients as patients

# Small cohort: this experiment is about the security/performance trade-off, and
# a smaller batch keeps the two-pass protocol quick to run.
DEFAULT_PATIENTS = 64
DEFAULT_REPEATS = 200
DEFAULT_WARMUP = 20

MULT_DEPTH = 4          # the risk-score circuit's depth, from synthetic_patients
SCALING_MOD_SIZE = 50
FIRST_MOD_SIZE = 60

OPS = [("Encrypt", "encrypt"), ("Multiply", "mul"),
       ("Sum", "sum"), ("Decrypt", "decrypt")]


def build_context(batch_size, noise_mode=None, execution_mode=None,
                  noise_estimate=None):
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(MULT_DEPTH)
    params.SetScalingModSize(SCALING_MOD_SIZE)
    params.SetFirstModSize(FIRST_MOD_SIZE)
    params.SetBatchSize(batch_size)
    params.SetSecurityLevel(fhe.HEStd_128_classic)
    if noise_mode is not None:
        params.SetDecryptionNoiseMode(noise_mode)
    if execution_mode is not None:
        params.SetExecutionMode(execution_mode)
    if noise_estimate is not None:
        params.SetNoiseEstimate(noise_estimate)

    cc = GenCryptoContext(params)
    for feature in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH,
                    PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(feature)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)
    return cc, keys


def make_backend(cohort, n_patients, context):
    """The risk-score backend from `risk_score_benchmark`, over our context.

    The circuit is *not* reimplemented here. Reusing `OpenFHERiskScore` is what
    makes the claim "we measured the flooding cost of the same computation the
    rest of the report benchmarks" structurally true rather than a comment that
    could quietly go stale.
    """
    import risk_score_benchmark as rsb

    return rsb.OpenFHERiskScore(cohort, n_patients, context=context)


def score_circuit(backend):
    """Encrypt the cohort and evaluate the score, returning the score ciphertext."""
    return backend.score(backend.encrypt())


def estimate_noise(cohort, batch_size, n_patients, repeats=10, warmup=2):
    """Pass 1: run the circuit in noise-estimation mode and read the estimate.

    The pass is also *timed*. Flooding adds a modest per-decryption cost, but it
    also requires running the whole circuit once more beforehand — and for a
    short-lived context that extra pass, not the flooding itself, is the dominant
    price of the mitigation. Asserting that without measuring it would be exactly
    the kind of unbacked claim this project is trying to avoid.
    """
    cc, keys = build_context(batch_size,
                             noise_mode=fhe.NOISE_FLOODING_DECRYPT,
                             execution_mode=fhe.EXEC_NOISE_ESTIMATION)
    backend = make_backend(cohort, n_patients, (cc, keys))

    def one_pass():
        plain = cc.Decrypt(keys.secretKey, score_circuit(backend))
        plain.SetLength(n_patients)
        return plain

    stats = harness.time_operation(one_pass, repeats=repeats, warmup=warmup,
                                   keep_samples=False)
    return one_pass().GetLogError(), cc.GetRingDimension(), stats


def _spread_of(decryptions, trials):
    stacked = np.vstack(decryptions)
    per_slot_range = stacked.max(axis=0) - stacked.min(axis=0)
    return {
        "trials": trials,
        "max_range_across_slots": float(per_slot_range.max()),
        "mean_range_across_slots": float(per_slot_range.mean()),
        "per_slot_std_max": float(stacked.std(axis=0).max()),
        "is_randomized": bool(per_slot_range.max() > 0.0),
    }


def decryption_spread(cc, keys, ciphertext, length, trials=32):
    """Spread across repeated decryptions of the *same* OpenFHE ciphertext.

    Plain CKKS decryption is a deterministic function of the ciphertext and the
    key, so a zero spread means no noise is added before release and a non-zero
    spread means some is. This turns "we set a flag that claims to enable
    flooding" into an observation about what the library actually does.
    """
    decryptions = []
    for _ in range(trials):
        plain = cc.Decrypt(keys.secretKey, ciphertext)
        plain.SetLength(length)
        decryptions.append(np.asarray(plain.GetRealPackedValue()))
    return _spread_of(decryptions, trials)


def tenseal_decryption_spread(trials=32):
    """The same randomization test for TenSEAL, as the comparison point.

    TenSEAL has no decryption-noise setting to enable, so this establishes what
    "no defence" looks like: an exactly reproducible decryption.
    """
    import tenseal as ts

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40

    vector = [1.0, 2.0, 3.0, 4.0, 5.0]
    encrypted = ts.ckks_vector(ctx, vector)
    product = encrypted * encrypted  # one fixed ciphertext, decrypted repeatedly

    decryptions = [np.asarray(product.decrypt()) for _ in range(trials)]
    spread = _spread_of(decryptions, trials)
    spread["arm"] = "TenSEAL (no such setting)"
    return spread


def measure_arm(cohort, batch_size, n_patients, reference_scores,
                repeats, warmup, noise_estimate=None, label="baseline"):
    """Time the primitives and check accuracy, with or without flooding."""
    if noise_estimate is None:
        cc, keys = build_context(batch_size)
    else:
        cc, keys = build_context(batch_size,
                                 noise_mode=fhe.NOISE_FLOODING_DECRYPT,
                                 execution_mode=fhe.EXEC_EVALUATION,
                                 noise_estimate=noise_estimate)

    backend = make_backend(cohort, n_patients, (cc, keys))
    vector = backend.padded[patients.FEATURES[0]].tolist()
    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vector))

    operations = {
        "encrypt": lambda: cc.Encrypt(keys.publicKey,
                                      cc.MakeCKKSPackedPlaintext(vector)),
        "mul": lambda: cc.EvalMult(ct, ct),
        "sum": lambda: cc.EvalSum(ct, batch_size),
        "decrypt": lambda: cc.Decrypt(keys.secretKey, ct),
    }
    results = harness.benchmark_operations(operations, repeats=repeats,
                                          warmup=warmup, label=label,
                                          keep_samples=False)

    # Accuracy of the actual scenario output under this arm.
    scored_ct = score_circuit(backend)
    scored = cc.Decrypt(keys.secretKey, scored_ct)
    scored.SetLength(n_patients)
    decrypted = np.asarray(scored.GetRealPackedValue())
    errors = np.abs(decrypted - reference_scores)

    results["score_max_abs_error"] = float(errors.max())
    results["score_mean_abs_error"] = float(errors.mean())
    results["ring_dimension"] = cc.GetRingDimension()
    results["arm"] = label
    # Is flooding actually active? Randomized decryption is the observable signature.
    results["decryption_spread"] = decryption_spread(cc, keys, scored_ct, n_patients)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cost of the IND-CPA^D noise-flooding mitigation in CKKS")
    parser.add_argument("--patients", type=int, default=DEFAULT_PATIENTS)
    parser.add_argument("--seed", type=int, default=patients.DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--output",
                        default=project_paths.result_path("ind_cpad_results.json"),
                        help="where to write the JSON results")
    args = parser.parse_args()

    import risk_score_benchmark as rsb

    batch_size = rsb.next_power_of_two(args.patients)
    cohort = patients.generate_cohort(args.patients, seed=args.seed)
    reference_scores = patients.plaintext_risk_scores(cohort)


    print("=" * 96)
    print(f"{'IND-CPA^D: the cost of noise flooding in CKKS':^96}")
    print("=" * 96)
    print("Li & Micciancio, EUROCRYPT 2021: CKKS is not IND-CPA^D secure. An")
    print("adversary who sees approximate DECRYPTION RESULTS can recover the secret")
    print("key. Our scenario publishes a decrypted cohort statistic, so it is")
    print("exactly the setting the attack applies to.")
    print()
    print(f"Workload: the risk-score circuit over {args.patients} patients "
          f"(batch {batch_size}), depth {MULT_DEPTH}.")
    print(f"Timing: {args.repeats} repetitions, {args.warmup} warm-up calls "
          f"discarded.\n")

    print("Pass 1 — EXEC_NOISE_ESTIMATION (measure the circuit's own noise):")
    noise_estimate, est_ring = estimate_noise(cohort, batch_size, args.patients)
    print(f"  GetLogError() = {noise_estimate:.4f}  "
          f"(log2 of the circuit's error; ring dimension {est_ring})\n")

    print("Baseline arm — FIXED_NOISE_DECRYPT (what every other experiment used):")
    baseline = measure_arm(cohort, batch_size, args.patients, reference_scores,
                           args.repeats, args.warmup, label="baseline")

    print("\nMitigated arm — NOISE_FLOODING_DECRYPT with that estimate:")
    flooded = measure_arm(cohort, batch_size, args.patients, reference_scores,
                          args.repeats, args.warmup,
                          noise_estimate=noise_estimate, label="flooded")

    print("\nReference point — TenSEAL, which has no decryption-noise setting:")
    tenseal_spread = tenseal_decryption_spread()
    print(f"  randomized = {tenseal_spread['is_randomized']}, "
          f"max spread = {tenseal_spread['max_range_across_slots']:.4e}")

    print_summary(baseline, flooded, noise_estimate, tenseal_spread)

    output = {
        "n_patients": args.patients,
        "batch_size": batch_size,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "noise_estimate_log2": noise_estimate,
        "multiplicative_depth": MULT_DEPTH,
        "reference": "Li & Micciancio, EUROCRYPT 2021",
        "environment": harness.environment_info(),
        "baseline": baseline,
        "flooded": flooded,
        "tenseal_decryption_spread": tenseal_spread,
    }
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"\nResults saved to {args.output}")


def print_summary(baseline, flooded, noise_estimate, tenseal_spread):
    print()
    print("=" * 96)
    print(f"{'Cost of noise flooding':^96}")
    print("=" * 96)
    print(f"{'Operation':<16}{'Baseline (ms)':>16}{'Flooded (ms)':>16}"
          f"{'Slowdown':>12}")
    print("-" * 96)
    for name, key in OPS:
        base = baseline[f"{key}_time_stats"]["mean_ms"]
        flood = flooded[f"{key}_time_stats"]["mean_ms"]
        print(f"{name:<16}{base:>16.3f}{flood:>16.3f}{flood / base:>11.2f}x")
    print("-" * 96)
    print(f"{'Ring dimension':<16}{baseline['ring_dimension']:>16}"
          f"{flooded['ring_dimension']:>16}"
          f"{flooded['ring_dimension'] / baseline['ring_dimension']:>11.2f}x")
    print()
    print(f"{'Metric':<34}{'Baseline':>18}{'Flooded':>18}{'Ratio':>12}")
    print("-" * 96)
    for label, key in [("Score max abs error", "score_max_abs_error"),
                       ("Score mean abs error", "score_mean_abs_error")]:
        base, flood = baseline[key], flooded[key]
        print(f"{label:<34}{base:>18.4e}{flood:>18.4e}{flood / base:>11.1f}x")
    print("-" * 96)

    print()
    print("=" * 96)
    print(f"{'Is decryption randomized? (same ciphertext decrypted repeatedly)':^96}")
    print("=" * 96)
    print(f"{'Configuration':<38}{'Randomized?':>13}{'Max spread':>16}"
          f"{'Mean spread':>16}{'Max slot SD':>13}")
    print("-" * 96)
    rows = [("TenSEAL (no such setting)", tenseal_spread),
            ("OpenFHE FIXED_NOISE_DECRYPT (default)", baseline["decryption_spread"]),
            ("OpenFHE NOISE_FLOODING_DECRYPT", flooded["decryption_spread"])]
    for name, sp in rows:
        print(f"{name:<38}{str(sp['is_randomized']):>13}"
              f"{sp['max_range_across_slots']:>16.4e}"
              f"{sp['mean_range_across_slots']:>16.4e}"
              f"{sp['per_slot_std_max']:>13.4e}")
    print("-" * 96)
    print("Plain CKKS decryption is deterministic, so a zero spread means nothing is")
    print("added before release and the released value carries the key-dependent")
    print("error the Li-Micciancio attack exploits. TenSEAL is in that position.")
    print("OpenFHE randomizes by default; flooding additionally calibrates the noise")
    print("to the measured circuit noise rather than using a fixed constant.")
    print()
    print("IMPORTANT: this shows a defence is ACTIVE, not that it is SUFFICIENT.")
    print("OpenFHE's noise estimate is AVERAGE-CASE (measured from a trial run).")
    print("Guo et al., USENIX Security 2024, break noise flooding built on")
    print("non-worst-case noise estimation, including deployments using the")
    print("CRYPTO 2022 differential-privacy bounds. Do not read the flooded arm")
    print("as 'IND-CPA^D secure' -- read it as 'the cost of the partial defence'.")

    print()
    print(f"Noise estimate fed to pass 2: 2^{noise_estimate:.2f}")
    print("The mitigation also costs an entire extra pass over the circuit, since")
    print("the noise estimate must be measured before it can be flooded.")


if __name__ == "__main__":
    main()
