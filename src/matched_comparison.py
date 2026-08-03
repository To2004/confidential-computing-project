"""
Like-for-like comparison: OpenFHE forced onto TenSEAL's parameters.

Why this experiment exists
--------------------------
The main benchmark compares TenSEAL at ring dimension 8192 with a $2^{40}$
scaling factor against OpenFHE at ring dimension 16384 with a $2^{50}$ scaling
factor, because those are what each library produces from its natural
configuration route: TenSEAL is given a modulus chain directly, while OpenFHE is
given a depth and a scaling size and picks a ring dimension to satisfy them at a
128-bit security level.

That makes the headline comparison ambiguous. OpenFHE was slower *and* more
precise, but it was also doing twice the polynomial work per operation and
keeping ten more bits of each value. Attributing either observation to the
library rather than to the parameters is unsound.

`SetRingDim` plus `SetSecurityLevel(HEStd_NotSet)` lets OpenFHE be pinned to
8192 with a 40-bit scaling factor and a 60-bit first modulus — the same as
TenSEAL's `[60, 40, 40, 60]`. Whatever difference survives that is a property of
the implementations.

Security of the matched arm — a correction
-----------------------------------------
An earlier version of this file argued that the pinned arm was still 128-bit
secure, because the modulus OpenFHE reports at ring 8192 is 181-200 bits and
SEAL's published ceiling at that ring dimension is 218. **That argument was
wrong**, and this experiment now measures rather than asserts.

Two things were missed. First, SEAL's ceiling bounds the *entire* coefficient
modulus including the special key-switching prime, whereas OpenFHE's reported
modulus does not include the auxiliary primes its HYBRID key switching carries —
so the two numbers being compared were not the same quantity. Second, and
decisively: given exactly this configuration and asked to choose a ring dimension
at `HEStd_128_classic`, **OpenFHE selects 16384, not 8192**. The library refuses
the very parameter set we pin it to. `probe_library_chosen_ring` below records
that refusal in the results file.

The matched arm is therefore run *below* 128-bit security, and the comparison
trades security for parameter parity. That is still the right experiment — it is
the only way to hold ring dimension and scale fixed across two libraries — but it
must be reported as what it is, and the report now says so.

The sweep also varies OpenFHE's *scaling technique*, which turns out to matter
more for precision than the scaling factor does.

Usage: python matched_comparison.py [--repeats 200]
"""

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import openfhe as fhe
import tenseal as ts
from openfhe import (CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature,
                     SecurityLevel)

import benchmark_harness as harness
import project_paths

VECTOR = [1.0, 2.0, 3.0, 4.0, 5.0]

# TenSEAL's configuration, which OpenFHE is being matched to.
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
SCALE_BITS = 40
FIRST_MOD_BITS = 60
MULT_DEPTH = 3

# A full pipeline of primitives per repetition, so fewer repetitions than the
# main benchmark; the point here is the ratio between arms, not absolute timing.
DEFAULT_REPEATS = 200
DEFAULT_WARMUP = 20

SCALING_TECHNIQUES = ("FIXEDMANUAL", "FIXEDAUTO", "FLEXIBLEAUTO", "FLEXIBLEAUTOEXT")

OPS = [("Encrypt", "encrypt"), ("Add", "add"), ("Multiply", "mul"),
       ("Sum", "sum"), ("Dot Product", "dot"), ("Decrypt", "decrypt")]


def _second_vector(vector):
    return [v * 0.5 + 1.0 for v in vector]


# ── TenSEAL arm ──────────────────────────────────────────────────────────────

def tenseal_arm(vector, repeats, warmup):
    ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                     poly_modulus_degree=POLY_MODULUS_DEGREE,
                     coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES)
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** SCALE_BITS

    enc = ts.ckks_vector(ctx, vector)
    enc2 = ts.ckks_vector(ctx, _second_vector(vector))

    operations = {
        "encrypt": lambda: ts.ckks_vector(ctx, vector),
        "add": lambda: enc + enc,
        "mul": lambda: enc * enc,
        "sum": lambda: enc.sum(),
        "dot": lambda: (enc * enc2).sum(),
        "decrypt": lambda: enc.decrypt(),
    }
    results = harness.benchmark_operations(operations, repeats=repeats, warmup=warmup,
                                           label="TenSEAL", keep_samples=False)

    results["mul_max_error"] = max(
        abs(a - v * v) for a, v in zip((enc * enc).decrypt(), vector))
    results["ring_dimension"] = POLY_MODULUS_DEGREE
    results["scale_bits"] = SCALE_BITS
    results["arm"] = "TenSEAL (reference configuration)"
    results["scaling_technique"] = "SEAL auto-rescale"
    results["ciphertext_bytes"] = len(enc.serialize())
    return results


# ── OpenFHE arms ─────────────────────────────────────────────────────────────

def openfhe_arm(vector, repeats, warmup, scaling_technique=None,
                ring_dim=POLY_MODULUS_DEGREE, scale_bits=SCALE_BITS,
                first_mod_bits=FIRST_MOD_BITS, label=""):
    n = len(vector)
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(MULT_DEPTH)
    params.SetScalingModSize(scale_bits)
    params.SetFirstModSize(first_mod_bits)
    params.SetBatchSize(8)

    if ring_dim is None:
        # Let OpenFHE choose, at a 128-bit security level: the default arm.
        params.SetSecurityLevel(fhe.HEStd_128_classic)
    else:
        # Pinning the ring dimension disables the library's own security check.
        params.SetRingDim(ring_dim)
        params.SetSecurityLevel(SecurityLevel.HEStd_NotSet)

    if scaling_technique is not None:
        params.SetScalingTechnique(getattr(fhe.ScalingTechnique, scaling_technique))

    cc = GenCryptoContext(params)
    for feature in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH,
                    PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(feature)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    ct = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vector))
    ct2 = cc.Encrypt(keys.publicKey,
                     cc.MakeCKKSPackedPlaintext(_second_vector(vector)))

    operations = {
        "encrypt": lambda: cc.Encrypt(keys.publicKey,
                                      cc.MakeCKKSPackedPlaintext(vector)),
        "add": lambda: cc.EvalAdd(ct, ct),
        "mul": lambda: cc.EvalMult(ct, ct),
        "sum": lambda: cc.EvalSum(ct, n),
        "dot": lambda: cc.EvalInnerProduct(ct, ct2, n),
        "decrypt": lambda: cc.Decrypt(keys.secretKey, ct),
    }
    results = harness.benchmark_operations(operations, repeats=repeats, warmup=warmup,
                                           label=label or "OpenFHE",
                                           keep_samples=False)

    out = cc.Decrypt(keys.secretKey, cc.EvalMult(ct, ct))
    out.SetLength(n)
    results["mul_max_error"] = max(
        abs(a - v * v) for a, v in zip(out.GetRealPackedValue(), vector))
    results["ring_dimension"] = cc.GetRingDimension()
    # Measured, not asserted: the previous version of this experiment claimed a
    # modulus size in prose with no code behind it.
    results["modulus_bits"] = int(cc.GetModulus()).bit_length()
    results["scale_bits"] = scale_bits
    results["scaling_technique"] = scaling_technique or "FLEXIBLEAUTOEXT (default)"
    results["arm"] = label
    results["security_level"] = ("HEStd_128_classic" if ring_dim is None
                                 else "HEStd_NotSet (ring dimension pinned)")
    return results


def probe_library_chosen_ring(scale_bits=SCALE_BITS, first_mod_bits=FIRST_MOD_BITS):
    """What ring dimension does OpenFHE itself pick for the matched parameters?

    This is the evidence behind the retraction in the module docstring. If the
    library, asked for `HEStd_128_classic` at this depth and scale, chooses a
    ring dimension larger than the one we pin, then the pinned arm is running
    below the security level it was claimed to meet.
    """
    findings = {}
    for technique in SCALING_TECHNIQUES:
        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(MULT_DEPTH)
        params.SetScalingModSize(scale_bits)
        params.SetFirstModSize(first_mod_bits)
        params.SetBatchSize(8)
        params.SetSecurityLevel(fhe.HEStd_128_classic)
        params.SetScalingTechnique(getattr(fhe.ScalingTechnique, technique))
        cc = GenCryptoContext(params)
        cc.Enable(PKESchemeFeature.PKE)
        findings[technique] = {
            "library_chosen_ring": cc.GetRingDimension(),
            "pinned_ring": POLY_MODULUS_DEGREE,
            "modulus_bits": int(cc.GetModulus()).bit_length(),
            "pinned_arm_is_128_bit": cc.GetRingDimension() <= POLY_MODULUS_DEGREE,
        }
    return findings


def main():
    parser = argparse.ArgumentParser(
        description="OpenFHE at TenSEAL's parameters: what survives matching?")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"timed repetitions per operation (default {DEFAULT_REPEATS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"discarded warm-up calls (default {DEFAULT_WARMUP})")
    parser.add_argument("--output",
                        default=project_paths.result_path("matched_comparison_results.json"),
                        help="where to write the JSON results")
    args = parser.parse_args()

    print("=" * 100)
    print(f"{'Matched-Parameter Comparison':^100}")
    print("=" * 100)
    print(f"TenSEAL reference: poly_modulus_degree={POLY_MODULUS_DEGREE}, "
          f"coeff_mod={COEFF_MOD_BIT_SIZES}, scale=2^{SCALE_BITS}")
    print(f"Timing: {args.repeats} repetitions per operation, "
          f"{args.warmup} discarded warm-up calls\n")

    arms = {}

    print("Security check — what ring dimension does OpenFHE choose itself?")
    ring_probe = probe_library_chosen_ring()
    for technique, f in ring_probe.items():
        verdict = ("pinned arm IS 128-bit secure" if f["pinned_arm_is_128_bit"]
                   else "pinned arm is NOT 128-bit secure")
        print(f"  {technique:<16} library picks ring {f['library_chosen_ring']:>6}"
              f"  (we pin {f['pinned_ring']})  -> {verdict}")
    print()

    print("TenSEAL (reference):")
    arms["tenseal"] = tenseal_arm(VECTOR, args.repeats, args.warmup)

    print("\nOpenFHE, parameters chosen by the library (the main benchmark's arm):")
    arms["openfhe_default"] = openfhe_arm(
        VECTOR, args.repeats, args.warmup, ring_dim=None, scale_bits=50,
        label="OpenFHE (library-chosen: ring 16384, scale 2^50)")

    for technique in SCALING_TECHNIQUES:
        print(f"\nOpenFHE, matched to TenSEAL, scaling technique {technique}:")
        arms[f"openfhe_matched_{technique.lower()}"] = openfhe_arm(
            VECTOR, args.repeats, args.warmup, scaling_technique=technique,
            label=f"OpenFHE matched ({technique})")

    print_summary(arms)

    output = {
        "vector": VECTOR,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "tenseal_reference_parameters": {
            "poly_modulus_degree": POLY_MODULUS_DEGREE,
            "coeff_mod_bit_sizes": COEFF_MOD_BIT_SIZES,
            "scale_bits": SCALE_BITS,
        },
        "environment": harness.environment_info(),
        "library_chosen_ring_probe": ring_probe,
        "arms": arms,
    }
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"\nResults saved to {args.output}")


def print_summary(arms):
    reference = arms["tenseal"]
    ref_mul = reference["mul_time_stats"]["mean_ms"]
    ref_err = reference["mul_max_error"]

    print()
    print("=" * 100)
    print(f"{'Multiplication: time and precision across arms':^100}")
    print("=" * 100)
    print(f"{'Arm':<44}{'Ring':>7}{'Scale':>7}{'Mul (ms)':>11}"
          f"{'vs TenSEAL':>12}{'Mul error':>12}{'vs TenSEAL':>10}")
    print("-" * 100)
    for key, res in arms.items():
        mul = res["mul_time_stats"]["mean_ms"]
        err = res["mul_max_error"]
        name = res["arm"][:43]
        print(f"{name:<44}{res['ring_dimension']:>7}"
              f"{res['scale_bits']:>7}{mul:>11.3f}"
              f"{mul / ref_mul:>11.2f}x{err:>12.2e}"
              f"{ref_err / err:>9.0f}x")
    print("-" * 100)
    print("'vs TenSEAL' on error is how many times MORE precise than TenSEAL "
          "(higher = better).")

    print()
    print("=" * 100)
    print(f"{'All operations, mean ms':^100}")
    print("=" * 100)
    header = f"{'Arm':<44}" + "".join(f"{n:>9}" for n, _ in OPS)
    print(header)
    print("-" * 100)
    for key, res in arms.items():
        row = f"{res['arm'][:43]:<44}"
        for _, op_key in OPS:
            st = res.get(f"{op_key}_time_stats")
            row += f"{st['mean_ms']:>9.2f}" if st else f"{'—':>9}"
        print(row)
    print("-" * 100)


if __name__ == "__main__":
    main()
