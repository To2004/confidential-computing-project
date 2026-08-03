"""
Scaling benchmark: end-to-end HE pipeline time at vector sizes 5, 10, 50, 100, 500.
Context and key generation run once per size and are not timed.
Saves results to scaling_results.json.

Usage: python benchmark_scaling.py [--repeats 1000] [--warmup 20]
"""

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import tenseal as ts

import benchmark_harness as harness
import project_paths

try:
    from openfhe import CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature
    OPENFHE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    OPENFHE_AVAILABLE = False

SIZES = [5, 10, 50, 100, 500]

# Shorter warm-up than the primitive benchmark: a full pipeline call is much slower.
DEFAULT_SCALING_WARMUP = 20


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


# ── TenSEAL ──────────────────────────────────────────────────────────────────

def tenseal_pipeline(vector):
    """Build the context once, return the callable that is timed."""
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40
    n = len(vector)

    def pipeline():
        enc = ts.ckks_vector(ctx, vector)
        enc_add = enc + enc
        enc_mul = enc * enc
        enc_sum = enc.sum()
        enc_avg = enc.sum() * (1.0 / n)
        enc_add.decrypt()
        enc_mul.decrypt()
        enc_sum.decrypt()
        enc_avg.decrypt()

    return pipeline


# ── OpenFHE ───────────────────────────────────────────────────────────────────

def openfhe_pipeline(vector):
    n = len(vector)
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(3)
    params.SetScalingModSize(50)
    params.SetBatchSize(_next_power_of_two(n))
    cc = GenCryptoContext(params)
    for feature in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH,
                    PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(feature)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    def dec(ciphertext, length):
        plain = cc.Decrypt(keys.secretKey, ciphertext)
        plain.SetLength(length)
        return plain.GetRealPackedValue()

    def pipeline():
        pt = cc.MakeCKKSPackedPlaintext(vector)
        ct = cc.Encrypt(keys.publicKey, pt)
        ct_add = cc.EvalAdd(ct, ct)
        ct_mul = cc.EvalMult(ct, ct)
        ct_sum = cc.EvalSum(ct, n)
        ct_avg = cc.EvalMult(cc.EvalSum(ct, n), 1.0 / n)
        dec(ct_add, n)
        dec(ct_mul, n)
        dec(ct_sum, 1)
        dec(ct_avg, 1)

    return pipeline, cc.GetRingDimension()


# ── Plaintext baseline ────────────────────────────────────────────────────────

def plaintext_pipeline(vector):
    n = len(vector)

    def pipeline():
        [v + v for v in vector]
        [v * v for v in vector]
        sum(vector)
        sum(vector) / n

    return pipeline


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HE scaling benchmark")
    parser.add_argument("--repeats", type=int, default=harness.DEFAULT_REPEATS,
                        help=f"timed repetitions per size (default {harness.DEFAULT_REPEATS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_SCALING_WARMUP,
                        help=f"discarded warm-up calls (default {DEFAULT_SCALING_WARMUP})")
    parser.add_argument("--output",
                        default=project_paths.result_path("scaling_results.json"),
                        help="where to write the JSON results")
    args = parser.parse_args()

    if not OPENFHE_AVAILABLE:
        print("[NOTE] OpenFHE unavailable — only TenSEAL and plaintext will be measured.\n")

    print(f"Timing: {args.repeats} repetitions per size, "
          f"{args.warmup} discarded warm-up calls.\n")

    results = {
        "sizes": SIZES,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "plaintext_ms": [],
        "tenseal_ms": [],
        "openfhe_ms": [],
        "plaintext_stats": [],
        "tenseal_stats": [],
        "openfhe_stats": [],
        "openfhe_ring_dimension": [],
        "openfhe_available": OPENFHE_AVAILABLE,
    }

    for size in SIZES:
        vec = [float(i + 1) for i in range(size)]
        print(f"Testing size = {size} ...", flush=True)

        pt_stats = harness.time_operation(plaintext_pipeline(vec),
                                          repeats=args.repeats, warmup=args.warmup)
        ts_stats = harness.time_operation(tenseal_pipeline(vec),
                                          repeats=args.repeats, warmup=args.warmup)
        if OPENFHE_AVAILABLE:
            fhe_fn, ring_dim = openfhe_pipeline(vec)
            fhe_stats = harness.time_operation(fhe_fn, repeats=args.repeats,
                                               warmup=args.warmup)
        else:
            fhe_stats, ring_dim = None, None

        results["plaintext_ms"].append(round(pt_stats["mean_ms"], 6))
        results["tenseal_ms"].append(round(ts_stats["mean_ms"], 4))
        results["openfhe_ms"].append(round(fhe_stats["mean_ms"], 4) if fhe_stats else None)
        results["plaintext_stats"].append(pt_stats)
        results["tenseal_stats"].append(ts_stats)
        results["openfhe_stats"].append(fhe_stats)
        results["openfhe_ring_dimension"].append(ring_dim)

        print(f"  Plaintext : {pt_stats['mean_ms']:.6f} ms "
              f"± {pt_stats['ci95_half_width_ms']:.6f}")
        print(f"  TenSEAL   : {ts_stats['mean_ms']:.4f} ms "
              f"± {ts_stats['ci95_half_width_ms']:.4f} "
              f"({ts_stats['rel_std_pct']:.1f}% rel. sd)")
        if fhe_stats:
            print(f"  OpenFHE   : {fhe_stats['mean_ms']:.4f} ms "
                  f"± {fhe_stats['ci95_half_width_ms']:.4f} "
                  f"({fhe_stats['rel_std_pct']:.1f}% rel. sd), ring dim {ring_dim}")

    print("\nScaling results (mean ms over "
          f"{args.repeats} repetitions):")
    print(f"{'Size':>6}{'Plaintext (ms)':>18}{'TenSEAL (ms)':>16}{'OpenFHE (ms)':>16}"
          f"{'TS overhead':>14}{'FHE overhead':>14}")
    print("-" * 84)
    for i, size in enumerate(SIZES):
        pt_ms = results["plaintext_ms"][i]
        ts_ms = results["tenseal_ms"][i]
        fhe_ms = results["openfhe_ms"][i]
        fhe_s = f"{fhe_ms:>16.2f}" if fhe_ms is not None else f"{'unavailable':>16}"
        fhe_o = f"{fhe_ms / pt_ms:>13,.0f}x" if fhe_ms is not None else f"{'—':>14}"
        print(f"{size:>6}{pt_ms:>18.6f}{ts_ms:>16.2f}{fhe_s}"
              f"{ts_ms / pt_ms:>13,.0f}x{fhe_o}")

    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
