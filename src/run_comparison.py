"""
Run all three benchmarks and print a side-by-side comparison.
Saves results to results.json for use by plot_results.py.

Every number is a mean over `--repeats` timed calls, taken after a discarded
warm-up phase. Dispersion (SD and a 95% confidence interval on the mean) is
reported alongside, so the comparison states how stable each measurement is.

Usage: python run_comparison.py [--repeats 1000] [--warmup 50]
"""

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import benchmark_harness as harness
import project_paths
import plaintext_baseline as pt_bench
import tenseal_benchmark as ts_bench

try:
    import openfhe_benchmark as fhe_bench
    OPENFHE_AVAILABLE = True
    OPENFHE_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:
    OPENFHE_AVAILABLE = False
    OPENFHE_ERROR = str(exc)

DATA = [1.0, 2.0, 3.0, 4.0, 5.0]

HE_OPS = [
    ("Encrypt", "encrypt"),
    ("Add", "add"),
    ("Multiply", "mul"),
    ("Sum", "sum"),
    ("Average", "avg"),
    ("Dot Product", "dot"),
    ("Decrypt", "decrypt"),
]

# Operations that also exist in the plaintext baseline, so a per-operation
# overhead ratio is meaningful for them.
COMPARABLE_OPS = [
    ("Add", "add"),
    ("Multiply", "mul"),
    ("Sum", "sum"),
    ("Average", "avg"),
    ("Dot Product", "dot"),
]

W = 92


def mean_ms(results, key):
    stats = results.get(f"{key}_time_stats")
    return stats["mean_ms"] if stats else None


def print_timing_table(pt, ts, fhe):
    fhe_header = "OpenFHE" if fhe else "OpenFHE (N/A)"
    print()
    print("=" * W)
    print(f"{'Mean Time per Operation (ms)':^{W}}")
    print("=" * W)
    print(f"{'Operation':<14}{'Plaintext':>14}{'TenSEAL':>14}{fhe_header:>16}"
          f"{'TS overhead':>16}{'FHE overhead':>16}")
    print("-" * W)

    for name, key in HE_OPS:
        p = mean_ms(pt, key)
        t = mean_ms(ts, key)
        f = mean_ms(fhe, key) if fhe else None

        p_s = f"{p:>14.6f}" if p is not None else f"{'—':>14}"
        t_s = f"{t:>14.4f}" if t is not None else f"{'—':>14}"
        f_s = f"{f:>16.4f}" if f is not None else f"{'unavailable':>16}"
        t_o = f"{t / p:>15,.0f}x" if (p and t) else f"{'—':>16}"
        f_o = f"{f / p:>15,.0f}x" if (p and f) else f"{'—':>16}"
        print(f"{name:<14}{p_s}{t_s}{f_s}{t_o}{f_o}")

    print("-" * W)
    # Two totals, because one number cannot honestly serve both purposes.
    # The comparable total sums the SAME five operations on both sides — the only
    # ratio that is a like-for-like slowdown. The session total additionally
    # charges encryption and decryption, which plaintext has no analogue for; it
    # is a deployment cost, not a slowdown. An earlier version divided a
    # seven-operation encrypted total by a five-operation plaintext one and
    # labelled the result a slowdown.
    pt_total = sum(mean_ms(pt, k) for _, k in COMPARABLE_OPS)
    ts_total = sum(mean_ms(ts, k) for _, k in COMPARABLE_OPS)
    fhe_total = sum(mean_ms(fhe, k) for _, k in COMPARABLE_OPS) if fhe else None
    ts_session = sum(mean_ms(ts, k) for _, k in HE_OPS)
    fhe_session = sum(mean_ms(fhe, k) for _, k in HE_OPS) if fhe else None

    fhe_total_s = f"{fhe_total:>16.4f}" if fhe_total else f"{'unavailable':>16}"
    print(f"{'Total (5 ops)':<14}{pt_total:>14.6f}{ts_total:>14.4f}{fhe_total_s}"
          f"{ts_total / pt_total:>15,.0f}x"
          + (f"{fhe_total / pt_total:>15,.0f}x" if fhe_total else f"{'—':>16}"))
    fhe_sess_s = f"{fhe_session:>16.4f}" if fhe_session else f"{'unavailable':>16}"
    print(f"{'Session (7)':<14}{'—':>14}{ts_session:>14.4f}{fhe_sess_s}"
          f"{'—':>16}{'—':>16}")
    print("Total sums the five operations plaintext also has. Session adds encrypt and")
    print("decrypt, which plaintext has no analogue for — a deployment cost, not a ratio.")
    return pt_total, ts_total, fhe_total, ts_session, fhe_session


def print_dispersion_table(ts, fhe):
    """Show that the means are stable, not single noisy samples."""
    print()
    print("=" * W)
    print(f"{'Measurement Stability (relative standard deviation of each mean)':^{W}}")
    print("=" * W)
    print(f"{'Operation':<14}{'TenSEAL mean':>15}{'TS rel. SD':>13}{'TS 95% CI ±':>14}"
          f"{'OpenFHE mean':>15}{'FHE rel. SD':>13}{'FHE 95% CI ±':>14}")
    print("-" * W)
    for name, key in HE_OPS:
        ts_st = ts[f"{key}_time_stats"]
        row = (f"{name:<14}{ts_st['mean_ms']:>15.4f}{ts_st['rel_std_pct']:>12.1f}%"
               f"{ts_st['ci95_half_width_ms']:>14.4f}")
        if fhe:
            f_st = fhe[f"{key}_time_stats"]
            row += (f"{f_st['mean_ms']:>15.4f}{f_st['rel_std_pct']:>12.1f}%"
                    f"{f_st['ci95_half_width_ms']:>14.4f}")
        print(row)
    print("-" * W)


def print_memory_table(pt, ts, fhe):
    fhe_header = "OpenFHE" if fhe else "OpenFHE (N/A)"
    print()
    print("=" * W)
    print(f"{'Peak Python-side Memory per Operation (KB)':^{W}}")
    print("=" * W)
    print(f"{'Operation':<16}{'Plaintext':>14}{'TenSEAL':>14}{fhe_header:>16}")
    print("-" * W)
    for name, key in HE_OPS:
        p = pt.get(f"{key}_mem_kb")
        p_s = f"{p:>14.4f}" if p is not None else f"{'—':>14}"
        f_s = f"{fhe[f'{key}_mem_kb']:>16.2f}" if fhe else f"{'unavailable':>16}"
        print(f"{name:<16}{p_s}{ts[f'{key}_mem_kb']:>14.2f}{f_s}")
    print("-" * W)
    print("tracemalloc sees Python allocations only; both libraries hold ciphertexts")
    print("in native heaps, so these are lower bounds on true memory use.")


def benchmark_key_generation(repeats, warmup):
    """Time context + key generation, the one-off setup cost of each library.

    This is a per-session cost rather than a per-operation one, so it is timed
    separately and with fewer repetitions than the operations themselves.
    """
    print(f"Timing key generation ({repeats} repetitions)...")
    keygen = {"repeats": repeats, "warmup": warmup}
    keygen["tenseal"] = harness.time_operation(
        ts_bench.make_context, repeats=repeats, warmup=warmup)
    if OPENFHE_AVAILABLE:
        keygen["openfhe"] = harness.time_operation(
            lambda: fhe_bench.make_context(len(DATA)), repeats=repeats, warmup=warmup)
    return keygen


def print_keygen_table(keygen):
    print()
    print("=" * W)
    print(f"{'Context + Key Generation (one-off setup cost)':^{W}}")
    print("=" * W)
    print(f"{'Library':<16}{'Mean (ms)':>14}{'SD (ms)':>12}{'95% CI ±':>12}{'Median':>12}")
    print("-" * W)
    for label, key in [("TenSEAL", "tenseal"), ("OpenFHE", "openfhe")]:
        st = keygen.get(key)
        if st is None:
            continue
        print(f"{label:<16}{st['mean_ms']:>14.3f}{st['std_ms']:>12.3f}"
              f"{st['ci95_half_width_ms']:>12.3f}{st['median_ms']:>12.3f}")
    print("-" * W)
    print(f"{keygen['repeats']} repetitions. Includes crypto context construction, secret/public")
    print("key generation, and the evaluation keys each library needs (relinearization,")
    print("Galois/rotation keys for TenSEAL; multiplication and summation keys for OpenFHE).")


def print_error_table(ts, fhe):
    fhe_header = "OpenFHE" if fhe else "OpenFHE (N/A)"
    print()
    print("=" * W)
    print(f"{'Approximation Error (max absolute error)':^{W}}")
    print("=" * W)
    print(f"{'Operation':<16}{'TenSEAL':>22}{fhe_header:>22}")
    print("-" * W)
    for name, key in [
        ("Add", "add_max_error"),
        ("Multiply", "mul_max_error"),
        ("Sum", "sum_error"),
        ("Average", "avg_error"),
        ("Dot Product", "dot_error"),
    ]:
        f_s = f"{fhe[key]:>22.4e}" if fhe else f"{'unavailable':>22}"
        print(f"{name:<16}{ts[key]:>22.4e}{f_s}")
    print("-" * W)


def main():
    parser = argparse.ArgumentParser(description="TenSEAL vs OpenFHE vs plaintext")
    harness.add_repeat_arguments(parser)
    parser.add_argument("--keygen-repeats", type=int, default=50,
                        help="repetitions for the key-generation timing (default 50)")
    parser.add_argument("--output",
                        default=project_paths.result_path("results.json"),
                        help="where to write the JSON results")
    args = parser.parse_args()

    print(f"Input vector : {DATA}")
    print(f"Vector length: {len(DATA)}")
    print(f"Timing       : {args.repeats} repetitions per operation, "
          f"{args.warmup} discarded warm-up calls\n")

    if not OPENFHE_AVAILABLE:
        print(f"[NOTE] OpenFHE unavailable on this platform: {OPENFHE_ERROR}")
        print("       OpenFHE ships Linux-only Python wheels, built for a specific")
        print("       CPython version. Documented as a practical limitation.\n")

    kwargs = dict(repeats=args.repeats, warmup=args.warmup,
                  memory_repeats=args.memory_repeats)

    print("Running plaintext baseline...")
    pt = pt_bench.run_benchmark(DATA, **kwargs)

    print("Running TenSEAL benchmark...")
    ts = ts_bench.run_benchmark(DATA, **kwargs)

    if OPENFHE_AVAILABLE:
        print("Running OpenFHE benchmark...")
        fhe = fhe_bench.run_benchmark(DATA, **kwargs)
    else:
        fhe = None

    keygen = benchmark_key_generation(args.keygen_repeats, min(args.warmup, 5))

    pt_total, ts_total, fhe_total, ts_session, fhe_session = print_timing_table(pt, ts, fhe)
    print_keygen_table(keygen)
    print_dispersion_table(ts, fhe)
    print_memory_table(pt, ts, fhe)
    print_error_table(ts, fhe)

    if "ciphertext_bytes" in ts:
        print(f"\nTenSEAL ciphertext size: {ts['ciphertext_bytes']} bytes "
              f"(plaintext vector: {len(DATA) * 8} bytes as float64)")
    if fhe and "ring_dimension" in fhe:
        print(f"OpenFHE ring dimension : {fhe['ring_dimension']}")

    output = {
        "vector": DATA,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "plaintext": pt,
        "tenseal": ts,
        "openfhe": fhe,
        "key_generation": keygen,
        "openfhe_available": OPENFHE_AVAILABLE,
        "environment": harness.environment_info(),
        "totals_ms": {
            "comparable_operations": {
                "note": "same five operations on all three columns",
                "plaintext": pt_total, "tenseal": ts_total, "openfhe": fhe_total,
            },
            "full_session": {
                "note": "adds encrypt and decrypt; no plaintext analogue",
                "tenseal": ts_session, "openfhe": fhe_session,
            },
        },
    }
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
