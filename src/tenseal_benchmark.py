"""
TenSEAL Benchmark — CKKS scheme
Times encrypt/add/mul/sum/avg/dot/decrypt and checks accuracy.

Install: pip install tenseal
Usage:   python tenseal_benchmark.py [--repeats 1000] [--warmup 50]
"""

import argparse


import tenseal as ts

import benchmark_harness as harness

OP_NAMES = [
    ("Encrypt", "encrypt"),
    ("Add", "add"),
    ("Multiply", "mul"),
    ("Sum", "sum"),
    ("Average", "avg"),
    ("Dot Product", "dot"),
    ("Decrypt", "decrypt"),
]

# Sanity bounds, ~3 orders of magnitude above the error these parameters produce.
ACCURACY_LIMITS = {
    "add_max_error": 1e-4,
    "mul_max_error": 1e-2,
    "sum_error": 1e-3,
    "avg_error": 1e-3,
    "dot_error": 1e-2,
}


def make_context():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40
    return ctx


def run_benchmark(vector, repeats=harness.DEFAULT_REPEATS, warmup=harness.DEFAULT_WARMUP,
                  memory_repeats=harness.DEFAULT_MEMORY_REPEATS, verbose=True):
    ctx = make_context()
    n = len(vector)
    vector2 = [v * 0.5 + 1.0 for v in vector]

    enc = ts.ckks_vector(ctx, vector)
    enc2 = ts.ckks_vector(ctx, vector2)

    # Each lambda rebuilds its result from enc/enc2, so repetitions do not
    # accumulate noise or consume levels.
    operations = {
        "encrypt": lambda: ts.ckks_vector(ctx, vector),
        "add": lambda: enc + enc,
        "mul": lambda: enc * enc,
        "sum": lambda: enc.sum(),
        "avg": lambda: enc.sum() * (1.0 / n),
        "dot": lambda: (enc * enc2).sum(),
        "decrypt": lambda: enc.decrypt(),
    }

    results = harness.benchmark_operations(
        operations, repeats=repeats, warmup=warmup,
        memory_repeats=memory_repeats, label="TenSEAL", verbose=verbose,
    )

    # Correctness checks (approximate: CKKS is not exact)
    dec_add = (enc + enc).decrypt()
    dec_mul = (enc * enc).decrypt()
    dec_sum = enc.sum().decrypt()[0]
    dec_avg = (enc.sum() * (1.0 / n)).decrypt()[0]
    dec_dot = (enc * enc2).sum().decrypt()[0]

    expected_add = [v + v for v in vector]
    expected_mul = [v * v for v in vector]
    expected_sum = sum(vector)
    expected_avg = expected_sum / n
    expected_dot = sum(a * b for a, b in zip(vector, vector2))

    results["add_max_error"] = max(abs(a - b) for a, b in zip(dec_add, expected_add))
    results["mul_max_error"] = max(abs(a - b) for a, b in zip(dec_mul, expected_mul))
    results["sum_error"] = abs(dec_sum - expected_sum)
    results["avg_error"] = abs(dec_avg - expected_avg)
    results["dot_error"] = abs(dec_dot - expected_dot)

    results["ciphertext_bytes"] = len(enc.serialize())

    # RSS, not tracemalloc: SEAL allocates ciphertext coefficients on the
    # native heap, which tracemalloc does not track.
    results["ciphertext_rss_bytes"] = harness.measure_retained_bytes(
        lambda: ts.ckks_vector(ctx, vector))

    harness.assert_accuracy(results, ACCURACY_LIMITS, label="TenSEAL")

    return results


def print_results(results):
    print()
    print(harness.format_stats_table(results, OP_NAMES, "TenSEAL (CKKS) Benchmark Results"))
    print()
    print(f"{'Operation':<14}{'Peak Python mem (KB)':>22}")
    print("-" * 36)
    for name, key in OP_NAMES:
        print(f"{name:<14}{results[f'{key}_mem_kb']:>22.2f}")
    print("-" * 36)
    print(f"Ciphertext size : {results['ciphertext_bytes']:,} bytes (serialized)")
    if results.get("ciphertext_rss_bytes"):
        print(f"Ciphertext RSS  : {results['ciphertext_rss_bytes']:,.0f} bytes resident")
    print(f"Add  max error  : {results['add_max_error']:.2e}")
    print(f"Mul  max error  : {results['mul_max_error']:.2e}")
    print(f"Sum  error      : {results['sum_error']:.2e}")
    print(f"Avg  error      : {results['avg_error']:.2e}")
    print(f"Dot  error      : {results['dot_error']:.2e}")


def main():
    parser = argparse.ArgumentParser(description="TenSEAL CKKS benchmark")
    harness.add_repeat_arguments(parser)
    args = parser.parse_args()

    print(f"Input vector: {harness.DEFAULT_VECTOR}\n")
    results = run_benchmark(harness.DEFAULT_VECTOR, repeats=args.repeats,
                            warmup=args.warmup, memory_repeats=args.memory_repeats)
    print_results(results)


if __name__ == "__main__":
    main()
