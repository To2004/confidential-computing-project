"""
OpenFHE Benchmark — CKKS scheme
Implements all mandatory operations from the project proposal.

Every operation is timed over many repetitions after a discarded warm-up phase,
and the reported figure is the mean; see `benchmark_harness` for the rationale.

Install: pip install openfhe
Usage:   python openfhe_benchmark.py [--repeats 1000] [--warmup 50]
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import os
import tempfile

from openfhe import (
    BINARY,
    CCParamsCKKSRNS,
    GenCryptoContext,
    HEStd_128_classic,
    PKESchemeFeature,
    SerializeToFile,
)

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

# See the equivalent table in tenseal_benchmark: loose bounds that catch a broken
# circuit rather than ordinary CKKS noise.
ACCURACY_LIMITS = {
    "add_max_error": 1e-6,
    "mul_max_error": 1e-6,
    "sum_error": 1e-6,
    "avg_error": 1e-6,
    "dot_error": 1e-6,
}


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def make_context(batch_size: int):
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(3)
    params.SetScalingModSize(50)
    params.SetBatchSize(_next_power_of_two(batch_size))
    # Stated explicitly rather than left to the default, so the report can name
    # the security level it is measuring at. OpenFHE derives the ring dimension
    # from this together with the depth and scaling size.
    params.SetSecurityLevel(HEStd_128_classic)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    return cc, keys


def run_benchmark(vector, repeats=harness.DEFAULT_REPEATS, warmup=harness.DEFAULT_WARMUP,
                  memory_repeats=harness.DEFAULT_MEMORY_REPEATS, verbose=True):
    n = len(vector)
    cc, keys = make_context(batch_size=n)
    vector2 = [v * 0.5 + 1.0 for v in vector]

    pt = cc.MakeCKKSPackedPlaintext(vector)
    ct = cc.Encrypt(keys.publicKey, pt)
    ct2 = cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vector2))

    # Each operation starts from the same freshly encrypted operands, so no
    # repetition inherits noise or a consumed level from the previous one.
    operations = {
        "encrypt": lambda: cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vector)),
        "add": lambda: cc.EvalAdd(ct, ct),
        "mul": lambda: cc.EvalMult(ct, ct),
        "sum": lambda: cc.EvalSum(ct, n),
        "avg": lambda: cc.EvalMult(cc.EvalSum(ct, n), 1.0 / n),
        "dot": lambda: cc.EvalInnerProduct(ct, ct2, n),
        "decrypt": lambda: cc.Decrypt(keys.secretKey, ct),
    }

    results = harness.benchmark_operations(
        operations, repeats=repeats, warmup=warmup,
        memory_repeats=memory_repeats, label="OpenFHE", verbose=verbose,
    )

    def decrypt_vec(ct_val, length):
        pt_out = cc.Decrypt(keys.secretKey, ct_val)
        pt_out.SetLength(length)
        return pt_out.GetRealPackedValue()

    dec_add = decrypt_vec(cc.EvalAdd(ct, ct), n)
    dec_mul = decrypt_vec(cc.EvalMult(ct, ct), n)
    dec_sum = decrypt_vec(cc.EvalSum(ct, n), 1)[0]
    dec_avg = decrypt_vec(cc.EvalMult(cc.EvalSum(ct, n), 1.0 / n), 1)[0]
    dec_dot = decrypt_vec(cc.EvalInnerProduct(ct, ct2, n), 1)[0]

    # Correctness checks
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

    results["ring_dimension"] = cc.GetRingDimension()
    results["security_level"] = "HEStd_128_classic"

    # Serialized ciphertext size. OpenFHE serializes through a file rather than
    # returning bytes, which is why the earlier version of this benchmark had no
    # figure here at all and the report could only quote TenSEAL's.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ciphertext.bin")
        if SerializeToFile(path, ct, BINARY):
            results["ciphertext_bytes"] = os.path.getsize(path)

    # Resident memory actually held by one ciphertext, which tracemalloc cannot see.
    results["ciphertext_rss_bytes"] = harness.measure_retained_bytes(
        lambda: cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(vector)))

    harness.assert_accuracy(results, ACCURACY_LIMITS, label="OpenFHE")

    return results


def print_results(results):
    print()
    print(harness.format_stats_table(results, OP_NAMES, "OpenFHE (CKKS) Benchmark Results"))
    print()
    print(f"{'Operation':<14}{'Peak Python mem (KB)':>22}")
    print("-" * 36)
    for name, key in OP_NAMES:
        print(f"{name:<14}{results[f'{key}_mem_kb']:>22.2f}")
    print("-" * 36)
    print(f"Ring dimension  : {results['ring_dimension']}")
    print(f"Security level  : {results['security_level']}")
    if "ciphertext_bytes" in results:
        print(f"Ciphertext size : {results['ciphertext_bytes']:,} bytes (serialized)")
    if results.get("ciphertext_rss_bytes"):
        print(f"Ciphertext RSS  : {results['ciphertext_rss_bytes']:,.0f} bytes resident")
    print(f"Add  max error  : {results['add_max_error']:.2e}")
    print(f"Mul  max error  : {results['mul_max_error']:.2e}")
    print(f"Sum  error      : {results['sum_error']:.2e}")
    print(f"Avg  error      : {results['avg_error']:.2e}")
    print(f"Dot  error      : {results['dot_error']:.2e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenFHE CKKS benchmark")
    harness.add_repeat_arguments(parser)
    args = parser.parse_args()

    DATA = [1.0, 2.0, 3.0, 4.0, 5.0]
    print(f"Input vector: {DATA}\n")
    res = run_benchmark(DATA, repeats=args.repeats, warmup=args.warmup,
                        memory_repeats=args.memory_repeats)
    print_results(res)
