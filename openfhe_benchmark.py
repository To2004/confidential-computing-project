"""
OpenFHE Benchmark — CKKS scheme
Implements all mandatory operations from the project proposal.
Install: pip install openfhe
"""

import time
import tracemalloc
from openfhe import (
    CCParamsCKKSRNS,
    GenCryptoContext,
    PKESchemeFeature,
)


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

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    return cc, keys


def _tick():
    return time.perf_counter()


def run_benchmark(vector: list[float]) -> dict:
    results = {}
    n = len(vector)
    cc, keys = make_context(batch_size=n)

    # --- Encryption ---
    tracemalloc.start()
    t0 = _tick()
    pt = cc.MakeCKKSPackedPlaintext(vector)
    ct = cc.Encrypt(keys.publicKey, pt)
    results["encrypt_time_s"] = _tick() - t0
    results["encrypt_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    # --- Encrypted Addition (ct + ct) ---
    tracemalloc.start()
    t0 = _tick()
    ct_add = cc.EvalAdd(ct, ct)
    results["add_time_s"] = _tick() - t0
    results["add_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    # --- Encrypted Multiplication (ct * ct) ---
    tracemalloc.start()
    t0 = _tick()
    ct_mul = cc.EvalMult(ct, ct)
    results["mul_time_s"] = _tick() - t0
    results["mul_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    # --- Encrypted Summation (sum all slots) ---
    tracemalloc.start()
    t0 = _tick()
    ct_sum = cc.EvalSum(ct, n)
    results["sum_time_s"] = _tick() - t0
    results["sum_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    # --- Encrypted Average (sum / n) ---
    tracemalloc.start()
    t0 = _tick()
    ct_avg = cc.EvalMult(cc.EvalSum(ct, n), 1.0 / n)
    results["avg_time_s"] = _tick() - t0
    results["avg_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    # --- Encrypted Dot Product (ct · ct2) ---
    vector2 = [v * 0.5 + 1.0 for v in vector]
    pt2 = cc.MakeCKKSPackedPlaintext(vector2)
    ct2 = cc.Encrypt(keys.publicKey, pt2)
    tracemalloc.start()
    t0 = _tick()
    ct_dot = cc.EvalInnerProduct(ct, ct2, n)
    results["dot_time_s"] = _tick() - t0
    results["dot_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    # --- Decryption ---
    def decrypt_vec(ct_val, length):
        pt_out = cc.Decrypt(keys.secretKey, ct_val)
        pt_out.SetLength(length)
        return pt_out.GetRealPackedValue()

    t0 = _tick()
    dec_add = decrypt_vec(ct_add, n)
    dec_mul = decrypt_vec(ct_mul, n)
    dec_sum_vec = decrypt_vec(ct_sum, 1)
    dec_avg_vec = decrypt_vec(ct_avg, 1)
    dec_dot_vec = decrypt_vec(ct_dot, 1)
    results["decrypt_time_s"] = _tick() - t0

    dec_sum = dec_sum_vec[0]
    dec_avg = dec_avg_vec[0]
    dec_dot = dec_dot_vec[0]

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

    return results


def print_results(results: dict):
    print("=" * 50)
    print("OpenFHE (CKKS) Benchmark Results")
    print("=" * 50)
    rows = [
        ("Encrypt",     results["encrypt_time_s"],  results["encrypt_mem_kb"]),
        ("Add",         results["add_time_s"],       results["add_mem_kb"]),
        ("Multiply",    results["mul_time_s"],       results["mul_mem_kb"]),
        ("Sum",         results["sum_time_s"],       results["sum_mem_kb"]),
        ("Average",     results["avg_time_s"],       results["avg_mem_kb"]),
        ("Dot Product", results["dot_time_s"],       results["dot_mem_kb"]),
        ("Decrypt",     results["decrypt_time_s"],   None),
    ]
    print(f"{'Operation':<14} {'Time (ms)':>10} {'Peak Mem (KB)':>14}")
    print("-" * 42)
    for name, t, m in rows:
        mem_str = f"{m:>13.2f}" if m is not None else "            -"
        print(f"{name:<14} {t*1000:>10.3f} {mem_str}")
    print("-" * 42)
    print(f"Add  max error  : {results['add_max_error']:.2e}")
    print(f"Mul  max error  : {results['mul_max_error']:.2e}")
    print(f"Sum  error      : {results['sum_error']:.2e}")
    print(f"Avg  error      : {results['avg_error']:.2e}")
    print(f"Dot  error      : {results['dot_error']:.2e}")


if __name__ == "__main__":
    DATA = [1.0, 2.0, 3.0, 4.0, 5.0]
    print(f"Input vector: {DATA}\n")
    res = run_benchmark(DATA)
    print_results(res)
