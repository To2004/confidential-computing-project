"""
TenSEAL Benchmark — CKKS scheme
Implements all mandatory operations from the project proposal.
Install: pip install tenseal
"""

import time
import tracemalloc
import tenseal as ts


def make_context():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40
    return ctx


def _tick():
    return time.perf_counter()


def run_benchmark(vector: list[float]) -> dict:
    results = {}
    ctx = make_context()
    n = len(vector)

    # --- Encryption ---
    tracemalloc.start()
    t0 = _tick()
    enc = ts.ckks_vector(ctx, vector)
    results["encrypt_time_s"] = _tick() - t0
    results["encrypt_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    # --- Encrypted Addition (enc + enc) ---
    tracemalloc.start()
    t0 = _tick()
    enc_add = enc + enc
    results["add_time_s"] = _tick() - t0
    results["add_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    # --- Encrypted Multiplication (enc * enc) ---
    tracemalloc.start()
    t0 = _tick()
    enc_mul = enc * enc
    results["mul_time_s"] = _tick() - t0
    results["mul_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    # --- Encrypted Summation (sum of all slots) ---
    tracemalloc.start()
    t0 = _tick()
    enc_sum = enc.sum()
    results["sum_time_s"] = _tick() - t0
    results["sum_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    # --- Encrypted Average (sum / n, plaintext scalar division) ---
    tracemalloc.start()
    t0 = _tick()
    enc_avg = enc.sum() * (1.0 / n)
    results["avg_time_s"] = _tick() - t0
    results["avg_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    # --- Encrypted Dot Product (enc · enc2, element-wise mul then sum) ---
    vector2 = [v * 0.5 + 1.0 for v in vector]
    enc2 = ts.ckks_vector(ctx, vector2)
    tracemalloc.start()
    t0 = _tick()
    enc_dot = (enc * enc2).sum()
    results["dot_time_s"] = _tick() - t0
    results["dot_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    # --- Decryption ---
    t0 = _tick()
    dec_add = enc_add.decrypt()
    dec_mul = enc_mul.decrypt()
    dec_sum = enc_sum.decrypt()[0]
    dec_avg = enc_avg.decrypt()[0]
    dec_dot = enc_dot.decrypt()[0]
    results["decrypt_time_s"] = _tick() - t0

    # Correctness checks (approximate, CKKS is not exact)
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

    # Ciphertext size in bytes (serialized)
    results["ciphertext_bytes"] = len(enc.serialize())

    return results


def print_results(results: dict):
    print("=" * 50)
    print("TenSEAL (CKKS) Benchmark Results")
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
    print(f"Ciphertext size : {results['ciphertext_bytes']} bytes")
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
