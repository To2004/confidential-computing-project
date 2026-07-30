"""
Scaling benchmark — measures total HE operation time at different vector sizes.
Tests sizes: 5, 10, 50, 100, 500
Saves results to scaling_results.json for plotting.

Usage: python benchmark_scaling.py
"""

import json
import time
import tenseal as ts

try:
    from openfhe import CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature
    OPENFHE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    OPENFHE_AVAILABLE = False


# ── TenSEAL ──────────────────────────────────────────────────────────────────

def tenseal_total_time(vector: list[float]) -> float:
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40
    n = len(vector)

    t0 = time.perf_counter()
    enc = ts.ckks_vector(ctx, vector)
    enc_add = enc + enc
    enc_mul = enc * enc
    enc_sum = enc.sum()
    enc_avg = enc.sum() * (1.0 / n)
    _ = enc_add.decrypt()
    _ = enc_mul.decrypt()
    _ = enc_sum.decrypt()
    _ = enc_avg.decrypt()
    return time.perf_counter() - t0


# ── OpenFHE ───────────────────────────────────────────────────────────────────

def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def openfhe_total_time(vector: list[float]) -> float:
    n = len(vector)
    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(3)
    params.SetScalingModSize(50)
    params.SetBatchSize(_next_power_of_two(n))
    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    t0 = time.perf_counter()
    pt = cc.MakeCKKSPackedPlaintext(vector)
    ct = cc.Encrypt(keys.publicKey, pt)
    ct_add = cc.EvalAdd(ct, ct)
    ct_mul = cc.EvalMult(ct, ct)
    ct_sum = cc.EvalSum(ct, n)
    ct_avg = cc.EvalMult(cc.EvalSum(ct, n), 1.0 / n)

    def dec(c, length):
        p = cc.Decrypt(keys.secretKey, c)
        p.SetLength(length)
        return p.GetRealPackedValue()

    dec(ct_add, n); dec(ct_mul, n); dec(ct_sum, 1); dec(ct_avg, 1)
    return time.perf_counter() - t0


# ── Plaintext baseline ────────────────────────────────────────────────────────

def plaintext_total_time(vector: list[float]) -> float:
    n = len(vector)
    t0 = time.perf_counter()
    _ = [v + v for v in vector]
    _ = [v * v for v in vector]
    _ = sum(vector)
    _ = sum(vector) / n
    return time.perf_counter() - t0


# ── Main ──────────────────────────────────────────────────────────────────────

SIZES = [5, 10, 50, 100, 500]

if not OPENFHE_AVAILABLE:
    print("[NOTE] OpenFHE unavailable — only TenSEAL and plaintext will be measured.\n")

results = {
    "sizes": SIZES,
    "plaintext_ms": [],
    "tenseal_ms": [],
    "openfhe_ms": [],
    "openfhe_available": OPENFHE_AVAILABLE,
}

for size in SIZES:
    vec = [float(i + 1) for i in range(size)]
    print(f"Testing size = {size} ...", flush=True)

    pt_t = plaintext_total_time(vec) * 1000
    ts_t = tenseal_total_time(vec) * 1000
    fhe_t = openfhe_total_time(vec) * 1000 if OPENFHE_AVAILABLE else None

    results["plaintext_ms"].append(round(pt_t, 6))
    results["tenseal_ms"].append(round(ts_t, 3))
    results["openfhe_ms"].append(round(fhe_t, 3) if fhe_t is not None else None)

    print(f"  Plaintext : {pt_t:.4f} ms")
    print(f"  TenSEAL   : {ts_t:.2f} ms")
    if fhe_t is not None:
        print(f"  OpenFHE   : {fhe_t:.2f} ms")

print("\nScaling results:")
print(f"{'Size':>6} {'Plaintext (ms)':>16} {'TenSEAL (ms)':>14} {'OpenFHE (ms)':>14}")
print("-" * 55)
for i, size in enumerate(SIZES):
    fhe_s = f"{results['openfhe_ms'][i]:>14.2f}" if results["openfhe_ms"][i] is not None else "    unavailable"
    print(
        f"{size:>6} "
        f"{results['plaintext_ms'][i]:>16.4f} "
        f"{results['tenseal_ms'][i]:>14.2f} "
        f"{fhe_s}"
    )

with open("scaling_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to scaling_results.json")
