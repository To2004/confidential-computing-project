"""
Run all three benchmarks and print a side-by-side comparison table.
Saves results to results.json for use by plot_results.py.
Usage: python run_comparison.py
"""

import json
import sys
import plaintext_baseline as pt_bench
import tenseal_benchmark   as ts_bench

try:
    import openfhe_benchmark as fhe_bench
    OPENFHE_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    OPENFHE_AVAILABLE = False
    OPENFHE_ERROR = str(e)

DATA = [1.0, 2.0, 3.0, 4.0, 5.0]

print(f"Input vector : {DATA}")
print(f"Vector length: {len(DATA)}\n")

if not OPENFHE_AVAILABLE:
    print(f"[NOTE] OpenFHE unavailable on this platform: {OPENFHE_ERROR}")
    print("       OpenFHE Python bindings are Linux-only (no Windows wheel).")
    print("       This is documented as a practical limitation in the project.\n")

# --- Plaintext ---
print("Running plaintext baseline...")
pt = pt_bench.run_benchmark(DATA)

# --- TenSEAL ---
print("Running TenSEAL benchmark...")
ts = ts_bench.run_benchmark(DATA)

# --- OpenFHE ---
if OPENFHE_AVAILABLE:
    print("Running OpenFHE benchmark...")
    fhe = fhe_bench.run_benchmark(DATA)
else:
    fhe = None

# ── Timing comparison table ──────────────────────────────────────────────────
print("\n")
W = 80
print("=" * W)
print(f"{'Timing Comparison (ms)':^{W}}")
print("=" * W)
fhe_header = "OpenFHE" if OPENFHE_AVAILABLE else "OpenFHE (N/A)"
print(f"{'Operation':<16} {'Plaintext':>12} {'TenSEAL':>12} {fhe_header:>16}")
print("-" * W)

pt_total_ms = pt["total_time_s"] * 1000

timing_rows = [
    ("Encrypt",     None,         ts["encrypt_time_s"]*1000, fhe["encrypt_time_s"]*1000 if fhe else None),
    ("Add",         None,         ts["add_time_s"]*1000,     fhe["add_time_s"]*1000     if fhe else None),
    ("Multiply",    None,         ts["mul_time_s"]*1000,     fhe["mul_time_s"]*1000     if fhe else None),
    ("Sum",         None,         ts["sum_time_s"]*1000,     fhe["sum_time_s"]*1000     if fhe else None),
    ("Average",     None,         ts["avg_time_s"]*1000,     fhe["avg_time_s"]*1000     if fhe else None),
    ("Dot Product", None,         ts["dot_time_s"]*1000,     fhe["dot_time_s"]*1000     if fhe else None),
    ("Decrypt",     None,         ts["decrypt_time_s"]*1000, fhe["decrypt_time_s"]*1000 if fhe else None),
    ("ALL (plain)", pt_total_ms,  None,                      None),
]

for name, p, t, f in timing_rows:
    p_s = f"{p:>11.4f}" if p is not None else "           -"
    t_s = f"{t:>11.3f}" if t is not None else "           -"
    f_s = f"{f:>15.3f}" if f is not None else "     unavailable"
    print(f"{name:<16} {p_s} {t_s} {f_s}")

print("-" * W)

ts_total_ms = sum([
    ts["encrypt_time_s"], ts["add_time_s"], ts["mul_time_s"],
    ts["sum_time_s"], ts["avg_time_s"], ts["dot_time_s"], ts["decrypt_time_s"],
]) * 1000

if fhe:
    fhe_total_ms = sum([
        fhe["encrypt_time_s"], fhe["add_time_s"], fhe["mul_time_s"],
        fhe["sum_time_s"], fhe["avg_time_s"], fhe["dot_time_s"], fhe["decrypt_time_s"],
    ]) * 1000
    fhe_total_s = f"{fhe_total_ms:>15.3f}"
    fhe_overhead = f"{fhe_total_ms/pt_total_ms:>15.0f}x"
else:
    fhe_total_ms = None
    fhe_total_s  = "     unavailable"
    fhe_overhead = "     unavailable"

print(f"{'Total (HE ops)':<16} {pt_total_ms:>11.4f} {ts_total_ms:>11.3f} {fhe_total_s}")
print(f"{'Overhead vs pt':<16} {'1x':>12} {ts_total_ms/pt_total_ms:>11.0f}x {fhe_overhead}")

# ── Memory comparison table ───────────────────────────────────────────────────
print("\n")
print("=" * W)
print(f"{'Memory Usage Comparison (KB peak)':^{W}}")
print("=" * W)
print(f"{'Operation':<16} {'Plaintext':>12} {'TenSEAL':>12} {fhe_header:>16}")
print("-" * W)

mem_rows = [
    ("Encrypt",     pt["peak_mem_kb"],  ts["encrypt_mem_kb"], fhe["encrypt_mem_kb"] if fhe else None),
    ("Add",         None,               ts["add_mem_kb"],     fhe["add_mem_kb"]     if fhe else None),
    ("Multiply",    None,               ts["mul_mem_kb"],     fhe["mul_mem_kb"]     if fhe else None),
    ("Sum",         None,               ts["sum_mem_kb"],     fhe["sum_mem_kb"]     if fhe else None),
    ("Average",     None,               ts["avg_mem_kb"],     fhe["avg_mem_kb"]     if fhe else None),
    ("Dot Product", None,               ts["dot_mem_kb"],     fhe["dot_mem_kb"]     if fhe else None),
]
for name, p, t, f in mem_rows:
    p_s = f"{p:>11.2f}" if p is not None else "           -"
    t_s = f"{t:>11.2f}" if t is not None else "           -"
    f_s = f"{f:>15.2f}" if f is not None else "     unavailable"
    print(f"{name:<16} {p_s} {t_s} {f_s}")

# ── Approximation error table ────────────────────────────────────────────────
print("\n")
print("=" * W)
print(f"{'Approximation Errors (max absolute error)':^{W}}")
print("=" * W)
print(f"{'Operation':<16} {'TenSEAL':>20} {fhe_header:>20}")
print("-" * W)
for name, tk, fk in [
    ("Add",         "add_max_error", "add_max_error"),
    ("Multiply",    "mul_max_error", "mul_max_error"),
    ("Sum",         "sum_error",     "sum_error"),
    ("Average",     "avg_error",     "avg_error"),
    ("Dot Product", "dot_error",     "dot_error"),
]:
    te = ts[tk]
    f_s = f"{fhe[fk]:>20.4e}" if fhe else "         unavailable"
    print(f"{name:<16} {te:>20.4e} {f_s}")

if "ciphertext_bytes" in ts:
    print(f"\nTenSEAL ciphertext size: {ts['ciphertext_bytes']} bytes")

# ── Save to JSON ──────────────────────────────────────────────────────────────
output = {
    "vector": DATA,
    "plaintext": pt,
    "tenseal": ts,
    "openfhe": fhe,
    "openfhe_available": OPENFHE_AVAILABLE,
    "totals_ms": {
        "plaintext": pt_total_ms,
        "tenseal": ts_total_ms,
        "openfhe": fhe_total_ms,
    },
}
with open("results.json", "w") as f_out:
    json.dump(output, f_out, indent=2)

print("\nResults saved to results.json")
