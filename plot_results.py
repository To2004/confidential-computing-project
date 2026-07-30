"""
Generate charts from benchmark results.
Reads:  results.json  (from run_comparison.py)
        scaling_results.json  (from benchmark_scaling.py)
Saves:  chart_operations.png, chart_memory.png, chart_errors.png, chart_scaling.png
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path) as f:
        return json.load(f)


data = load("results.json")
ts   = data["tenseal"]
fhe  = data.get("openfhe")   # may be None
fhe_ok = fhe is not None

OPS      = ["Add", "Multiply", "Sum", "Average", "Dot Product"]
OP_KEYS  = ["add", "mul", "sum", "avg", "dot"]

C_TS  = "#2196F3"
C_FHE = "#FF5722"
C_PT  = "#4CAF50"


# ── Chart 1: Operation timing ─────────────────────────────────────────────────

ts_times = [ts[f"{k}_time_s"] * 1000 for k in OP_KEYS]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(OPS))
w = 0.35 if fhe_ok else 0.5

if fhe_ok:
    fhe_times = [fhe[f"{k}_time_s"] * 1000 for k in OP_KEYS]
    bars1 = ax.bar(x - w/2, ts_times,  w, label="TenSEAL", color=C_TS,  alpha=0.85)
    bars2 = ax.bar(x + w/2, fhe_times, w, label="OpenFHE", color=C_FHE, alpha=0.85)
    ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=8)
else:
    bars1 = ax.bar(x, ts_times, w, label="TenSEAL", color=C_TS, alpha=0.85)

ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=8)
ax.set_xlabel("Operation")
ax.set_ylabel("Time (ms)")
ax.set_title("Encrypted Operation Time: TenSEAL" + (" vs OpenFHE" if fhe_ok else " (OpenFHE: Linux only)"))
ax.set_xticks(x)
ax.set_xticklabels(OPS)
ax.legend()
if not fhe_ok:
    ax.text(0.99, 0.97, "OpenFHE requires Linux", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))
plt.tight_layout()
plt.savefig("chart_operations.png", dpi=150)
plt.close()
print("Saved chart_operations.png")


# ── Chart 2: Memory usage ─────────────────────────────────────────────────────

ts_mem = [ts[f"{k}_mem_kb"] for k in OP_KEYS]

fig, ax = plt.subplots(figsize=(10, 5))
if fhe_ok:
    fhe_mem = [fhe[f"{k}_mem_kb"] for k in OP_KEYS]
    bars1 = ax.bar(x - w/2, ts_mem,  w, label="TenSEAL", color=C_TS,  alpha=0.85)
    bars2 = ax.bar(x + w/2, fhe_mem, w, label="OpenFHE", color=C_FHE, alpha=0.85)
    ax.bar_label(bars2, fmt="%.1f", padding=2, fontsize=8)
else:
    bars1 = ax.bar(x, ts_mem, w, label="TenSEAL", color=C_TS, alpha=0.85)

ax.bar_label(bars1, fmt="%.1f", padding=2, fontsize=8)
ax.set_xlabel("Operation")
ax.set_ylabel("Peak Memory (KB)")
ax.set_title("Peak Memory Usage per Operation")
ax.set_xticks(x)
ax.set_xticklabels(OPS)
ax.legend()
plt.tight_layout()
plt.savefig("chart_memory.png", dpi=150)
plt.close()
print("Saved chart_memory.png")


# ── Chart 3: Approximation errors ────────────────────────────────────────────

ERR_KEYS = ["add_max_error", "mul_max_error", "sum_error", "avg_error", "dot_error"]
ts_errs  = [ts[k] for k in ERR_KEYS]
xe = np.arange(len(OPS))

fig, ax = plt.subplots(figsize=(10, 5))
if fhe_ok:
    fhe_errs = [fhe[k] for k in ERR_KEYS]
    bars1 = ax.bar(xe - w/2, ts_errs,  w, label="TenSEAL", color=C_TS,  alpha=0.85)
    bars2 = ax.bar(xe + w/2, fhe_errs, w, label="OpenFHE", color=C_FHE, alpha=0.85)
else:
    bars1 = ax.bar(xe, ts_errs, w, label="TenSEAL", color=C_TS, alpha=0.85)

ax.set_xlabel("Operation")
ax.set_ylabel("Max Absolute Error (log scale)")
ax.set_title("CKKS Approximation Error per Operation")
ax.set_xticks(xe)
ax.set_xticklabels(OPS)
ax.set_yscale("log")
ax.legend()
ax.grid(True, which="both", axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chart_errors.png", dpi=150)
plt.close()
print("Saved chart_errors.png")


# ── Chart 4: Scaling ─────────────────────────────────────────────────────────

sc = load("scaling_results.json")
sizes  = sc["sizes"]
pt_ms  = sc["plaintext_ms"]
ts_ms  = sc["tenseal_ms"]
fhe_ms = sc.get("openfhe_ms", [])
sc_fhe_ok = sc.get("openfhe_available", False) and any(v is not None for v in fhe_ms)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: HE libraries (plain is too fast to see on same scale)
axes[0].plot(sizes, ts_ms, "o-", color=C_TS, label="TenSEAL", linewidth=2, markersize=6)
if sc_fhe_ok:
    axes[0].plot(sizes, fhe_ms, "s-", color=C_FHE, label="OpenFHE", linewidth=2, markersize=6)
axes[0].set_xlabel("Vector Size")
axes[0].set_ylabel("Total Time (ms)")
axes[0].set_title("Scaling: TenSEAL" + (" vs OpenFHE" if sc_fhe_ok else ""))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: All on log scale — plaintext vs encrypted
axes[1].plot(sizes, pt_ms,  "^-", color=C_PT,  label="Plaintext", linewidth=2, markersize=6)
axes[1].plot(sizes, ts_ms,  "o-", color=C_TS,  label="TenSEAL",   linewidth=2, markersize=6)
if sc_fhe_ok:
    axes[1].plot(sizes, fhe_ms, "s-", color=C_FHE, label="OpenFHE", linewidth=2, markersize=6)
axes[1].set_xlabel("Vector Size")
axes[1].set_ylabel("Total Time (ms) — log scale")
axes[1].set_title("Plaintext vs Encrypted (log scale)")
axes[1].set_yscale("log")
axes[1].legend()
axes[1].grid(True, which="both", alpha=0.3)

plt.suptitle("Performance Scaling with Vector Size", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("chart_scaling.png", dpi=150)
plt.close()
print("Saved chart_scaling.png")

print("\nAll charts saved.")
