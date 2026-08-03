"""
Generate charts from benchmark results.

Reads:  results/*.json    (written by the benchmark scripts)
Saves:  figures/chart_*.png
"""

import json
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import project_paths

# Colourblind-safe categorical palette.
C_TS = "#2a78d6"   # TenSEAL
C_FHE = "#eb6834"  # OpenFHE
C_PT = "#1baf7a"   # Plaintext

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dedddA"

OPS = ["Add", "Multiply", "Sum", "Average", "Dot Product"]
OP_KEYS = ["add", "mul", "sum", "avg", "dot"]
ALL_OPS = ["Encrypt", "Add", "Multiply", "Sum", "Average", "Dot Product", "Decrypt"]
ALL_OP_KEYS = ["encrypt", "add", "mul", "sum", "avg", "dot", "decrypt"]

BAR_W = 0.38
GAP = 0.02  # gap between adjacent bars


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
    })


def load(name, required=True):
    """Load a results JSON by bare filename from results/."""
    path = project_paths.result_path(name)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"{path} not found.")
        return None
    with open(path) as handle:
        return json.load(handle)


def means_and_ci(results, keys):
    """Mean and 95% CI half-width in ms for a list of operation keys."""
    means, errs = [], []
    for key in keys:
        stats = results.get(f"{key}_time_stats")
        if stats is None:
            # Older results files store only the mean, with no CI.
            means.append(results[f"{key}_time_s"] * 1000)
            errs.append(0.0)
        else:
            means.append(stats["mean_ms"])
            errs.append(stats["ci95_half_width_ms"])
    return np.array(means), np.array(errs)


def label_bars(ax, bars, values, fmt="{:.2f}", errs=None):
    """Write value labels above bars; errs offsets the label above the error cap."""
    for i, (bar, value) in enumerate(zip(bars, values)):
        top = bar.get_height()
        if errs is not None:
            top += errs[i]
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, top),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=7.5, color=INK_SOFT,
        )


def repeat_note(data):
    reps = data.get("repeats")
    warm = data.get("warmup")
    if reps is None:
        return ""
    return (f"Mean of {reps:,} timed repetitions per operation "
            f"({warm} warm-up calls discarded); bars show 95% CI.")


# ── Chart 1: Operation timing + overhead ─────────────────────────────────────

def chart_operations(data):
    ts = data["tenseal"]
    fhe = data.get("openfhe")
    pt = data.get("plaintext")
    fhe_ok = fhe is not None

    ts_mean, ts_err = means_and_ci(ts, ALL_OP_KEYS)
    x = np.arange(len(ALL_OPS))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    ax = axes[0]
    bars1 = ax.bar(x - (BAR_W + GAP) / 2 if fhe_ok else x, ts_mean,
                   BAR_W if fhe_ok else 0.55, yerr=ts_err, capsize=2.5,
                   label="TenSEAL", color=C_TS,
                   error_kw=dict(ecolor=INK_SOFT, lw=0.8))
    label_bars(ax, bars1, ts_mean, errs=ts_err)
    if fhe_ok:
        fhe_mean, fhe_err = means_and_ci(fhe, ALL_OP_KEYS)
        bars2 = ax.bar(x + (BAR_W + GAP) / 2, fhe_mean, BAR_W, yerr=fhe_err,
                       capsize=2.5, label="OpenFHE", color=C_FHE,
                       error_kw=dict(ecolor=INK_SOFT, lw=0.8))
        label_bars(ax, bars2, fhe_mean, errs=fhe_err)

    ax.set_ylabel("Mean time (ms) — log scale")
    ax.set_title("Encrypted operation time" + (": TenSEAL vs OpenFHE" if fhe_ok else ""))
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_OPS, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="major", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    # Right panel: overhead vs plaintext, for operations present in both.
    ax = axes[1]
    pt_mean, _ = means_and_ci(pt, OP_KEYS)
    ts_cmp, _ = means_and_ci(ts, OP_KEYS)
    xo = np.arange(len(OPS))
    ts_over = ts_cmp / pt_mean

    bars1 = ax.bar(xo - (BAR_W + GAP) / 2 if fhe_ok else xo, ts_over,
                   BAR_W if fhe_ok else 0.55, label="TenSEAL", color=C_TS)
    label_bars(ax, bars1, ts_over, fmt="{:,.0f}x")
    if fhe_ok:
        fhe_cmp, _ = means_and_ci(fhe, OP_KEYS)
        fhe_over = fhe_cmp / pt_mean
        bars2 = ax.bar(xo + (BAR_W + GAP) / 2, fhe_over, BAR_W,
                       label="OpenFHE", color=C_FHE)
        label_bars(ax, bars2, fhe_over, fmt="{:,.0f}x")

    ax.set_ylabel("Slowdown vs plaintext (x) — log scale")
    ax.set_title("Cost of encryption per operation")
    ax.set_xticks(xo)
    ax.set_xticklabels(OPS, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="major", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    fig.suptitle("CKKS operation cost", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.005, repeat_note(data), ha="center", fontsize=8, color=INK_SOFT)
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.savefig(project_paths.figure_path("chart_operations.png"), dpi=150)
    plt.close()
    print("Saved figures/chart_operations.png")


# ── Chart 2: Memory usage ────────────────────────────────────────────────────

def chart_memory(data):
    ts = data["tenseal"]
    fhe = data.get("openfhe")
    fhe_ok = fhe is not None

    ts_mem = [ts[f"{k}_mem_kb"] for k in OP_KEYS]
    x = np.arange(len(OPS))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    bars1 = ax.bar(x - (BAR_W + GAP) / 2 if fhe_ok else x, ts_mem,
                   BAR_W if fhe_ok else 0.55, label="TenSEAL", color=C_TS)
    label_bars(ax, bars1, ts_mem, fmt="{:.2f}")
    if fhe_ok:
        fhe_mem = [fhe[f"{k}_mem_kb"] for k in OP_KEYS]
        bars2 = ax.bar(x + (BAR_W + GAP) / 2, fhe_mem, BAR_W,
                       label="OpenFHE", color=C_FHE)
        label_bars(ax, bars2, fhe_mem, fmt="{:.2f}")

    ax.set_ylabel("Peak Python-side memory (KB)")
    ax.set_title("What tracemalloc sees — Python wrappers only")
    ax.set_xticks(x)
    ax.set_xticklabels(OPS, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend()

    # Right panel: real ciphertext size. tracemalloc misses the native heap, so
    # resident-set growth and serialized size are also plotted.
    ax = axes[1]
    plaintext_bytes = len(data.get("vector", [])) * 8 or None
    groups, colours, labels = [], [], []
    for key, colour, name in (("tenseal", C_TS, "TenSEAL"), ("openfhe", C_FHE, "OpenFHE")):
        node = data.get(key)
        if not node:
            continue
        resident = node.get("ciphertext_rss_bytes")
        serialized = node.get("ciphertext_bytes")
        traced = node.get("encrypt_mem_kb", 0) * 1024
        groups.append((traced, resident, serialized))
        colours.append(colour)
        labels.append(name)

    if groups:
        measures = ["tracemalloc\n(Python only)", "Resident\n(RSS)", "Serialized"]
        xm = np.arange(len(measures))
        width = 0.8 / len(groups)
        for i, (values, colour, name) in enumerate(zip(groups, colours, labels)):
            offset = (i - (len(groups) - 1) / 2) * (width + 0.01)
            heights = [v if v else np.nan for v in values]
            bars = ax.bar(xm + offset, heights, width, label=name, color=colour)
            label_bars(ax, bars, heights, fmt="{:,.0f}")
        if plaintext_bytes:
            ax.axhline(plaintext_bytes, color=C_PT, linewidth=1.6)
            ax.annotate(f"plaintext vector: {plaintext_bytes} bytes",
                        xy=(len(measures) - 0.5, plaintext_bytes), xytext=(0, 5),
                        textcoords="offset points", ha="right", fontsize=8, color=INK)
        ax.set_xticks(xm)
        ax.set_xticklabels(measures, fontsize=8.5)
        ax.set_ylabel("Bytes per ciphertext — log scale")
        ax.set_title("What a ciphertext actually costs")
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="major", alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left")

    fig.suptitle("Memory: the Python-visible figure is not the cost",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.005,
             "tracemalloc sees Python allocations only. Resident-set growth and serialized "
             "size agree within ~7% and are six orders of magnitude larger.",
             ha="center", fontsize=8, color=INK_SOFT)
    plt.tight_layout(rect=(0, 0.035, 1, 0.94))
    plt.savefig(project_paths.figure_path("chart_memory.png"), dpi=150)
    plt.close()
    print("Saved figures/chart_memory.png")


# ── Chart 3: Approximation errors ────────────────────────────────────────────

def chart_errors(data):
    ts = data["tenseal"]
    fhe = data.get("openfhe")
    fhe_ok = fhe is not None
    err_keys = ["add_max_error", "mul_max_error", "sum_error", "avg_error", "dot_error"]

    ts_errs = [ts[k] for k in err_keys]
    x = np.arange(len(OPS))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - (BAR_W + GAP) / 2 if fhe_ok else x, ts_errs,
                   BAR_W if fhe_ok else 0.55, label="TenSEAL", color=C_TS)
    label_bars(ax, bars1, ts_errs, fmt="{:.1e}")
    if fhe_ok:
        fhe_errs = [fhe[k] for k in err_keys]
        bars2 = ax.bar(x + (BAR_W + GAP) / 2, fhe_errs, BAR_W,
                       label="OpenFHE", color=C_FHE)
        label_bars(ax, bars2, fhe_errs, fmt="{:.1e}")

    ax.set_ylabel("Max absolute error — log scale")
    ax.set_title("CKKS approximation error per operation")
    ax.set_xticks(x)
    ax.set_xticklabels(OPS)
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="major", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(project_paths.figure_path("chart_errors.png"), dpi=150)
    plt.close()
    print("Saved figures/chart_errors.png")


# ── Chart 4: Scaling ─────────────────────────────────────────────────────────

def chart_scaling(sc):
    sizes = sc["sizes"]
    pt_ms = sc["plaintext_ms"]
    ts_ms = sc["tenseal_ms"]
    fhe_ms = sc.get("openfhe_ms", [])
    fhe_ok = sc.get("openfhe_available", False) and any(v is not None for v in fhe_ms)

    def ci(stats_list):
        if not stats_list or stats_list[0] is None:
            return None
        return [s["ci95_half_width_ms"] for s in stats_list]

    ts_err = ci(sc.get("tenseal_stats"))
    fhe_err = ci(sc.get("openfhe_stats"))
    pt_err = ci(sc.get("plaintext_stats"))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    ax = axes[0]
    ax.errorbar(sizes, ts_ms, yerr=ts_err, fmt="o-", color=C_TS, label="TenSEAL",
                linewidth=2, markersize=7, capsize=3, ecolor=INK_SOFT, elinewidth=0.8)
    if fhe_ok:
        ax.errorbar(sizes, fhe_ms, yerr=fhe_err, fmt="s-", color=C_FHE, label="OpenFHE",
                    linewidth=2, markersize=7, capsize=3, ecolor=INK_SOFT, elinewidth=0.8)
    ax.set_xlabel("Vector size (elements)")
    ax.set_ylabel("Mean pipeline time (ms)")
    ax.set_title("Encrypted pipeline scaling")
    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend()
    # Label the last point of each series only.
    for series, colour in [(ts_ms, C_TS)] + ([(fhe_ms, C_FHE)] if fhe_ok else []):
        ax.annotate(f"{series[-1]:,.0f} ms", xy=(sizes[-1], series[-1]),
                    xytext=(-6, 8), textcoords="offset points",
                    ha="right", fontsize=8, color=INK_SOFT)

    ax = axes[1]
    ax.errorbar(sizes, pt_ms, yerr=pt_err, fmt="^-", color=C_PT, label="Plaintext",
                linewidth=2, markersize=7, capsize=3, ecolor=INK_SOFT, elinewidth=0.8)
    ax.errorbar(sizes, ts_ms, yerr=ts_err, fmt="o-", color=C_TS, label="TenSEAL",
                linewidth=2, markersize=7, capsize=3, ecolor=INK_SOFT, elinewidth=0.8)
    if fhe_ok:
        ax.errorbar(sizes, fhe_ms, yerr=fhe_err, fmt="s-", color=C_FHE, label="OpenFHE",
                    linewidth=2, markersize=7, capsize=3, ecolor=INK_SOFT, elinewidth=0.8)
    ax.set_xlabel("Vector size (elements)")
    ax.set_ylabel("Mean pipeline time (ms) — log scale")
    ax.set_title("Plaintext vs encrypted")
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend()

    fig.suptitle("Performance scaling with vector size", fontsize=12.5,
                 fontweight="bold", color=INK)
    fig.text(0.5, 0.005, repeat_note(sc), ha="center", fontsize=8, color=INK_SOFT)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(project_paths.figure_path("chart_scaling.png"), dpi=150)
    plt.close()
    print("Saved figures/chart_scaling.png")


# ── Chart 5: Encrypted medical risk score ────────────────────────────────────

def chart_risk_score(rs):
    stage_labels = ["Encrypt\nfeatures", "Score\nevaluation", "Cohort\nmean", "Decrypt\nscores"]
    stage_keys = ["encrypt_features", "score_evaluation", "cohort_mean", "decrypt_scores"]
    libs = [(name, rs[name]) for name in ("tenseal", "openfhe") if name in rs]
    colours = {"tenseal": C_TS, "openfhe": C_FHE}
    display = {"tenseal": "TenSEAL", "openfhe": "OpenFHE"}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))

    # Panel 1: per-stage mean time
    ax = axes[0]
    x = np.arange(len(stage_labels))
    n_libs = len(libs)
    for i, (name, res) in enumerate(libs):
        offset = (i - (n_libs - 1) / 2) * (BAR_W + GAP)
        means, errs = means_and_ci(res, stage_keys)
        bars = ax.bar(x + offset, means, BAR_W, yerr=errs, capsize=2.5,
                      label=display[name], color=colours[name],
                      error_kw=dict(ecolor=INK_SOFT, lw=0.8))
        label_bars(ax, bars, means, fmt="{:.0f}", errs=errs)
    ax.set_ylabel("Mean time (ms) — log scale")
    ax.set_title("Pipeline stage cost")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=8.5)
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="major", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    # Panel 2: distribution of the plaintext reference scores
    ax = axes[1]
    try:
        import synthetic_patients as patients
        cohort = patients.generate_cohort(rs["n_patients"], seed=rs["seed"])
        scores = patients.plaintext_risk_scores(cohort)
        counts, _, _ = ax.hist(scores, bins=40, color=C_PT,
                               edgecolor="white", linewidth=0.5)
        mean_score = float(np.mean(scores))
        # Headroom so the annotation does not sit on a bar.
        ax.set_ylim(0, counts.max() * 1.18)
        ax.axvline(mean_score, color=INK, linewidth=1.4)
        ax.annotate(f"cohort mean {mean_score:.2f}",
                    xy=(mean_score, counts.max() * 1.08),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=INK)
        ax.set_xlabel("Synthetic risk score (0–100)")
        ax.set_ylabel(f"Patients (n = {rs['n_patients']:,})")
        ax.set_title("Cohort score distribution")
        ax.grid(True, axis="y", alpha=0.6)
        ax.set_axisbelow(True)
    except Exception as exc:  # plotting must not fail the whole chart run
        ax.text(0.5, 0.5, f"distribution unavailable\n({exc})", ha="center",
                va="center", transform=ax.transAxes, fontsize=8, color=INK_SOFT)

    # Panel 3: accuracy of the encrypted score
    ax = axes[2]
    metrics = [("Max abs\nerror", "score_max_abs_error"),
               ("Mean abs\nerror", "score_mean_abs_error"),
               ("Cohort mean\nabs error", "cohort_mean_abs_error")]
    xm = np.arange(len(metrics))
    for i, (name, res) in enumerate(libs):
        offset = (i - (n_libs - 1) / 2) * (BAR_W + GAP)
        values = [res[key] for _, key in metrics]
        bars = ax.bar(xm + offset, values, BAR_W, label=display[name], color=colours[name])
        label_bars(ax, bars, values, fmt="{:.1e}")
    ax.set_ylabel("Absolute error, score points — log scale")
    ax.set_title("Encrypted vs plaintext accuracy")
    ax.set_xticks(xm)
    ax.set_xticklabels([label for label, _ in metrics], fontsize=8.5)
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="major", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    fig.suptitle(f"Encrypted synthetic medical risk score — {rs['n_patients']:,} patients, "
                 "5 features, interaction and quadratic terms",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.005,
             "Synthetic score for demonstration only — not a validated clinical model.  "
             + repeat_note(rs),
             ha="center", fontsize=8, color=INK_SOFT)
    plt.tight_layout(rect=(0, 0.035, 1, 0.94))
    plt.savefig(project_paths.figure_path("chart_risk_score.png"), dpi=150)
    plt.close()
    print("Saved figures/chart_risk_score.png")


# ── Chart 6: Matched-parameter comparison ────────────────────────────────────

def chart_matched(mc):
    """Plot speed and error for OpenFHE arms against TenSEAL at matched parameters."""
    arms = mc["arms"]
    order = [k for k in ("tenseal", "openfhe_default") if k in arms] + \
            [k for k in arms if k.startswith("openfhe_matched")]

    def short(key):
        if key == "tenseal":
            return "TenSEAL\n(reference)"
        if key == "openfhe_default":
            return "OpenFHE\nlibrary-chosen"
        return "OpenFHE matched\n" + key.replace("openfhe_matched_", "").upper()

    labels = [short(k) for k in order]
    # TenSEAL first, then all OpenFHE arms in one colour.
    colours = [C_TS] + [C_FHE] * (len(order) - 1)

    mul_ms = [arms[k]["mul_time_stats"]["mean_ms"] for k in order]
    mul_ci = [arms[k]["mul_time_stats"]["ci95_half_width_ms"] for k in order]
    mul_err = [arms[k]["mul_max_error"] for k in order]
    rings = [arms[k]["ring_dimension"] for k in order]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    x = np.arange(len(order))

    ax = axes[0]
    bars = ax.bar(x, mul_ms, 0.6, yerr=mul_ci, capsize=3, color=colours,
                  error_kw=dict(ecolor=INK_SOFT, lw=0.8))
    label_bars(ax, bars, mul_ms, fmt="{:.2f}", errs=mul_ci)
    ax.set_ylabel("Ciphertext × ciphertext multiply (ms)")
    ax.set_title("Speed: most of the gap is the ring dimension")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    for i, ring in enumerate(rings):
        ax.annotate(f"n={ring:,}", xy=(i, 0), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color="white", fontweight="bold")

    ax = axes[1]
    bars = ax.bar(x, mul_err, 0.6, color=colours)
    label_bars(ax, bars, mul_err, fmt="{:.1e}")
    ax.set_ylabel("Max absolute error — log scale")
    ax.set_title("Precision: a gap survives matching")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="major", alpha=0.6)
    ax.set_axisbelow(True)

    fig.suptitle("OpenFHE placed on TenSEAL's parameters (ring 8192, scale $2^{40}$)",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.005,
             "Lower is better in both panels. Matching the ring dimension removes most of the "
             "speed gap; the precision gap does not come from the scaling factor alone.",
             ha="center", fontsize=8, color=INK_SOFT)
    plt.tight_layout(rect=(0, 0.035, 1, 0.94))
    plt.savefig(project_paths.figure_path("chart_matched_comparison.png"), dpi=150)
    plt.close()
    print("Saved figures/chart_matched_comparison.png")


# ── Chart 7: IND-CPA^D noise flooding ────────────────────────────────────────

def chart_ind_cpad(ic):
    """Plot the timing cost of noise flooding and the decryption spread per library."""
    ops = [("Encrypt", "encrypt"), ("Multiply", "mul"),
           ("Sum", "sum"), ("Decrypt", "decrypt")]
    base_ms = [ic["baseline"][f"{k}_time_stats"]["mean_ms"] for _, k in ops]
    base_ci = [ic["baseline"][f"{k}_time_stats"]["ci95_half_width_ms"] for _, k in ops]
    flood_ms = [ic["flooded"][f"{k}_time_stats"]["mean_ms"] for _, k in ops]
    flood_ci = [ic["flooded"][f"{k}_time_stats"]["ci95_half_width_ms"] for _, k in ops]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))

    ax = axes[0]
    x = np.arange(len(ops))
    b1 = ax.bar(x - (BAR_W + GAP) / 2, base_ms, BAR_W, yerr=base_ci, capsize=2.5,
                label="FIXED_NOISE_DECRYPT (default)", color=C_TS,
                error_kw=dict(ecolor=INK_SOFT, lw=0.8))
    b2 = ax.bar(x + (BAR_W + GAP) / 2, flood_ms, BAR_W, yerr=flood_ci, capsize=2.5,
                label="NOISE_FLOODING_DECRYPT", color=C_FHE,
                error_kw=dict(ecolor=INK_SOFT, lw=0.8))
    label_bars(ax, b1, base_ms, fmt="{:.1f}", errs=base_ci)
    label_bars(ax, b2, flood_ms, fmt="{:.1f}", errs=flood_ci)
    ax.set_ylabel("Mean time (ms) — log scale")
    ax.set_title("Flooding costs decryption, and only decryption")
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _ in ops])
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="major", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8)
    ratio = flood_ms[-1] / base_ms[-1]
    ax.annotate(f"{ratio:.2f}×", xy=(x[-1] + (BAR_W + GAP) / 2, flood_ms[-1]),
                xytext=(14, 6), textcoords="offset points", fontsize=10,
                fontweight="bold", color=INK)

    # Right panel: decryption spread. Zero cannot be drawn on a log axis, so a
    # zero spread is drawn at an artificial floor.
    ax = axes[1]
    spreads = [
        ("TenSEAL\n(no setting)", ic["tenseal_decryption_spread"]["max_range_across_slots"], C_TS),
        ("OpenFHE\ndefault", ic["baseline"]["decryption_spread"]["max_range_across_slots"], C_FHE),
        ("OpenFHE\nflooding", ic["flooded"]["decryption_spread"]["max_range_across_slots"], C_FHE),
    ]
    nonzero = [v for _, v, _ in spreads if v > 0]
    floor = min(nonzero) / 100 if nonzero else 1e-16
    heights = [v if v > 0 else floor for _, v, _ in spreads]
    xs = np.arange(len(spreads))
    ax.bar(xs, heights, 0.55, color=[c for _, _, c in spreads])
    # Not label_bars: a floored bar must be labelled 0, not its drawn height.
    for i, (_, value, _) in enumerate(spreads):
        text = "0 — deterministic" if value == 0 else f"{value:.1e}"
        ax.annotate(text, xy=(i, heights[i]), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=INK_SOFT,
                    fontweight="bold" if value == 0 else "normal")
    ax.set_ylabel("Spread across repeated decryptions — log scale")
    ax.set_title("Is the released value randomized?")
    ax.set_xticks(xs)
    ax.set_xticklabels([n for n, _, _ in spreads])
    ax.set_yscale("log")
    ax.set_ylim(floor / 3, max(heights) * 30)
    ax.grid(True, axis="y", which="major", alpha=0.6)
    ax.set_axisbelow(True)

    fig.suptitle("IND-CPA$^D$: releasing decrypted values (Li & Micciancio, EUROCRYPT 2021)",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.005,
             "A deterministic decryption leaks the key-dependent error the attack exploits. "
             "Randomization shows a defence is active — not that it is sufficient: OpenFHE's "
             "noise estimate is average-case, which Guo et al. (USENIX Sec. 2024) attack.",
             ha="center", fontsize=8, color=INK_SOFT)
    plt.tight_layout(rect=(0, 0.05, 1, 0.94))
    plt.savefig(project_paths.figure_path("chart_ind_cpad.png"), dpi=150)
    plt.close()
    print("Saved figures/chart_ind_cpad.png")


def main():
    apply_style()

    data = load("results.json")
    chart_operations(data)
    chart_memory(data)
    chart_errors(data)

    chart_scaling(load("scaling_results.json"))

    optional = [
        ("risk_score_results.json", chart_risk_score, "risk_score_benchmark.py"),
        ("matched_comparison_results.json", chart_matched, "matched_comparison.py"),
        ("ind_cpad_results.json", chart_ind_cpad, "ind_cpad_flooding.py"),
    ]
    for path, drawer, producer in optional:
        payload = load(path, required=False)
        if payload is None:
            print(f"{path} not found — skipping its chart (run {producer} first)")
        else:
            drawer(payload)

    print("\nAll charts saved.")


if __name__ == "__main__":
    main()
