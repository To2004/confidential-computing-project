"""
Shared measurement harness for all benchmarks in this project: timing, memory,
summary statistics and drift diagnostics.

Timing and memory are measured in separate passes; `tracemalloc` hooks every
Python allocation and distorts short timings, so it is never active while the
clock is running.
"""

import gc
import os
import platform
import statistics
import sys
import time
import tracemalloc
import warnings

# The OpenFHE wheel prints a build-support banner the first time it is imported.
# It is not actionable here and would interleave with the benchmark tables.
# Every script imports this module first, so silencing it once covers all of them.
warnings.filterwarnings("ignore", message=".*Deprecation notice.*",
                        category=UserWarning)

# The input vector shared by the primitive-operation benchmarks, so the
# plaintext, TenSEAL and OpenFHE columns are all measured on the same data.
DEFAULT_VECTOR = [1.0, 2.0, 3.0, 4.0, 5.0]

# Defaults used by every benchmark unless overridden on the command line.
DEFAULT_REPEATS = 1000
DEFAULT_WARMUP = 50
DEFAULT_MEMORY_REPEATS = 5

# Live objects used to measure retained native memory: enough that RSS growth
# dwarfs allocator noise, small enough to stay inside RAM.
DEFAULT_RETAIN_COUNT = 50

# |first-half vs second-half| difference above which a measurement is flagged.
DRIFT_WARN_PCT = 5.0

# Fixed seed so the bootstrap interval is reproducible run to run.
BOOTSTRAP_SEED = 20260730
BOOTSTRAP_RESAMPLES = 2000

_Z95 = 1.959963984540054  # two-sided 95% normal quantile

SRC_ROOT = os.path.dirname(os.path.abspath(__file__))


# ── public API ───────────────────────────────────────────────────────────────

def summarize(samples_s, warmup=0, bootstrap=True):
    """Summary statistics for per-call durations in seconds. All outputs in ms."""
    if not samples_s:
        raise ValueError("summarize() requires at least one sample")

    ordered_ms = sorted(s * 1000.0 for s in samples_s)
    n = len(ordered_ms)
    mean_ms = statistics.fmean(ordered_ms)
    # stdev is undefined for a single sample; report 0.0 so callers can format it.
    std_ms = statistics.stdev(ordered_ms) if n > 1 else 0.0
    median_ms = statistics.median(ordered_ms)

    stats = {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "trimmed_mean_ms": _trimmed_mean(ordered_ms, 0.10),
        "mad_ms": _median_absolute_deviation(ordered_ms, median_ms),
        "iqr_ms": _percentile(ordered_ms, 75.0) - _percentile(ordered_ms, 25.0),
        "min_ms": ordered_ms[0],
        "max_ms": ordered_ms[-1],
        "p95_ms": _percentile(ordered_ms, 95.0),
        "p99_ms": _percentile(ordered_ms, 99.0),
        # Assumes independent samples, which back-to-back timings are not.
        "ci95_iid_half_width_ms": _Z95 * std_ms / (n ** 0.5) if n > 1 else 0.0,
        # Placeholder: time_operation overwrites this with the batch-means
        # interval, which it can compute because it has the acquisition order.
        "ci95_half_width_ms": _Z95 * std_ms / (n ** 0.5) if n > 1 else 0.0,
        "ci95_method": "iid (no acquisition order available)",
        "rel_std_pct": (std_ms / mean_ms * 100.0) if mean_ms > 0 else 0.0,
        # How far the mean sits above the median, relative to the median.
        "mean_over_median_pct": ((mean_ms - median_ms) / median_ms * 100.0)
                                if median_ms > 0 else 0.0,
        "outlier_fraction_pct": _outlier_fraction_pct(ordered_ms),
        "repeats": n,
        "warmup": warmup,
    }

    if bootstrap and n > 1:
        low, high = _bootstrap_median_ci(ordered_ms)
        stats["median_ci95_low_ms"] = low
        stats["median_ci95_high_ms"] = high

    return stats


def time_operation(fn, repeats=DEFAULT_REPEATS, warmup=DEFAULT_WARMUP,
                   keep_samples=True):
    """Time `fn` over `repeats` calls after `warmup` discarded calls.

    GC is disabled during the measured loop (as `timeit` does) so a collection
    triggered by unrelated allocations is not charged to the operation.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    for _ in range(warmup):
        fn()

    samples_s = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples_s.append(time.perf_counter() - start)
    finally:
        if gc_was_enabled:
            gc.enable()

    stats = summarize(samples_s, warmup=warmup)
    # Needs samples in acquisition order, which summarize() sorts away.
    stats.update(_drift_diagnostics(samples_s))

    # Prefer the dependence-aware interval; ci95_method records the fallback.
    batched = _batch_means_half_width_ms(samples_s)
    if batched is None:
        stats["ci95_half_width_ms"] = stats["ci95_iid_half_width_ms"]
        stats["ci95_method"] = "iid (too few samples to batch)"
    else:
        stats["ci95_half_width_ms"] = batched
        stats["ci95_method"] = "batch means (25 batches)"
        iid = stats["ci95_iid_half_width_ms"]
        stats["ci95_inflation_vs_iid"] = (batched / iid) if iid > 0 else None
    if keep_samples:
        stats["samples_ms"] = [round(s * 1000.0, 6) for s in samples_s]
    return stats


def measure_peak_memory_kb(fn, repeats=DEFAULT_MEMORY_REPEATS):
    """Peak Python-side memory of `fn`, in KB.

    tracemalloc sees only Python allocations; TenSEAL/OpenFHE keep ciphertexts
    in native heaps, so this is a lower bound. Use `measure_retained_bytes` for
    the native cost.
    """
    fn()  # discarded warm-up call
    peaks_kb = []
    for _ in range(repeats):
        tracemalloc.start()
        fn()
        peaks_kb.append(tracemalloc.get_traced_memory()[1] / 1024.0)
        tracemalloc.stop()
    return statistics.fmean(peaks_kb)


def measure_retained_bytes(factory, count=DEFAULT_RETAIN_COUNT):
    """Average resident bytes retained per object produced by `factory`.

    Builds `count` objects, keeps them alive, and measures RSS growth, so
    native allocations are included. Returns None without /proc (Linux only).
    """
    rss_before = _current_rss_bytes()
    if rss_before is None:
        return None

    # Settle the allocator first: the first object can trigger arena growth
    # that would otherwise be charged to it.
    warm = [factory() for _ in range(min(5, count))]
    del warm
    gc.collect()

    rss_before = _current_rss_bytes()
    held = []
    try:
        for _ in range(count):
            held.append(factory())
        rss_after = _current_rss_bytes()
    finally:
        del held
        gc.collect()

    grown = rss_after - rss_before
    # Allocator noise can make a tiny measurement come out negative.
    return max(0.0, grown / count)


def benchmark_operations(operations, repeats=DEFAULT_REPEATS, warmup=DEFAULT_WARMUP,
                         memory_repeats=DEFAULT_MEMORY_REPEATS, label="",
                         verbose=True, keep_samples=True):
    """Run `time_operation` + `measure_peak_memory_kb` over a dict of operations.

    `operations` maps a short key (e.g. "add") to a zero-argument callable.
    The returned dict carries, for every key:
        <key>_time_s      mean seconds (flat key used by the plotting scripts)
        <key>_mem_kb      mean peak Python-side KB
        <key>_time_stats  full statistics from `summarize`
    """
    results = {}
    drifting = []
    for key, fn in operations.items():
        if verbose:
            prefix = f"[{label}] " if label else ""
            print(f"  {prefix}{key:<12} timing {repeats} reps "
                  f"(+{warmup} warm-up)...", end="", flush=True)

        stats = time_operation(fn, repeats=repeats, warmup=warmup,
                               keep_samples=keep_samples)
        results[f"{key}_time_stats"] = stats
        results[f"{key}_time_s"] = stats["mean_ms"] / 1000.0
        results[f"{key}_mem_kb"] = measure_peak_memory_kb(fn, repeats=memory_repeats)

        if stats.get("drift_flagged"):
            drifting.append(key)

        if verbose:
            flag = "  <-- DRIFT" if stats.get("drift_flagged") else ""
            print(f" mean {stats['mean_ms']:9.4f} ms "
                  f"± {stats['ci95_half_width_ms']:.4f} "
                  f"(median {stats['median_ms']:.4f}, "
                  f"{stats['rel_std_pct']:.1f}% rel. sd){flag}", flush=True)

    results["repeats"] = repeats
    results["warmup"] = warmup
    if drifting:
        results["drifting_operations"] = drifting
        if verbose:
            print(f"  [!] first/second-half means differ by more than "
                  f"{DRIFT_WARN_PCT}% for: {', '.join(drifting)}")
            print("      Treat those means as contaminated by machine load, "
                  "not as steady-state cost.")
    return results


class AccuracyError(AssertionError):
    """Raised when a decrypted result is further from the reference than allowed."""


def assert_accuracy(results, thresholds, label=""):
    """Fail if any measured error exceeds its threshold.

    Thresholds are set orders of magnitude above the observed errors, so this
    catches broken crypto parameters, not ordinary CKKS noise.
    """
    failures = []
    for key, limit in thresholds.items():
        measured = results.get(key)
        if measured is None:
            failures.append(f"{key} was never measured")
        elif not measured <= limit:
            failures.append(f"{key} = {measured:.3e} exceeds limit {limit:.3e}")
    if failures:
        where = f" [{label}]" if label else ""
        raise AccuracyError(
            f"decrypted results are not accurate enough{where}: "
            + "; ".join(failures)
            + ". This usually means the crypto parameters cannot support the "
              "circuit (too few levels, or scale out of bounds)."
        )


def environment_info():
    """Machine, threading and library-version context for the results file.

    Thread settings are included because both libraries run multithreaded
    native code, so timings are only comparable at the same thread count.
    """
    info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "thread_env": {
            var: os.environ.get(var)
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }
    try:
        with open("/proc/loadavg") as handle:
            info["loadavg_at_start"] = handle.read().split()[:3]
    except OSError:
        pass
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    versions = {}
    for package in ("tenseal", "openfhe", "numpy", "matplotlib"):
        try:
            import importlib.metadata as md
            versions[package] = md.version(package)
        except Exception:
            try:
                versions[package] = __import__(package).__version__
            except Exception:
                versions[package] = None
    info["library_versions"] = versions

    for key, command in (("git_commit", "git rev-parse --short HEAD"),):
        try:
            import subprocess
            info[key] = subprocess.check_output(
                command.split(), cwd=os.path.dirname(SRC_ROOT) or ".",
                stderr=subprocess.DEVNULL, timeout=5).decode().strip()
        except Exception:
            info[key] = None
    return info


def format_stats_table(results, op_names, title):
    """Render the mean/median/sd/CI table used in the report."""
    lines = [
        "=" * 104,
        f"{title:^104}",
        "=" * 104,
        f"{'Operation':<14}{'Mean (ms)':>11}{'Median':>11}{'Trim mean':>11}"
        f"{'SD':>10}{'95% CI ±':>10}{'MAD':>10}{'p95':>10}{'p99':>10}{'Drift %':>9}",
        "-" * 104,
    ]
    for name, key in op_names:
        st = results.get(f"{key}_time_stats")
        if st is None:
            continue
        drift = st.get("drift_pct")
        drift_s = f"{drift:>8.1f}{'*' if st.get('drift_flagged') else ' '}" \
            if drift is not None else f"{'—':>9}"
        lines.append(
            f"{name:<14}{st['mean_ms']:>11.4f}{st['median_ms']:>11.4f}"
            f"{st['trimmed_mean_ms']:>11.4f}{st['std_ms']:>10.4f}"
            f"{st['ci95_half_width_ms']:>10.4f}{st['mad_ms']:>10.4f}"
            f"{st['p95_ms']:>10.4f}{st['p99_ms']:>10.4f}{drift_s}"
        )
    lines.append("-" * 104)
    n = results.get("repeats", "?")
    w = results.get("warmup", "?")
    lines.append(f"{n} timed repetitions per operation, {w} discarded warm-up calls. "
                 f"Trim mean = 10% trimmed. Drift = (2nd half - 1st half) / 1st half;")
    lines.append(f"* marks |drift| > {DRIFT_WARN_PCT}%, meaning the samples are not "
                 f"steady state and the mean should not be trusted.")
    return "\n".join(lines)


def next_power_of_two(n):
    """Round `n` up to a power of two. OpenFHE requires power-of-two batch sizes."""
    p = 1
    while p < n:
        p <<= 1
    return p


def add_repeat_arguments(parser):
    """Attach the standard --repeats / --warmup / --memory-repeats flags."""
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help=f"timed repetitions per operation (default {DEFAULT_REPEATS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"discarded warm-up calls per operation (default {DEFAULT_WARMUP})")
    parser.add_argument("--memory-repeats", type=int, default=DEFAULT_MEMORY_REPEATS,
                        help=f"repetitions for the memory pass (default {DEFAULT_MEMORY_REPEATS})")
    parser.add_argument("--no-samples", action="store_true",
                        help="omit per-call samples from the JSON (smaller output)")
    return parser


# ── internals ────────────────────────────────────────────────────────────────

def _percentile(sorted_values, pct):
    """Linear-interpolation percentile over an already-sorted list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (rank - low) * (sorted_values[high] - sorted_values[low])


def _trimmed_mean(sorted_values, proportion):
    """Mean after discarding `proportion` of the samples from each end."""
    n = len(sorted_values)
    cut = int(n * proportion)
    kept = sorted_values[cut:n - cut] if n - 2 * cut > 0 else sorted_values
    return statistics.fmean(kept)


def _median_absolute_deviation(sorted_values, median):
    """Median of |x - median|: a spread measure that outliers cannot inflate."""
    if len(sorted_values) == 1:
        return 0.0
    return statistics.median([abs(v - median) for v in sorted_values])


def _outlier_fraction_pct(sorted_values):
    """Percentage of samples beyond 1.5 IQR above Q3.

    Upper tail only: a call cannot take less time than the work requires, so a
    low outlier is clock noise rather than a disturbed measurement.
    """
    if len(sorted_values) < 4:
        return 0.0
    q1 = _percentile(sorted_values, 25.0)
    q3 = _percentile(sorted_values, 75.0)
    threshold = q3 + 1.5 * (q3 - q1)
    return sum(1 for v in sorted_values if v > threshold) / len(sorted_values) * 100.0


def _bootstrap_median_ci(sorted_values):
    """Percentile bootstrap 95% interval for the median. Seeded, so reproducible."""
    import random

    rng = random.Random(BOOTSTRAP_SEED)
    n = len(sorted_values)
    medians = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        resample = [sorted_values[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(resample))
    medians.sort()
    return _percentile(medians, 2.5), _percentile(medians, 97.5)


def _lag1_autocorrelation(values):
    """Lag-1 autocorrelation of the samples in acquisition order."""
    n = len(values)
    if n < 3:
        return 0.0
    mean = statistics.fmean(values)
    centred = [v - mean for v in values]
    denominator = sum(c * c for c in centred)
    if denominator <= 0:
        return 0.0
    numerator = sum(centred[i] * centred[i + 1] for i in range(n - 1))
    return numerator / denominator


def _batch_means_half_width_ms(samples_s, n_batches=25):
    """95% half-width for the mean using non-overlapping batch means.

    Consecutive timings are autocorrelated (shared cache, CPU frequency and
    scheduler state), so s/sqrt(n) is far too narrow. The means of contiguous
    blocks are much closer to independent. Returns None if n < 4 * n_batches.
    """
    n = len(samples_s)
    if n < n_batches * 4:
        return None
    size = n // n_batches
    means = [statistics.fmean(samples_s[i * size:(i + 1) * size]) * 1000.0
             for i in range(n_batches)]
    if len(means) < 2:
        return None
    return _Z95 * statistics.stdev(means) / (len(means) ** 0.5)


def _mann_kendall_z(values):
    """Normal-approximation Mann-Kendall statistic for a monotonic trend.

    Rank-based, so it catches a slow ramp (thermal throttling, allocator
    growth) that the halves comparison misses. |z| > 1.96 is the two-sided 5%
    level.
    """
    n = len(values)
    if n < 10:
        return 0.0
    # This loop is O(m^2), so subsample to cap m; a monotonic trend survives.
    step = max(1, n // 300)
    sub = values[::step]
    m = len(sub)
    s = 0
    for i in range(m - 1):
        vi = sub[i]
        for j in range(i + 1, m):
            if sub[j] > vi:
                s += 1
            elif sub[j] < vi:
                s -= 1
    variance = m * (m - 1) * (2 * m + 5) / 18.0
    if variance <= 0:
        return 0.0
    if s > 0:
        return (s - 1) / (variance ** 0.5)
    if s < 0:
        return (s + 1) / (variance ** 0.5)
    return 0.0


def _drift_diagnostics(samples_s):
    """Stationarity and dependence diagnostics; samples must be in acquisition order.

    Reports three things: the first-half vs second-half gap (a step change
    near the midpoint), the Mann-Kendall trend statistic (a monotonic ramp,
    which the halves gap misses), and the lag-1 autocorrelation. Flagged if
    the halves gap exceeds DRIFT_WARN_PCT or the trend test is significant.
    """
    n = len(samples_s)
    if n < 4:
        return {"drift_pct": None, "drift_flagged": False}

    half = n // 2
    first = statistics.fmean(samples_s[:half])
    second = statistics.fmean(samples_s[half:])
    if first <= 0:
        return {"drift_pct": None, "drift_flagged": False}

    drift_pct = (second - first) / first * 100.0
    trend_z = _mann_kendall_z(samples_s)
    acf1 = _lag1_autocorrelation(samples_s)

    return {
        "first_half_mean_ms": first * 1000.0,
        "second_half_mean_ms": second * 1000.0,
        "drift_pct": drift_pct,
        "halves_flagged": abs(drift_pct) > DRIFT_WARN_PCT,
        "trend_z": trend_z,
        "trend_flagged": abs(trend_z) > 1.96,
        "lag1_autocorrelation": acf1,
        "drift_flagged": abs(drift_pct) > DRIFT_WARN_PCT or abs(trend_z) > 1.96,
    }


def _current_rss_bytes():
    """Resident set size of this process, or None if /proc is unavailable."""
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return pages * os.sysconf("SC_PAGE_SIZE")
