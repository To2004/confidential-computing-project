"""
Plaintext Baseline — the same operations without encryption.
Used as the comparison reference for overhead calculations.

Each operation is timed separately over many repetitions after a discarded
warm-up phase, so that plaintext and encrypted figures are directly comparable
per operation rather than only in aggregate.

Usage: python plaintext_baseline.py [--repeats 1000] [--warmup 50]
"""

import argparse

import benchmark_harness as harness

OP_NAMES = [
    ("Add", "add"),
    ("Multiply", "mul"),
    ("Sum", "sum"),
    ("Average", "avg"),
    ("Dot Product", "dot"),
]


def run_benchmark(vector, repeats=harness.DEFAULT_REPEATS, warmup=harness.DEFAULT_WARMUP,
                  memory_repeats=harness.DEFAULT_MEMORY_REPEATS, verbose=True):
    n = len(vector)
    vector2 = [v * 0.5 + 1.0 for v in vector]

    operations = {
        "add": lambda: [v + v for v in vector],
        "mul": lambda: [v * v for v in vector],
        "sum": lambda: sum(vector),
        "avg": lambda: sum(vector) / n,
        "dot": lambda: sum(a * b for a, b in zip(vector, vector2)),
    }

    results = harness.benchmark_operations(
        operations, repeats=repeats, warmup=warmup,
        memory_repeats=memory_repeats, label="Plaintext", verbose=verbose,
    )

    # Total across the five operations, the reference for overhead ratios.
    results["total_time_s"] = sum(results[f"{key}_time_s"] for _, key in OP_NAMES)
    results["peak_mem_kb"] = max(results[f"{key}_mem_kb"] for _, key in OP_NAMES)

    # Computed values, kept so the results can be eyeballed for sanity.
    results["add_result"] = operations["add"]()
    results["mul_result"] = operations["mul"]()
    results["sum_result"] = operations["sum"]()
    results["avg_result"] = operations["avg"]()
    results["dot_result"] = operations["dot"]()

    return results


def print_results(results):
    print()
    print(harness.format_stats_table(results, OP_NAMES, "Plaintext Baseline Results"))
    print()
    print(f"Total (5 ops) : {results['total_time_s'] * 1000:.6f} ms")
    print(f"Peak mem      : {results['peak_mem_kb']:.4f} KB")
    print(f"Add result    : {results['add_result']}")
    print(f"Mul result    : {results['mul_result']}")
    print(f"Sum result    : {results['sum_result']}")
    print(f"Avg result    : {results['avg_result']}")
    print(f"Dot result    : {results['dot_result']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plaintext baseline benchmark")
    harness.add_repeat_arguments(parser)
    args = parser.parse_args()

    DATA = [1.0, 2.0, 3.0, 4.0, 5.0]
    print(f"Input vector: {DATA}\n")
    res = run_benchmark(DATA, repeats=args.repeats, warmup=args.warmup,
                        memory_repeats=args.memory_repeats)
    print_results(res)
