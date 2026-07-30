"""
Plaintext Baseline — same operations without encryption.
Used as the comparison reference for overhead calculations.
"""

import time
import tracemalloc


def _tick():
    return time.perf_counter()


def run_benchmark(vector: list[float]) -> dict:
    results = {}
    n = len(vector)

    tracemalloc.start()
    t0 = _tick()

    plain_add = [v + v for v in vector]
    plain_mul = [v * v for v in vector]
    plain_sum = sum(vector)
    plain_avg = plain_sum / n

    results["total_time_s"] = _tick() - t0
    results["peak_mem_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    results["add_result"]  = plain_add
    results["mul_result"]  = plain_mul
    results["sum_result"]  = plain_sum
    results["avg_result"]  = plain_avg

    return results


def print_results(results: dict):
    print("=" * 50)
    print("Plaintext Baseline Results")
    print("=" * 50)
    print(f"Total time : {results['total_time_s']*1000:.6f} ms")
    print(f"Peak mem   : {results['peak_mem_kb']:.4f} KB")
    print(f"Add result : {results['add_result']}")
    print(f"Mul result : {results['mul_result']}")
    print(f"Sum result : {results['sum_result']}")
    print(f"Avg result : {results['avg_result']}")


if __name__ == "__main__":
    DATA = [1.0, 2.0, 3.0, 4.0, 5.0]
    print(f"Input vector: {DATA}\n")
    res = run_benchmark(DATA)
    print_results(res)
