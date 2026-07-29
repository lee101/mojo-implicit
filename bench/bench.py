"""Locked benchmarks against upstream implicit on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import implicit
import numpy as np
from scipy.sparse import csr_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_implicit as mi  # noqa: E402


def sparse_feedback(users, items, per_user, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(users, dtype=np.int32), per_user)
    cols = rng.integers(0, items, size=len(rows), dtype=np.int32)
    values = rng.integers(1, 5, size=len(rows)).astype(np.float64)
    matrix = csr_matrix((values, (rows, cols)), shape=(users, items))
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return best


def als_case():
    matrix = sparse_feedback(8_000, 2_500, 15)
    kwargs = dict(
        factors=24,
        regularization=0.05,
        dtype=np.float64,
        use_cg=False,
        iterations=2,
        random_state=1,
    )
    return (
        lambda: mi.AlternatingLeastSquares(**kwargs).fit(matrix, show_progress=False),
        lambda: implicit.cpu.als.AlternatingLeastSquares(**kwargs).fit(
            matrix, show_progress=False
        ),
    )


def bpr_case():
    matrix = sparse_feedback(15_000, 4_000, 15)
    kwargs = dict(
        factors=32,
        learning_rate=0.03,
        regularization=0.01,
        dtype=np.float64,
        iterations=8,
        random_state=2,
    )
    return (
        lambda: mi.BayesianPersonalizedRanking(**kwargs).fit(
            matrix, show_progress=False
        ),
        lambda: implicit.cpu.bpr.BayesianPersonalizedRanking(**kwargs).fit(
            matrix, show_progress=False
        ),
    )


def recommend_case():
    rng = np.random.default_rng(3)
    users, items, factors = 400, 10_000, 64
    user_factors = rng.normal(size=(users, factors))
    item_factors = rng.normal(size=(items, factors))
    matrix = sparse_feedback(users, items, 30, seed=4)
    ours = mi.AlternatingLeastSquares(factors=factors, dtype=np.float64)
    theirs = implicit.cpu.als.AlternatingLeastSquares(
        factors=factors, dtype=np.float64
    )
    for model in (ours, theirs):
        model.user_factors = user_factors.copy()
        model.item_factors = item_factors.copy()
    userids = np.arange(users)
    return (
        lambda: ours.recommend(userids, matrix, N=20),
        lambda: theirs.recommend(userids, matrix, N=20),
    )


CASES = [
    ("ALS.fit exact, 8k x 2.5k, 120k nnz, f=24, 2 it", als_case),
    ("BPR.fit, 15k x 4k, 225k nnz, f=32, 8 epochs", bpr_case),
    ("recommend, 400 x 10k candidates, f=64, N=20", recommend_case),
]


def main():
    processor = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            processor = next(
                line.split(":", 1)[1].strip()
                for line in cpuinfo
                if line.startswith("model name")
            )
    except (OSError, StopIteration):
        processor = processor or platform.machine()
    print(f"Machine: {processor} ({platform.platform()})")
    print()
    print("| Case | mojo-implicit | implicit 0.7.3 | Upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, build in CASES:
        ours, theirs = build()
        ours()
        ours_time = timeit(ours)
        theirs_time = timeit(theirs)
        print(
            f"| {name} | {ours_time * 1000:.2f} ms | "
            f"{theirs_time * 1000:.2f} ms | {theirs_time / ours_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
