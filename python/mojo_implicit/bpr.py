"""Bayesian Personalized Ranking with pairwise SGD in Mojo."""

from __future__ import annotations

import time

import numpy as np

from ._lib import addr, f64, i64, lib, worker_count
from ._matrix_factorization import (
    MatrixFactorizationBase,
    canonical_f64_csr,
    check_random_state,
)


class BayesianPersonalizedRanking(MatrixFactorizationBase):
    _constructor_parameters = (
        "factors",
        "learning_rate",
        "regularization",
        "dtype",
        "iterations",
        "num_threads",
        "verify_negative_samples",
        "random_state",
    )

    def __init__(
        self,
        factors=100,
        learning_rate=0.01,
        regularization=0.01,
        dtype=np.float32,
        iterations=100,
        use_gpu=False,
        num_threads=0,
        verify_negative_samples=True,
        random_state=None,
    ):
        if use_gpu:
            raise NotImplementedError("GPU BPR is outside mojo-implicit's covered subset")
        super().__init__(num_threads)
        self.factors = int(factors)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype must be float32 or float64")
        self.iterations = int(iterations)
        if self.factors <= 0:
            raise ValueError("factors must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning_rate must be finite and non-negative")
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and non-negative")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        self.verify_negative_samples = bool(verify_negative_samples)
        self.random_state = random_state

    def fit(self, user_items, show_progress=True, callback=None):
        matrix = canonical_f64_csr(user_items)
        users_count, items_count = matrix.shape
        nnz = matrix.nnz
        rng = check_random_state(self.random_state)
        user_counts = np.diff(matrix.indptr)
        userids = i64(np.repeat(np.arange(users_count), user_counts))
        itemids = i64(matrix.indices)
        indptr = i64(matrix.indptr)
        if self.item_factors is None:
            item_factors = (
                rng.random((items_count, self.factors + 1), dtype=self.dtype) - 0.5
            ) / self.factors
            item_counts = np.bincount(matrix.indices, minlength=items_count)
            item_factors[item_counts == 0] = 0
        else:
            item_factors = self.item_factors
        if self.user_factors is None:
            user_factors = (
                rng.random((users_count, self.factors + 1), dtype=self.dtype) - 0.5
            ) / self.factors
            user_factors[user_counts == 0] = 0
        else:
            user_factors = self.user_factors
        expected_users = (users_count, self.factors + 1)
        expected_items = (items_count, self.factors + 1)
        if np.shape(user_factors) != expected_users:
            raise ValueError("existing user_factors have the wrong shape")
        if np.shape(item_factors) != expected_items:
            raise ValueError("existing item_factors have the wrong shape")
        user_factors = f64(user_factors, copy=True)
        item_factors = f64(item_factors, copy=True)
        user_factors[:, self.factors] = 1.0
        skipped = np.zeros(1, dtype=np.int64)
        workers = min(
            worker_count(nnz, nnz * self.factors, self.num_threads),
            16,
        )
        locks = np.zeros(users_count + items_count, dtype=np.int64)
        counts = np.empty(2 * workers, dtype=np.int64)
        if nnz == 0:
            self.user_factors = user_factors.astype(self.dtype, copy=False)
            self.item_factors = item_factors.astype(self.dtype, copy=False)
            self._item_norms = self._user_norms = None
            return
        for epoch in range(self.iterations):
            started = time.perf_counter()
            liked_samples = i64(rng.integers(0, nnz, size=nnz))
            disliked_samples = i64(rng.integers(0, nnz, size=nnz))
            correct = lib().mi_bpr_update(
                addr(userids),
                addr(itemids),
                addr(indptr),
                addr(liked_samples),
                addr(disliked_samples),
                addr(user_factors),
                addr(item_factors),
                addr(skipped),
                addr(locks),
                addr(counts),
                nnz,
                self.factors,
                users_count,
                workers,
                self.learning_rate,
                self.regularization,
                int(self.verify_negative_samples),
            )
            if callback is not None:
                callback(epoch, time.perf_counter() - started, correct, int(skipped[0]))
        self.user_factors = user_factors.astype(self.dtype, copy=False)
        self.item_factors = item_factors.astype(self.dtype, copy=False)
        self._item_norms = self._user_norms = None
