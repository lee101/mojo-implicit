"""Alternating least squares for weighted implicit feedback."""

from __future__ import annotations

import time

import numpy as np
from scipy import linalg

from ._lib import least_squares
from ._matrix_factorization import (
    MatrixFactorizationBase,
    canonical_f64_csr,
    check_csr,
    check_random_state,
)


class AlternatingLeastSquares(MatrixFactorizationBase):
    _constructor_parameters = (
        "factors",
        "regularization",
        "alpha",
        "dtype",
        "use_native",
        "use_cg",
        "iterations",
        "calculate_training_loss",
        "num_threads",
        "random_state",
    )

    def __init__(
        self,
        factors=100,
        regularization=0.01,
        alpha=1.0,
        dtype=np.float32,
        use_native=True,
        use_cg=True,
        use_gpu=False,
        iterations=15,
        calculate_training_loss=False,
        num_threads=0,
        random_state=None,
    ):
        if use_gpu:
            raise NotImplementedError("GPU ALS is outside mojo-implicit's covered subset")
        super().__init__(num_threads)
        self.factors = int(factors)
        self.regularization = float(regularization)
        self.alpha = float(alpha)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("dtype must be float32 or float64")
        self.use_native = bool(use_native)
        self.use_cg = bool(use_cg)
        self.iterations = int(iterations)
        if self.factors <= 0:
            raise ValueError("factors must be positive")
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and non-negative")
        if not np.isfinite(self.alpha):
            raise ValueError("alpha must be finite")
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        self.calculate_training_loss = bool(calculate_training_loss)
        self.random_state = random_state
        self.cg_steps = 3
        self.fit_callback = None
        self._YtY = None
        self._XtX = None

    def fit(self, user_items, show_progress=True, callback=None):
        matrix = canonical_f64_csr(user_items)
        if self.alpha != 1.0:
            matrix = matrix * self.alpha
        item_users = matrix.T.tocsr()
        users, items = matrix.shape
        rng = check_random_state(self.random_state)
        user_factors = (
            rng.random((users, self.factors), dtype=self.dtype) * 0.01
            if self.user_factors is None
            else np.asarray(self.user_factors)
        )
        item_factors = (
            rng.random((items, self.factors), dtype=self.dtype) * 0.01
            if self.item_factors is None
            else np.asarray(self.item_factors)
        )
        if np.shape(user_factors) != (users, self.factors):
            raise ValueError("existing user_factors have the wrong shape")
        if np.shape(item_factors) != (items, self.factors):
            raise ValueError("existing item_factors have the wrong shape")
        user_factors = np.ascontiguousarray(user_factors, dtype=np.float64)
        item_factors = np.ascontiguousarray(item_factors, dtype=np.float64)
        callback = callback or self.fit_callback
        for epoch in range(self.iterations):
            started = time.perf_counter()
            least_squares(
                matrix,
                user_factors,
                item_factors,
                self.regularization,
                self.num_threads,
            )
            least_squares(
                item_users,
                item_factors,
                user_factors,
                self.regularization,
                self.num_threads,
            )
            loss = (
                self._training_loss(matrix, user_factors, item_factors)
                if self.calculate_training_loss
                else None
            )
            if callback is not None:
                callback(epoch, time.perf_counter() - started, loss)
        self.user_factors = np.asarray(user_factors, dtype=self.dtype)
        self.item_factors = np.asarray(item_factors, dtype=self.dtype)
        self._item_norms = self._user_norms = None
        self._YtY = self._XtX = None

    def _training_loss(self, matrix, users, items):
        item_gram = items.T @ items
        loss = float(np.sum((users @ item_gram) * users))
        total_confidence = 0.0
        for user in range(matrix.shape[0]):
            start, end = matrix.indptr[user : user + 2]
            ids = matrix.indices[start:end]
            confidence = np.abs(matrix.data[start:end])
            predictions = items[ids] @ users[user]
            preference = (matrix.data[start:end] > 0).astype(np.float64)
            loss += np.sum(
                confidence * (preference - predictions) ** 2 - predictions**2
            )
            total_confidence += confidence.sum()
        loss += self.regularization * (
            np.sum(users * users) + np.sum(items * items)
        )
        denominator = total_confidence + matrix.shape[0] * matrix.shape[1] - matrix.nnz
        return loss / denominator

    @property
    def YtY(self):
        self._require_fit()
        if self._YtY is None:
            self._YtY = self.item_factors.T @ self.item_factors
        return self._YtY

    @property
    def XtX(self):
        self._require_fit()
        if self._XtX is None:
            self._XtX = self.user_factors.T @ self.user_factors
        return self._XtX

    def recalculate_user(self, userid, user_items):
        matrix = canonical_f64_csr(user_items)
        if self.alpha != 1.0:
            matrix = matrix * self.alpha
        count = 1 if np.isscalar(userid) else len(userid)
        if matrix.shape[0] != count:
            raise ValueError("user_items should have one row for every user")
        result = np.zeros((count, self.factors), dtype=np.float64)
        least_squares(
            matrix,
            result,
            self.item_factors,
            self.regularization,
            self.num_threads,
        )
        result = result.astype(self.dtype)
        return result[0] if np.isscalar(userid) else result

    def recalculate_item(self, itemid, item_users):
        matrix = canonical_f64_csr(item_users)
        if self.alpha != 1.0:
            matrix = matrix * self.alpha
        count = 1 if np.isscalar(itemid) else len(itemid)
        if matrix.shape[0] != count:
            raise ValueError("item_users should have one row for every item")
        result = np.zeros((count, self.factors), dtype=np.float64)
        least_squares(
            matrix,
            result,
            self.user_factors,
            self.regularization,
            self.num_threads,
        )
        result = result.astype(self.dtype)
        return result[0] if np.isscalar(itemid) else result

    def partial_fit_users(self, userids, user_items):
        userids = np.asarray(userids, dtype=np.int64)
        if userids.ndim != 1 or not len(userids) or np.any(userids < 0):
            raise ValueError("userids must be a non-empty array of non-negative ids")
        if len(userids) != user_items.shape[0]:
            raise ValueError("user_items must contain 1 row for every user in userids")
        updated = self.recalculate_user(userids, user_items)
        if userids.max() >= len(self.user_factors):
            extra = userids.max() + 1 - len(self.user_factors)
            self.user_factors = np.vstack(
                (self.user_factors, np.zeros((extra, self.factors), dtype=self.dtype))
            )
        self.user_factors[userids] = updated
        self._user_norms = self._XtX = None

    def partial_fit_items(self, itemids, item_users):
        itemids = np.asarray(itemids, dtype=np.int64)
        if itemids.ndim != 1 or not len(itemids) or np.any(itemids < 0):
            raise ValueError("itemids must be a non-empty array of non-negative ids")
        if len(itemids) != item_users.shape[0]:
            raise ValueError("item_users must contain 1 row for every item in itemids")
        updated = self.recalculate_item(itemids, item_users)
        if itemids.max() >= len(self.item_factors):
            extra = itemids.max() + 1 - len(self.item_factors)
            self.item_factors = np.vstack(
                (self.item_factors, np.zeros((extra, self.factors), dtype=self.dtype))
            )
        self.item_factors[itemids] = updated
        self._item_norms = self._YtY = None

    def explain(self, userid, user_items, itemid, user_weights=None, N=10):
        matrix = check_csr(user_items)
        row = matrix.getrow(userid)
        confidence = row.data.astype(np.float64) * self.alpha
        factors = self.item_factors[row.indices].astype(np.float64)
        if user_weights is None:
            absolute = np.abs(confidence)
            a = self.YtY.astype(np.float64) + self.regularization * np.eye(self.factors)
            a += (factors.T * (absolute - 1.0)) @ factors
            user_weights = linalg.cho_factor(a)
        weighted_item = linalg.cho_solve(
            user_weights, self.item_factors[itemid].astype(np.float64)
        )
        scores = factors @ weighted_item * confidence
        positive = confidence > 0
        pairs = sorted(
            zip(row.indices[positive], scores[positive]),
            key=lambda pair: pair[1],
            reverse=True,
        )[:N]
        return float(scores[positive].sum()), pairs, user_weights
