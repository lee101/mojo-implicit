"""Shared recommendation operations for ALS and BPR."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csr_matrix

from ._lib import score


def check_csr(value) -> csr_matrix:
    if isinstance(value, csr_matrix):
        return value
    warnings.warn("converting sparse input to CSR", RuntimeWarning, stacklevel=2)
    return value.tocsr()


def canonical_f64_csr(value) -> csr_matrix:
    matrix = check_csr(value)
    if np.issubdtype(matrix.dtype, np.complexfloating):
        raise TypeError("complex confidence values are not supported")
    needs_cleanup = not matrix.has_canonical_format or not matrix.has_sorted_indices
    if matrix.dtype != np.float64:
        matrix = matrix.astype(np.float64)
    elif needs_cleanup:
        matrix = matrix.copy()
    if not matrix.has_canonical_format:
        matrix.sum_duplicates()
    if not matrix.has_sorted_indices:
        matrix.sort_indices()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("confidence values must be finite")
    return matrix


def checked_ids(value, name: str) -> np.ndarray:
    ids = np.asarray(value)
    if ids.ndim > 1 or not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    ids = np.atleast_1d(ids).astype(np.int64, copy=False)
    if np.any(ids < 0):
        raise IndexError(f"{name} must be non-negative")
    return ids


def check_random_state(value):
    if isinstance(value, np.random.RandomState):
        return np.random.default_rng(value.randint(2**31))
    return np.random.default_rng(value)


class MatrixFactorizationBase:
    def __init__(self, num_threads: int = 0):
        self.item_factors = None
        self.user_factors = None
        self._item_norms = None
        self._user_norms = None
        self.num_threads = int(num_threads)
        if self.num_threads < 0:
            raise ValueError("num_threads must be non-negative")

    def _require_fit(self):
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("model has not been fit")

    def _rank(self, queries, factors, candidate_ids, n, filtered=None):
        candidate_ids = checked_ids(candidate_ids, "candidate ids")
        n = int(n)
        if n < 0:
            raise ValueError("N must be non-negative")
        scores = score(queries, factors, candidate_ids, self.num_threads)
        if filtered is not None:
            floor = -np.finfo(self.dtype).max
            sorted_unique_candidates = (
                len(candidate_ids) < 2
                or bool(np.all(candidate_ids[:-1] < candidate_ids[1:]))
            )
            for row, blocked in enumerate(filtered):
                if sorted_unique_candidates:
                    blocked = np.asarray(blocked, dtype=np.int64)
                    positions = np.searchsorted(candidate_ids, blocked)
                    valid = positions < len(candidate_ids)
                    positions = positions[valid]
                    blocked = blocked[valid]
                    positions = positions[candidate_ids[positions] == blocked]
                    scores[row, positions] = floor
                else:
                    scores[row, np.isin(candidate_ids, blocked)] = floor
        n = min(n, len(candidate_ids))
        result_ids = np.full((scores.shape[0], n), -1, dtype=np.int32)
        result_scores = np.full(
            (scores.shape[0], n), -np.finfo(self.dtype).max, dtype=self.dtype
        )
        if n == 0:
            return result_ids, result_scores
        for row in range(scores.shape[0]):
            take = min(n, len(candidate_ids))
            if take < len(candidate_ids):
                selected = np.argpartition(scores[row], -take)[-take:]
            else:
                selected = np.arange(len(candidate_ids))
            order = np.lexsort(
                (candidate_ids[selected], -scores[row, selected])
            )
            selected = selected[order][:take]
            result_ids[row, :take] = candidate_ids[selected]
            result_scores[row, :take] = scores[row, selected]
        return result_ids, result_scores

    def recommend(
        self,
        userid,
        user_items,
        N=10,
        filter_already_liked_items=True,
        filter_items=None,
        recalculate_user=False,
        items=None,
    ):
        self._require_fit()
        scalar = np.isscalar(userid)
        userids = checked_ids(userid, "userids")
        if len(userids) and np.any(userids >= len(self.user_factors)):
            raise IndexError("userid is outside the model")
        if filter_already_liked_items or recalculate_user:
            if not isinstance(user_items, csr_matrix):
                raise ValueError("user_items needs to be a CSR sparse matrix")
            if user_items.shape[0] != len(userids):
                raise ValueError("user_items must contain 1 row for every user in userids")
        queries = (
            self.recalculate_user(userid, user_items)
            if recalculate_user
            else self.user_factors[userids]
        )
        candidates = (
            np.arange(self.item_factors.shape[0], dtype=np.int64)
            if items is None
            else np.sort(checked_ids(items, "items"))
        )
        if items is not None and filter_items is not None:
            raise ValueError("Can't set both items and filter_items in recommend call")
        if len(candidates) and (
            candidates[0] < 0 or candidates[-1] >= self.item_factors.shape[0]
        ):
            raise IndexError("Some itemids in the items parameter are not in the model")
        blocked = []
        global_filter = (
            checked_ids(filter_items, "filter_items")
            if filter_items is not None
            else np.empty(0, dtype=np.int64)
        )
        for row in range(len(userids)):
            liked = (
                user_items.indices[user_items.indptr[row] : user_items.indptr[row + 1]]
                if filter_already_liked_items
                else np.empty(0, dtype=np.int64)
            )
            blocked.append(np.concatenate((liked, global_filter)))
        ids, scores = self._rank(queries, self.item_factors, candidates, N, blocked)
        return (ids[0], scores[0]) if scalar else (ids, scores)

    def similar_users(self, userid, N=10, filter_users=None, users=None):
        self._require_fit()
        if users is not None and filter_users is not None:
            raise ValueError("Can't set both users and filter_users in similar_users call")
        scalar = np.isscalar(userid)
        query_ids = checked_ids(userid, "userids")
        if len(query_ids) and np.any(query_ids >= len(self.user_factors)):
            raise IndexError("userid is outside the model")
        candidates = (
            np.arange(len(self.user_factors), dtype=np.int64)
            if users is None
            else checked_ids(users, "users")
        )
        queries = self.user_factors[query_ids] / self.user_norms[query_ids, None]
        normalized = self.user_factors / self.user_norms[:, None]
        blocked = [
            checked_ids(filter_users, "filter_users")
            if filter_users is not None
            else np.empty(0, dtype=np.int64)
            for _ in query_ids
        ]
        ids, scores = self._rank(queries, normalized, candidates, N, blocked)
        return (ids[0], scores[0]) if scalar else (ids, scores)

    def similar_items(
        self,
        itemid,
        N=10,
        recalculate_item=False,
        item_users=None,
        filter_items=None,
        items=None,
    ):
        self._require_fit()
        if items is not None and filter_items is not None:
            raise ValueError("Can't set both items and filter_items in similar_items call")
        scalar = np.isscalar(itemid)
        query_ids = checked_ids(itemid, "itemids")
        if len(query_ids) and np.any(query_ids >= len(self.item_factors)):
            raise IndexError("itemid is outside the model")
        raw_queries = (
            self.recalculate_item(itemid, item_users)
            if recalculate_item
            else self.item_factors[query_ids]
        )
        raw_queries = np.atleast_2d(raw_queries)
        query_norms = np.linalg.norm(raw_queries, axis=1)
        query_norms[query_norms == 0] = 1e-10
        queries = raw_queries / query_norms[:, None]
        normalized = self.item_factors / self.item_norms[:, None]
        candidates = (
            np.arange(len(self.item_factors), dtype=np.int64)
            if items is None
            else checked_ids(items, "items")
        )
        blocked = [
            checked_ids(filter_items, "filter_items")
            if filter_items is not None
            else np.empty(0, dtype=np.int64)
            for _ in query_ids
        ]
        ids, scores = self._rank(queries, normalized, candidates, N, blocked)
        return (ids[0], scores[0]) if scalar else (ids, scores)

    @property
    def user_norms(self):
        if self._user_norms is None:
            self._user_norms = np.linalg.norm(self.user_factors, axis=1)
            self._user_norms[self._user_norms == 0] = 1e-10
        return self._user_norms

    @property
    def item_norms(self):
        if self._item_norms is None:
            self._item_norms = np.linalg.norm(self.item_factors, axis=1)
            self._item_norms[self._item_norms == 0] = 1e-10
        return self._item_norms

    def recalculate_user(self, userid, user_items):
        raise NotImplementedError("recalculate_user is not supported with this model")

    def recalculate_item(self, itemid, item_users):
        raise NotImplementedError("recalculate_item is not supported with this model")

    def to_gpu(self):
        raise NotImplementedError("GPU models are outside mojo-implicit's covered subset")

    def save(self, fileobj_or_path):
        values = {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_") and value is not None
        }
        if "dtype" in values:
            values["dtype"] = values["dtype"].name
        np.savez(fileobj_or_path, **values)

    @classmethod
    def load(cls, fileobj_or_path):
        with np.load(fileobj_or_path, allow_pickle=False) as data:
            values = {key: data[key] for key in data.files}
        constructor = {}
        for key in cls._constructor_parameters:
            if key in values:
                value = values.pop(key)
                constructor[key] = value.item() if value.ndim == 0 else value
        model = cls(**constructor)
        for key, value in values.items():
            setattr(model, key, value)
        return model
