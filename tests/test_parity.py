"""Numerical and behavioral parity with implicit 0.7.x."""

from __future__ import annotations

import inspect
import tempfile

import implicit
import numpy as np
import pytest
from scipy.sparse import csr_matrix

import mojo_implicit as mi
from mojo_implicit import _lib


@pytest.fixture
def confidence():
    rng = np.random.default_rng(12)
    dense = rng.integers(0, 6, size=(24, 18)).astype(np.float64)
    dense[rng.random(dense.shape) < 0.72] = 0
    return csr_matrix(dense)


def als_pair(matrix, *, iterations=2, factors=6, alpha=1.0, loss=False):
    kwargs = dict(
        factors=factors,
        regularization=0.07,
        alpha=alpha,
        dtype=np.float64,
        use_cg=False,
        iterations=iterations,
        calculate_training_loss=loss,
        num_threads=1,
        random_state=9,
    )
    ours = mi.AlternatingLeastSquares(**kwargs)
    theirs = implicit.cpu.als.AlternatingLeastSquares(**kwargs)
    return ours, theirs


def test_public_constructor_signatures_cover_upstream():
    als = inspect.signature(mi.AlternatingLeastSquares).parameters
    bpr = inspect.signature(mi.BayesianPersonalizedRanking).parameters
    for name in inspect.signature(implicit.als.AlternatingLeastSquares).parameters:
        assert name in als
    for name in inspect.signature(implicit.bpr.BayesianPersonalizedRanking).parameters:
        assert name in bpr


def test_als_exact_solver_matches_upstream(confidence):
    ours, theirs = als_pair(confidence)
    ours.fit(confidence, show_progress=False)
    theirs.fit(confidence, show_progress=False)
    assert np.allclose(ours.user_factors, theirs.user_factors, atol=2e-11)
    assert np.allclose(ours.item_factors, theirs.item_factors, atol=2e-11)


def test_als_float32_matches_upstream(confidence):
    kwargs = dict(
        factors=5,
        regularization=0.1,
        dtype=np.float32,
        use_cg=False,
        iterations=2,
        num_threads=1,
        random_state=4,
    )
    ours = mi.AlternatingLeastSquares(**kwargs)
    theirs = implicit.cpu.als.AlternatingLeastSquares(**kwargs)
    ours.fit(confidence, show_progress=False)
    theirs.fit(confidence, show_progress=False)
    assert ours.user_factors.dtype == np.float32
    assert ours.item_factors.dtype == np.float32
    assert np.allclose(ours.user_factors, theirs.user_factors, atol=3e-5)
    assert np.allclose(ours.item_factors, theirs.item_factors, atol=3e-5)


def test_als_alpha_and_negative_confidence_match_upstream(confidence):
    matrix = confidence.copy()
    matrix.data[::4] *= -1
    ours, theirs = als_pair(matrix, alpha=2.5)
    ours.fit(matrix, show_progress=False)
    theirs.fit(matrix, show_progress=False)
    assert np.allclose(ours.user_factors, theirs.user_factors, atol=2e-11)
    assert np.allclose(ours.item_factors, theirs.item_factors, atol=2e-11)


def test_als_training_loss_and_callback_match_upstream(confidence):
    ours, theirs = als_pair(confidence, iterations=1, loss=True)
    ours_calls = []
    their_calls = []
    ours.fit(confidence, show_progress=False, callback=lambda e, t, loss: ours_calls.append((e, loss)))
    theirs.fit(
        confidence,
        show_progress=False,
        callback=lambda e, t, loss: their_calls.append((e, loss)),
    )
    assert ours_calls[0][0] == their_calls[0][0] == 0
    assert ours_calls[0][1] == pytest.approx(their_calls[0][1], rel=1e-9)


def test_als_recalculate_user_and_item_match_upstream(confidence):
    ours, theirs = als_pair(confidence)
    ours.fit(confidence, show_progress=False)
    theirs.fit(confidence, show_progress=False)
    rows = np.array([2, 7, 11])
    assert np.allclose(
        ours.recalculate_user(rows, confidence[rows]),
        theirs.recalculate_user(rows, confidence[rows]),
        atol=2e-11,
    )
    item_users = confidence.T.tocsr()
    items = np.array([1, 5])
    assert np.allclose(
        ours.recalculate_item(items, item_users[items]),
        theirs.recalculate_item(items, item_users[items]),
        atol=2e-11,
    )


def test_als_simd_tail_and_parallel_path_match_serial():
    rng = np.random.default_rng(21)
    rows, items, factors = 300, 80, 19
    dense = rng.random((rows, items))
    dense[dense > 0.08] = 0
    matrix = csr_matrix(dense)
    fixed = rng.normal(scale=0.1, size=(items, factors))
    serial = np.empty((rows, factors), dtype=np.float64)
    parallel = np.empty_like(serial)
    _lib.least_squares(matrix, serial, fixed, 0.2, num_threads=1)
    _lib.least_squares(matrix, parallel, fixed, 0.2, num_threads=4)
    assert np.allclose(parallel, serial, atol=2e-12)


def test_als_partial_fit_grows_model(confidence):
    model, _ = als_pair(confidence, iterations=1)
    model.fit(confidence, show_progress=False)
    item_users = confidence.T.tocsr()
    expected_item = model.recalculate_item([25], item_users[[0]])
    model.partial_fit_items([25], item_users[[0]])
    assert model.item_factors.shape == (26, model.factors)
    assert np.allclose(model.item_factors[25], expected_item[0])
    model, _ = als_pair(confidence, iterations=1)
    model.fit(confidence, show_progress=False)
    expected = model.recalculate_user([30], confidence[[0]])
    model.partial_fit_users([30], confidence[[0]])
    assert model.user_factors.shape == (31, model.factors)
    assert np.allclose(model.user_factors[30], expected[0])


def test_recommend_single_and_batch_match_upstream(confidence):
    ours, theirs = als_pair(confidence)
    ours.fit(confidence, show_progress=False)
    theirs.user_factors = ours.user_factors.copy()
    theirs.item_factors = ours.item_factors.copy()
    ids, scores = ours.recommend(3, confidence[3], N=5)
    ref_ids, ref_scores = theirs.recommend(3, confidence[3], N=5)
    assert np.array_equal(ids, ref_ids)
    assert np.allclose(scores, ref_scores)
    users = np.array([1, 4, 9])
    ids, scores = ours.recommend(users, confidence[users], N=4)
    ref_ids, ref_scores = theirs.recommend(users, confidence[users], N=4)
    assert np.array_equal(ids, ref_ids)
    assert np.allclose(scores, ref_scores)


def test_recommend_filters_and_candidate_subset(confidence):
    model, _ = als_pair(confidence)
    model.fit(confidence, show_progress=False)
    candidates = [9, 3, 12, 1, 16]
    ids, _ = model.recommend(
        0,
        confidence[0],
        N=3,
        filter_already_liked_items=False,
        items=candidates,
    )
    assert set(ids).issubset(candidates)
    filtered = [int(ids[0])]
    ids2, _ = model.recommend(
        0, confidence[0], N=3, filter_already_liked_items=False, filter_items=filtered
    )
    assert filtered[0] not in ids2
    with pytest.raises(ValueError):
        model.recommend(0, confidence[0], filter_items=[1], items=[2])


def test_score_simd_tail_and_parallel_threshold():
    rng = np.random.default_rng(22)
    queries = rng.normal(size=(37, 7))
    candidates = rng.normal(size=(503, 7))
    candidate_ids = np.arange(len(candidates), dtype=np.int64)
    serial = _lib.score(queries, candidates, candidate_ids, num_threads=1)
    parallel = _lib.score(queries, candidates, candidate_ids, num_threads=4)
    assert np.allclose(parallel, serial, atol=2e-12)
    assert _lib.worker_count(2, _lib._PARALLEL_WORK_THRESHOLD - 1, 8) == 1
    assert _lib.worker_count(2, _lib._PARALLEL_WORK_THRESHOLD, 8) == 2


def test_ffi_rejects_unsafe_shapes_and_indices():
    matrix = csr_matrix(np.eye(3))
    with pytest.raises(ValueError, match="fixed row count"):
        _lib.least_squares(matrix, np.empty((3, 2)), np.empty((2, 2)), 0.1)
    with pytest.raises(ValueError, match="result shape"):
        _lib.least_squares(matrix, np.empty((2, 2)), np.empty((3, 2)), 0.1)
    with pytest.raises(ValueError, match="same positive factor count"):
        _lib.score(np.empty((1, 2)), np.empty((3, 3)), np.array([0]))
    with pytest.raises(IndexError, match="outside"):
        _lib.score(np.empty((1, 2)), np.empty((3, 2)), np.array([3]))
    with pytest.raises(TypeError, match="integer"):
        _lib.score(np.empty((1, 2)), np.empty((3, 2)), np.array([1.5]))


def test_warm_start_shapes_are_checked_before_native_call(confidence):
    als = mi.AlternatingLeastSquares(factors=3, iterations=1)
    als.user_factors = np.empty((1, 3))
    with pytest.raises(ValueError, match="user_factors"):
        als.fit(confidence, show_progress=False)
    bpr = mi.BayesianPersonalizedRanking(factors=3, iterations=1)
    bpr.item_factors = np.empty((1, 4))
    with pytest.raises(ValueError, match="item_factors"):
        bpr.fit(confidence, show_progress=False)


def test_bpr_accepts_completely_empty_matrix():
    matrix = csr_matrix((3, 4), dtype=np.float64)
    model = mi.BayesianPersonalizedRanking(
        factors=3, iterations=2, random_state=0, dtype=np.float64
    )
    model.fit(matrix, show_progress=False)
    assert model.user_factors.shape == (3, 4)
    assert model.item_factors.shape == (4, 4)
    assert np.array_equal(model.user_factors[:, :-1], np.zeros((3, 3)))
    assert np.array_equal(model.user_factors[:, -1], np.ones(3))
    assert np.array_equal(model.item_factors, np.zeros((4, 4)))


def test_similarity_matches_upstream(confidence):
    ours, theirs = als_pair(confidence)
    ours.fit(confidence, show_progress=False)
    theirs.user_factors = ours.user_factors.copy()
    theirs.item_factors = ours.item_factors.copy()
    for ours_result, their_result in (
        (ours.similar_items(4, N=6), theirs.similar_items(4, N=6)),
        (ours.similar_users(4, N=6), theirs.similar_users(4, N=6)),
    ):
        assert np.array_equal(ours_result[0], their_result[0])
        assert np.allclose(ours_result[1], their_result[1], atol=1e-12)


def test_batch_similarity_candidates_and_filters(confidence):
    model, _ = als_pair(confidence)
    model.fit(confidence, show_progress=False)
    item_ids, _ = model.similar_items(
        [1, 3], N=2, items=[1, 2, 3, 4]
    )
    user_ids, _ = model.similar_users(
        [1, 3], N=2, users=[1, 2, 3, 4]
    )
    assert item_ids.shape == user_ids.shape == (2, 2)
    assert np.all(np.isin(item_ids, [1, 2, 3, 4]))
    assert np.all(np.isin(user_ids, [1, 2, 3, 4]))
    assert 1 not in model.similar_items(1, N=4, filter_items=[1])[0]
    assert 1 not in model.similar_users(1, N=4, filter_users=[1])[0]


def test_als_explain_matches_upstream(confidence):
    ours, theirs = als_pair(confidence)
    ours.fit(confidence, show_progress=False)
    theirs.user_factors = ours.user_factors.copy()
    theirs.item_factors = ours.item_factors.copy()
    ours_total, ours_pairs, _ = ours.explain(2, confidence, 5, N=4)
    their_total, their_pairs, _ = theirs.explain(2, confidence, 5, N=4)
    assert ours_total == pytest.approx(their_total, abs=2e-11)
    assert [pair[0] for pair in ours_pairs] == [pair[0] for pair in their_pairs]
    assert np.allclose(
        [pair[1] for pair in ours_pairs],
        [pair[1] for pair in their_pairs],
        atol=2e-11,
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: mi.AlternatingLeastSquares(factors=4, iterations=1, random_state=2),
        lambda: mi.BayesianPersonalizedRanking(factors=4, iterations=1, random_state=2),
    ],
)
def test_save_load_roundtrip(confidence, factory):
    model = factory()
    model.fit(confidence, show_progress=False)
    with tempfile.NamedTemporaryFile(suffix=".npz") as handle:
        model.save(handle.name)
        restored = type(model).load(handle.name)
    assert np.array_equal(restored.user_factors, model.user_factors)
    assert np.array_equal(restored.item_factors, model.item_factors)
    assert restored.factors == model.factors


def python_bpr_update(
    userids,
    itemids,
    indptr,
    liked_samples,
    disliked_samples,
    users,
    items,
    learning_rate,
    regularization,
):
    correct = skipped = 0
    factors = users.shape[1] - 1
    for liked_index, disliked_index in zip(liked_samples, disliked_samples):
        user_id = userids[liked_index]
        liked_id = itemids[liked_index]
        disliked_id = itemids[disliked_index]
        if disliked_id in itemids[indptr[user_id] : indptr[user_id + 1]]:
            skipped += 1
            continue
        user = users[user_id]
        liked = items[liked_id]
        disliked = items[disliked_id]
        score = user @ (liked - disliked)
        z = 1.0 / (1.0 + np.exp(score))
        correct += z < 0.5
        for factor in range(factors):
            old_user = user[factor]
            user[factor] += learning_rate * (
                z * (liked[factor] - disliked[factor])
                - regularization * user[factor]
            )
            liked[factor] += learning_rate * (
                z * old_user - regularization * liked[factor]
            )
            disliked[factor] += learning_rate * (
                -z * old_user - regularization * disliked[factor]
            )
        liked[factors] += learning_rate * (z - regularization * liked[factors])
        disliked[factors] += learning_rate * (-z - regularization * disliked[factors])
    return correct, skipped


def test_bpr_mojo_simd_tail_matches_reference():
    indptr = _lib.i64([0, 2, 4, 5])
    itemids = _lib.i64([0, 2, 1, 3, 4])
    userids = _lib.i64([0, 0, 1, 1, 2])
    liked_samples = _lib.i64([0, 2, 4, 1, 3])
    disliked_samples = _lib.i64([2, 4, 0, 3, 1])
    rng = np.random.default_rng(3)
    users = rng.normal(scale=0.1, size=(3, 6))
    users[:, -1] = 1
    items = rng.normal(scale=0.1, size=(5, 6))
    ref_users, ref_items = users.copy(), items.copy()
    expected = python_bpr_update(
        userids,
        itemids,
        indptr,
        liked_samples,
        disliked_samples,
        ref_users,
        ref_items,
        0.03,
        0.02,
    )
    skipped = np.zeros(1, dtype=np.int64)
    locks = np.zeros(len(users) + len(items), dtype=np.int64)
    counts = np.empty(2, dtype=np.int64)
    correct = _lib.lib().mi_bpr_update(
        _lib.addr(userids),
        _lib.addr(itemids),
        _lib.addr(indptr),
        _lib.addr(liked_samples),
        _lib.addr(disliked_samples),
        _lib.addr(users),
        _lib.addr(items),
        _lib.addr(skipped),
        _lib.addr(locks),
        _lib.addr(counts),
        len(liked_samples),
        5,
        len(users),
        1,
        0.03,
        0.02,
        1,
    )
    assert (correct, skipped[0]) == expected
    assert np.allclose(users, ref_users, atol=2e-15)
    assert np.allclose(items, ref_items, atol=2e-15)


def test_bpr_parallel_independent_updates_match_serial():
    userids = _lib.i64([0, 1, 2, 3, 0, 1, 2, 3])
    itemids = _lib.i64(np.arange(8))
    indptr = _lib.i64([0, 2, 4, 6, 8])
    liked_samples = _lib.i64([0, 1, 2, 3])
    disliked_samples = _lib.i64([4, 5, 6, 7])
    rng = np.random.default_rng(23)
    initial_users = rng.normal(scale=0.1, size=(4, 6))
    initial_users[:, -1] = 1
    initial_items = rng.normal(scale=0.1, size=(8, 6))

    def update(workers):
        users = initial_users.copy()
        items = initial_items.copy()
        skipped = np.zeros(1, dtype=np.int64)
        locks = np.zeros(len(users) + len(items), dtype=np.int64)
        counts = np.empty(2 * workers, dtype=np.int64)
        correct = _lib.lib().mi_bpr_update(
            _lib.addr(userids),
            _lib.addr(itemids),
            _lib.addr(indptr),
            _lib.addr(liked_samples),
            _lib.addr(disliked_samples),
            _lib.addr(users),
            _lib.addr(items),
            _lib.addr(skipped),
            _lib.addr(locks),
            _lib.addr(counts),
            len(liked_samples),
            5,
            len(users),
            workers,
            0.03,
            0.02,
            0,
        )
        return correct, skipped[0], users, items

    serial = update(1)
    parallel = update(4)
    assert parallel[:2] == serial[:2]
    assert np.array_equal(parallel[2], serial[2])
    assert np.array_equal(parallel[3], serial[3])


def grouped_feedback():
    rng = np.random.default_rng(14)
    dense = np.zeros((60, 30), dtype=np.float32)
    for user in range(60):
        group = user % 3
        liked = rng.choice(np.arange(group * 10, (group + 1) * 10), 6, replace=False)
        dense[user, liked] = 1
    return csr_matrix(dense)


def pairwise_accuracy(model, matrix):
    correct = total = 0
    all_items = np.arange(matrix.shape[1])
    for user in range(matrix.shape[0]):
        liked = matrix.indices[matrix.indptr[user] : matrix.indptr[user + 1]]
        disliked = np.setdiff1d(all_items, liked)
        positive = model.item_factors[liked] @ model.user_factors[user]
        negative = model.item_factors[disliked] @ model.user_factors[user]
        correct += np.sum(positive[:, None] > negative[None, :])
        total += positive.size * negative.size
    return correct / total


def test_bpr_learns_pairwise_ranking_comparable_to_upstream():
    matrix = grouped_feedback()
    kwargs = dict(
        factors=8,
        learning_rate=0.05,
        regularization=0.01,
        dtype=np.float64,
        iterations=40,
        num_threads=1,
        random_state=5,
    )
    ours = mi.BayesianPersonalizedRanking(**kwargs)
    theirs = implicit.cpu.bpr.BayesianPersonalizedRanking(**kwargs)
    ours.fit(matrix, show_progress=False)
    theirs.fit(matrix, show_progress=False)
    ours_accuracy = pairwise_accuracy(ours, matrix)
    their_accuracy = pairwise_accuracy(theirs, matrix)
    assert ours_accuracy > 0.8
    assert abs(ours_accuracy - their_accuracy) < 0.15


def test_bpr_shape_bias_and_callback(confidence):
    calls = []
    model = mi.BayesianPersonalizedRanking(
        factors=5, iterations=3, random_state=1, dtype=np.float32
    )
    model.fit(
        confidence,
        show_progress=False,
        callback=lambda epoch, elapsed, correct, skipped: calls.append(
            (epoch, correct, skipped)
        ),
    )
    assert model.user_factors.shape == (confidence.shape[0], 6)
    assert model.item_factors.shape == (confidence.shape[1], 6)
    assert np.array_equal(model.user_factors[:, -1], np.ones(confidence.shape[0]))
    assert [call[0] for call in calls] == [0, 1, 2]
    assert all(call[1] + call[2] <= confidence.nnz for call in calls)


def test_bpr_recommendation_filters_seen_items():
    matrix = grouped_feedback()
    model = mi.BayesianPersonalizedRanking(
        factors=6, iterations=10, random_state=6
    )
    model.fit(matrix, show_progress=False)
    ids, scores = model.recommend(0, matrix[0], N=8)
    assert not np.intersect1d(ids, matrix[0].indices).size
    assert np.all(scores[:-1] >= scores[1:])


def test_empty_rows_are_zeroed(confidence):
    from scipy.sparse import vstack

    matrix = vstack(
        [confidence, csr_matrix((1, confidence.shape[1]))], format="csr"
    )
    als = mi.AlternatingLeastSquares(factors=3, iterations=1, random_state=0)
    als.fit(matrix, show_progress=False)
    assert np.array_equal(als.user_factors[-1], np.zeros(3))
    bpr = mi.BayesianPersonalizedRanking(factors=3, iterations=1, random_state=0)
    bpr.fit(matrix, show_progress=False)
    assert np.array_equal(bpr.user_factors[-1, :-1], np.zeros(3))
    assert bpr.user_factors[-1, -1] == 1


def test_invalid_values_and_ids_fail_in_python(confidence):
    bad = confidence.copy()
    bad.data[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mi.AlternatingLeastSquares(factors=3).fit(bad, show_progress=False)
    with pytest.raises(TypeError, match="complex"):
        mi.AlternatingLeastSquares(factors=3).fit(
            confidence.astype(np.complex128), show_progress=False
        )
    model, _ = als_pair(confidence, iterations=1)
    model.fit(confidence, show_progress=False)
    with pytest.raises(TypeError, match="integers"):
        model.recommend(1.5, confidence[1], N=2)
    with pytest.raises(IndexError, match="non-negative"):
        model.similar_items(-1)
    with pytest.raises(ValueError, match="non-negative"):
        model.recommend(1, confidence[1], N=-1)
