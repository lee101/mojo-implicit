"""C ABI kernels for implicit-feedback matrix factorization."""

from std.algorithm import parallelize
from std.atomic import Atomic
from std.builtin._startup import _ensure_runtime_init
from std.math import exp, sqrt
from std.sys import simd_width_of as simdwidthof

comptime W = simdwidthof[DType.float64]()
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


def cholesky(a: Ptr, n: Int) -> Bool:
    for i in range(n):
        for j in range(i + 1):
            var value = a[i * n + j]
            var acc = SIMD[DType.float64, W](0.0)
            var k = 0
            while k + W <= j:
                acc += (a + i * n).load[width=W](k) * (a + j * n).load[width=W](
                    k
                )
                k += W
            value -= acc.reduce_add()
            while k < j:
                value -= a[i * n + k] * a[j * n + k]
                k += 1
            if i == j:
                if value <= 0.0:
                    return False
                a[i * n + i] = sqrt(value)
            else:
                a[i * n + j] = value / a[j * n + j]
    return True


def cholesky_solve(a: Ptr, b: Ptr, n: Int):
    for i in range(n):
        var value = b[i]
        var acc = SIMD[DType.float64, W](0.0)
        var k = 0
        while k + W <= i:
            acc += (a + i * n).load[width=W](k) * b.load[width=W](k)
            k += W
        value -= acc.reduce_add()
        while k < i:
            value -= a[i * n + k] * b[k]
            k += 1
        b[i] = value / a[i * n + i]
    for ri in range(n):
        var i = n - 1 - ri
        var value = b[i]
        for k in range(i + 1, n):
            value -= a[k * n + i] * b[k]
        b[i] = value / a[i * n + i]


@export("mi_gram")
def mi_gram(x_addr: Int, dst_addr: Int, rows: Int, factors: Int) abi("C"):
    var x = fp(x_addr)
    var dst = fp(dst_addr)
    var i = 0
    var total = factors * factors
    while i + W <= total:
        dst.store(i, SIMD[DType.float64, W](0.0))
        i += W
    while i < total:
        dst[i] = 0.0
        i += 1
    for row in range(rows):
        var base = x + row * factors
        for i in range(factors):
            var row_base = dst + i * factors
            var scaled = SIMD[DType.float64, W](base[i])
            var j = 0
            while j + W <= i + 1:
                row_base.store(
                    j,
                    row_base.load[width=W](j) + scaled * base.load[width=W](j),
                )
                j += W
            while j <= i:
                dst[i * factors + j] += base[i] * base[j]
                j += 1
    for i in range(factors):
        for j in range(i + 1, factors):
            dst[i * factors + j] = dst[j * factors + i]


def solve_row(
    indptr: IPtr,
    indices: IPtr,
    data: Ptr,
    result: Ptr,
    fixed: Ptr,
    gram: Ptr,
    matrix_work: Ptr,
    vector_work: Ptr,
    row: Int,
    n_factors: Int,
    regularization: Float64,
) -> Bool:
    if indptr[row] == indptr[row + 1]:
        var result_row = result + row * n_factors
        var j = 0
        while j + W <= n_factors:
            result_row.store(j, SIMD[DType.float64, W](0.0))
            j += W
        while j < n_factors:
            result_row[j] = 0.0
            j += 1
        return True

    var i = 0
    var matrix_size = n_factors * n_factors
    while i + W <= matrix_size:
        matrix_work.store(i, gram.load[width=W](i))
        i += W
    while i < matrix_size:
        matrix_work[i] = gram[i]
        i += 1
    i = 0
    while i + W <= n_factors:
        vector_work.store(i, SIMD[DType.float64, W](0.0))
        i += W
    while i < n_factors:
        vector_work[i] = 0.0
        i += 1
    for i in range(n_factors):
        matrix_work[i * n_factors + i] += regularization

    for offset in range(Int(indptr[row]), Int(indptr[row + 1])):
        var item = Int(indices[offset])
        var confidence = data[offset]
        var base = fixed + item * n_factors
        if confidence > 0.0:
            var confidence_vec = SIMD[DType.float64, W](confidence)
            var j = 0
            while j + W <= n_factors:
                vector_work.store(
                    j,
                    vector_work.load[width=W](j)
                    + confidence_vec * base.load[width=W](j),
                )
                j += W
            while j < n_factors:
                vector_work[j] += confidence * base[j]
                j += 1
        else:
            confidence = -confidence
        var weight = confidence - 1.0
        for i in range(n_factors):
            var scaled = SIMD[DType.float64, W](weight * base[i])
            var matrix_row = matrix_work + i * n_factors
            var j = 0
            while j + W <= i + 1:
                matrix_row.store(
                    j,
                    matrix_row.load[width=W](j)
                    + scaled * base.load[width=W](j),
                )
                j += W
            while j <= i:
                matrix_row[j] += weight * base[i] * base[j]
                j += 1
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            matrix_work[i * n_factors + j] = matrix_work[j * n_factors + i]
    if not cholesky(matrix_work, n_factors):
        return False
    cholesky_solve(matrix_work, vector_work, n_factors)
    var result_row = result + row * n_factors
    i = 0
    while i + W <= n_factors:
        result_row.store(i, vector_work.load[width=W](i))
        i += W
    while i < n_factors:
        result_row[i] = vector_work[i]
        i += 1
    return True


def least_squares_parallel(
    indptr: IPtr,
    indices: IPtr,
    data: Ptr,
    result: Ptr,
    fixed: Ptr,
    gram: Ptr,
    matrix_work: Ptr,
    vector_work: Ptr,
    failure_work: IPtr,
    rows: Int,
    n_factors: Int,
    workers: Int,
    regularization: Float64,
) -> Int:
    for worker in range(workers):
        failure_work[worker] = 0

    @parameter
    def run_worker(worker: Int):
        var worker_matrix = matrix_work + worker * n_factors * n_factors
        var worker_vector = vector_work + worker * n_factors
        var row = worker
        while row < rows:
            if not solve_row(
                indptr,
                indices,
                data,
                result,
                fixed,
                gram,
                worker_matrix,
                worker_vector,
                row,
                n_factors,
                regularization,
            ):
                failure_work[worker] = Int64(row + 1)
                return
            row += workers

    parallelize[run_worker](workers)
    var first_failure = rows + 1
    for worker in range(workers):
        if (
            failure_work[worker] != 0
            and Int(failure_work[worker]) < first_failure
        ):
            first_failure = Int(failure_work[worker])
    return 0 if first_failure == rows + 1 else first_failure


@export("mi_least_squares")
def mi_least_squares(
    indptr_addr: Int,
    indices_addr: Int,
    data_addr: Int,
    factors_addr: Int,
    fixed_addr: Int,
    gram_addr: Int,
    matrix_work_addr: Int,
    vector_work_addr: Int,
    failure_work_addr: Int,
    rows: Int,
    n_factors: Int,
    workers: Int,
    regularization: Float64,
) abi("C") -> Int:
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var data = fp(data_addr)
    var result = fp(factors_addr)
    var fixed = fp(fixed_addr)
    var gram = fp(gram_addr)
    var matrix_work = fp(matrix_work_addr)
    var vector_work = fp(vector_work_addr)
    var failure_work = ip(failure_work_addr)

    if workers > 1:
        _ensure_runtime_init()
        return least_squares_parallel(
            indptr,
            indices,
            data,
            result,
            fixed,
            gram,
            matrix_work,
            vector_work,
            failure_work,
            rows,
            n_factors,
            workers,
            regularization,
        )

    for row in range(rows):
        if not solve_row(
            indptr,
            indices,
            data,
            result,
            fixed,
            gram,
            matrix_work,
            vector_work,
            row,
            n_factors,
            regularization,
        ):
            return row + 1
    return 0


def contains(indptr: IPtr, indices: IPtr, row: Int, item: Int64) -> Bool:
    var lo = Int(indptr[row])
    var hi = Int(indptr[row + 1])
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if indices[mid] < item:
            lo = mid + 1
        else:
            hi = mid
    return lo < Int(indptr[row + 1]) and indices[lo] == item


def bpr_apply(
    user: Ptr,
    liked: Ptr,
    disliked: Ptr,
    factors: Int,
    learning_rate: Float64,
    regularization: Float64,
) -> Bool:
    var width = factors + 1
    var liked_acc = SIMD[DType.float64, W](0.0)
    var disliked_acc = SIMD[DType.float64, W](0.0)
    var j = 0
    while j + W <= width:
        var user_vec = user.load[width=W](j)
        liked_acc += user_vec * liked.load[width=W](j)
        disliked_acc += user_vec * disliked.load[width=W](j)
        j += W
    var score = liked_acc.reduce_add() - disliked_acc.reduce_add()
    while j < width:
        score += user[j] * liked[j] - user[j] * disliked[j]
        j += 1
    var z = 1.0 / (1.0 + exp(score))
    var learning_vec = SIMD[DType.float64, W](learning_rate)
    var regularization_vec = SIMD[DType.float64, W](regularization)
    var z_vec = SIMD[DType.float64, W](z)
    j = 0
    while j + W <= factors:
        var old_user = user.load[width=W](j)
        var old_liked = liked.load[width=W](j)
        var old_disliked = disliked.load[width=W](j)
        user.store(
            j,
            old_user
            + learning_vec
            * (
                z_vec * (old_liked - old_disliked)
                - regularization_vec * old_user
            ),
        )
        liked.store(
            j,
            old_liked
            + learning_vec
            * (z_vec * old_user - regularization_vec * old_liked),
        )
        disliked.store(
            j,
            old_disliked
            + learning_vec
            * (-z_vec * old_user - regularization_vec * old_disliked),
        )
        j += W
    while j < factors:
        var old_user = user[j]
        user[j] += learning_rate * (
            z * (liked[j] - disliked[j]) - regularization * user[j]
        )
        liked[j] += learning_rate * (z * old_user - regularization * liked[j])
        disliked[j] += learning_rate * (
            -z * old_user - regularization * disliked[j]
        )
        j += 1
    liked[factors] += learning_rate * (z - regularization * liked[factors])
    disliked[factors] += learning_rate * (
        -z - regularization * disliked[factors]
    )
    return z < 0.5


def acquire_lock(lock: IPtr):
    while True:
        var expected = Int64(0)
        if Atomic[DType.int64].compare_exchange(lock, expected, Int64(1)):
            return


def release_lock(lock: IPtr):
    Atomic[DType.int64].store(lock, Int64(0))


def bpr_parallel(
    userids: IPtr,
    itemids: IPtr,
    indptr: IPtr,
    liked_samples: IPtr,
    disliked_samples: IPtr,
    users: Ptr,
    items: Ptr,
    locks: IPtr,
    counts: IPtr,
    samples: Int,
    factors: Int,
    user_count: Int,
    workers: Int,
    learning_rate: Float64,
    regularization: Float64,
    verify_negative: Int,
) -> Int:
    var width = factors + 1

    @parameter
    def run_worker(worker: Int):
        var start = samples * worker // workers
        var end = samples * (worker + 1) // workers
        var correct = 0
        var skipped = 0
        for sample in range(start, end):
            var liked_index = Int(liked_samples[sample])
            var disliked_index = Int(disliked_samples[sample])
            var user_id = Int(userids[liked_index])
            var liked_id = Int(itemids[liked_index])
            var disliked_id = Int(itemids[disliked_index])
            if verify_negative != 0 and contains(
                indptr, itemids, user_id, Int64(disliked_id)
            ):
                skipped += 1
                continue
            var first_item = min(liked_id, disliked_id)
            var second_item = max(liked_id, disliked_id)
            acquire_lock(locks + user_id)
            acquire_lock(locks + user_count + first_item)
            if second_item != first_item:
                acquire_lock(locks + user_count + second_item)
            if bpr_apply(
                users + user_id * width,
                items + liked_id * width,
                items + disliked_id * width,
                factors,
                learning_rate,
                regularization,
            ):
                correct += 1
            if second_item != first_item:
                release_lock(locks + user_count + second_item)
            release_lock(locks + user_count + first_item)
            release_lock(locks + user_id)
        counts[worker] = Int64(correct)
        counts[workers + worker] = Int64(skipped)

    parallelize[run_worker](workers)
    var total_correct = 0
    for worker in range(workers):
        total_correct += Int(counts[worker])
    return total_correct


@export("mi_bpr_update")
def mi_bpr_update(
    userids_addr: Int,
    itemids_addr: Int,
    indptr_addr: Int,
    liked_samples_addr: Int,
    disliked_samples_addr: Int,
    users_addr: Int,
    items_addr: Int,
    skipped_addr: Int,
    locks_addr: Int,
    counts_addr: Int,
    samples: Int,
    factors: Int,
    user_count: Int,
    workers: Int,
    learning_rate: Float64,
    regularization: Float64,
    verify_negative: Int,
) abi("C") -> Int:
    var userids = ip(userids_addr)
    var itemids = ip(itemids_addr)
    var indptr = ip(indptr_addr)
    var liked_samples = ip(liked_samples_addr)
    var disliked_samples = ip(disliked_samples_addr)
    var users = fp(users_addr)
    var items = fp(items_addr)
    var skipped = ip(skipped_addr)
    var locks = ip(locks_addr)
    var counts = ip(counts_addr)
    var width = factors + 1
    var correct = 0
    skipped[0] = 0

    if workers > 1:
        _ensure_runtime_init()
        var total_correct = bpr_parallel(
            userids,
            itemids,
            indptr,
            liked_samples,
            disliked_samples,
            users,
            items,
            locks,
            counts,
            samples,
            factors,
            user_count,
            workers,
            learning_rate,
            regularization,
            verify_negative,
        )
        for worker in range(workers):
            skipped[0] += counts[workers + worker]
        return total_correct

    for sample in range(samples):
        var liked_index = Int(liked_samples[sample])
        var disliked_index = Int(disliked_samples[sample])
        var user_id = Int(userids[liked_index])
        var liked_id = Int(itemids[liked_index])
        var disliked_id = Int(itemids[disliked_index])
        if verify_negative != 0 and contains(
            indptr, itemids, user_id, Int64(disliked_id)
        ):
            skipped[0] += 1
            continue
        if bpr_apply(
            users + user_id * width,
            items + liked_id * width,
            items + disliked_id * width,
            factors,
            learning_rate,
            regularization,
        ):
            correct += 1
    return correct


def score_parallel(
    queries: Ptr,
    candidates: Ptr,
    candidate_ids: IPtr,
    dst: Ptr,
    query_count: Int,
    candidate_count: Int,
    factors: Int,
    workers: Int,
):
    var total = query_count * candidate_count

    @parameter
    def run_worker(worker: Int):
        var pos = total * worker // workers
        var end = total * (worker + 1) // workers
        if pos == end:
            return
        var q = pos // candidate_count
        var c = pos - q * candidate_count
        while pos < end:
            var candidate = Int(candidate_ids[c])
            dst[pos] = dot(
                queries + q * factors,
                candidates + candidate * factors,
                factors,
            )
            pos += 1
            c += 1
            if c == candidate_count:
                c = 0
                q += 1

    parallelize[run_worker](workers)


@export("mi_score")
def mi_score(
    queries_addr: Int,
    candidates_addr: Int,
    candidate_ids_addr: Int,
    dst_addr: Int,
    query_count: Int,
    candidate_count: Int,
    factors: Int,
    workers: Int,
) abi("C"):
    var queries = fp(queries_addr)
    var candidates = fp(candidates_addr)
    var candidate_ids = ip(candidate_ids_addr)
    var dst = fp(dst_addr)
    var total = query_count * candidate_count
    if workers > 1:
        _ensure_runtime_init()
        score_parallel(
            queries,
            candidates,
            candidate_ids,
            dst,
            query_count,
            candidate_count,
            factors,
            workers,
        )
        return
    for q in range(query_count):
        for c in range(candidate_count):
            var candidate = Int(candidate_ids[c])
            dst[q * candidate_count + c] = dot(
                queries + q * factors,
                candidates + candidate * factors,
                factors,
            )
