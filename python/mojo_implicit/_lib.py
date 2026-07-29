"""ctypes bindings for the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "kernels.mojo")
LIB = os.environ.get("MOJO_IMPLICIT_LIB") or os.path.join(
    ROOT, "dist", "libmojo-implicit.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mi_gram": ([I, I, I, I], None),
    "mi_least_squares": ([I] * 12 + [F], I),
    "mi_bpr_update": ([I] * 14 + [F, F, I], I),
    "mi_score": ([I] * 8, None),
}

_PARALLEL_WORK_THRESHOLD = 100_000


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    mojo = shutil.which("mojo")
    if not mojo:
        raise BuildError("mojo not found; run `pixi run build` or set MOJO_IMPLICIT_LIB")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64(value, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(value: np.ndarray) -> int:
    address = int(value.ctypes.data)
    if value.size and not address:
        raise ValueError("cannot pass a non-empty array with a null data pointer")
    return address


def _matrix_2d(name: str, value) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-dimensional array")
    return array


def _validate_csr(matrix) -> None:
    if getattr(matrix, "ndim", None) != 2:
        raise ValueError("matrix must be 2-dimensional")
    rows, columns = matrix.shape
    indptr = np.asarray(matrix.indptr)
    indices = np.asarray(matrix.indices)
    data = np.asarray(matrix.data)
    if indptr.ndim != 1 or len(indptr) != rows + 1:
        raise ValueError("CSR indptr length must equal the row count plus one")
    if indices.ndim != 1 or data.ndim != 1 or len(indices) != len(data):
        raise ValueError("CSR indices and data must be one-dimensional and equally sized")
    if len(indptr) and (
        indptr[0] != 0
        or indptr[-1] != len(indices)
        or np.any(indptr[1:] < indptr[:-1])
    ):
        raise ValueError("CSR indptr is inconsistent with its data")
    if len(indices) and (np.any(indices < 0) or np.any(indices >= columns)):
        raise ValueError("CSR column index is outside the matrix shape")


def worker_count(work_items: int, work: int, num_threads: int = 0) -> int:
    if work_items < 2 or work < _PARALLEL_WORK_THRESHOLD or num_threads == 1:
        return 1
    if num_threads > 1:
        available = num_threads
    else:
        try:
            available = len(os.sched_getaffinity(0))
        except AttributeError:
            available = os.cpu_count() or 1
    return max(1, min(work_items, available))


def gram(factors: np.ndarray) -> np.ndarray:
    factors = f64(_matrix_2d("factors", factors))
    if factors.shape[1] == 0:
        raise ValueError("factors must have at least one column")
    result = np.empty((factors.shape[1], factors.shape[1]), dtype=np.float64)
    lib().mi_gram(addr(factors), addr(result), *factors.shape)
    return result


def least_squares(
    matrix,
    result: np.ndarray,
    fixed: np.ndarray,
    regularization: float,
    num_threads: int = 0,
) -> None:
    _validate_csr(matrix)
    fixed_input = _matrix_2d("fixed", fixed)
    result_input = _matrix_2d("result", result)
    if fixed_input.shape[0] != matrix.shape[1]:
        raise ValueError("fixed row count must equal the matrix column count")
    if result_input.shape != (matrix.shape[0], fixed_input.shape[1]):
        raise ValueError("result shape must be (matrix rows, factor count)")
    if not np.issubdtype(result_input.dtype, np.floating):
        raise TypeError("result dtype must be floating point")
    if fixed_input.shape[1] == 0:
        raise ValueError("factor count must be positive")
    if not np.isfinite(regularization):
        raise ValueError("regularization must be finite")
    indptr = i64(matrix.indptr)
    indices = i64(matrix.indices)
    data = f64(matrix.data)
    fixed = f64(fixed_input)
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(fixed)):
        raise ValueError("matrix data and fixed factors must be finite")
    result64 = f64(result, copy=result.dtype != np.float64)
    fixed_gram = gram(fixed)
    n_factors = fixed.shape[1]
    workers = worker_count(
        matrix.shape[0],
        matrix.shape[0] * n_factors * n_factors,
        num_threads,
    )
    matrix_work = np.empty((workers, n_factors, n_factors), dtype=np.float64)
    vector_work = np.empty((workers, n_factors), dtype=np.float64)
    failure_work = np.empty(workers, dtype=np.int64)
    failed = lib().mi_least_squares(
        addr(indptr),
        addr(indices),
        addr(data),
        addr(result64),
        addr(fixed),
        addr(fixed_gram),
        addr(matrix_work),
        addr(vector_work),
        addr(failure_work),
        matrix.shape[0],
        n_factors,
        workers,
        regularization,
    )
    if failed:
        raise ValueError(
            f"Cholesky factorization failed on row {failed - 1}; "
            "try increasing regularization"
        )
    if result64 is not result:
        result[...] = result64


def score(
    queries: np.ndarray,
    candidates: np.ndarray,
    ids: np.ndarray,
    num_threads: int = 0,
) -> np.ndarray:
    queries = f64(_matrix_2d("queries", np.atleast_2d(queries)))
    candidates = f64(_matrix_2d("candidates", candidates))
    raw_ids = np.asarray(ids)
    if raw_ids.ndim != 1 or not np.issubdtype(raw_ids.dtype, np.integer):
        raise TypeError("ids must be a one-dimensional integer array")
    ids = i64(raw_ids)
    if queries.shape[1] == 0 or candidates.shape[1] != queries.shape[1]:
        raise ValueError("queries and candidates must have the same positive factor count")
    if len(ids) and (np.any(ids < 0) or np.any(ids >= len(candidates))):
        raise IndexError("candidate id is outside the candidates array")
    result = np.empty((queries.shape[0], len(ids)), dtype=np.float64)
    workers = worker_count(
        result.size,
        result.size * queries.shape[1],
        num_threads,
    )
    lib().mi_score(
        addr(queries),
        addr(candidates),
        addr(ids),
        addr(result),
        queries.shape[0],
        len(ids),
        queries.shape[1],
        workers,
    )
    return result
