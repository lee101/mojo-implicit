# mojo-implicit

`mojo-implicit` is a standalone Mojo port of the compute-heavy matrix
factorization paths in Ben Frederickson's
[`implicit`](https://github.com/benfred/implicit) Python package. It provides
weighted implicit-feedback Alternating Least Squares (ALS) and Bayesian
Personalized Ranking (BPR), with a NumPy/SciPy Python API matching the covered
upstream class names and call signatures.

This is an independent port, not a binding to upstream. The test suite installs
the real `implicit` 0.7.3 package and compares results against it.

## Coverage

Implemented:

- `AlternatingLeastSquares`: weighted exact least-squares fitting, positive and
  negative confidences, `alpha`, training loss and callbacks, user/item
  recalculation, partial user/item updates, and recommendation explanations.
- `BayesianPersonalizedRanking`: pairwise logistic SGD, item bias factors,
  optional verified negative sampling, and the upstream callback contract.
- Shared scalar and batch `recommend`, `similar_items`, `similar_users`,
  candidate restriction, seen-item filtering, extra filters, and `.npz`
  `save`/`load`.
- `float32` and `float64` model storage.

Not implemented:

- CUDA/GPU models, logistic matrix factorization, item-item nearest-neighbor
  recommenders, approximate-nearest-neighbor adapters, and evaluation helpers.
- OpenMP training. ALS rows and the scoring grid are independent, so the Python
  shim splits them into contiguous spans and runs them on a `ThreadPoolExecutor`
  (at most eight workers); `num_threads` controls the span count and small jobs
  stay serial. BPR stays serial: each sample takes three spin locks around a
  handful of fused multiply-adds, so it is synchronisation bound.
- Upstream's three-step conjugate-gradient ALS solver. `use_cg` and
  `use_native` are accepted, while this port always uses the more accurate
  Cholesky solve.
- Progress-bar rendering. `show_progress` is accepted and callbacks still run.

Passing `use_gpu=True` raises `NotImplementedError` rather than silently moving
work to another implementation.

No GPU path was added. The implemented kernels operate on host-owned NumPy
buffers, and adding a device implementation is outside this port's scope.

## Install

The checked-in Pixi environment pins the Mojo nightly used by this source and
includes NumPy, SciPy, pytest, and upstream `implicit`:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-implicit.so`. Set `MOJO_IMPLICIT_LIB` to use a
prebuilt library at another path.

## Usage

```python
import numpy as np
from scipy.sparse import csr_matrix

from mojo_implicit.als import AlternatingLeastSquares
from mojo_implicit.bpr import BayesianPersonalizedRanking

user_items = csr_matrix(
    np.array(
        [
            [3.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 4.0],
        ]
    )
)

als = AlternatingLeastSquares(
    factors=8, regularization=0.05, iterations=5, random_state=0
)
als.fit(user_items, show_progress=False)
item_ids, scores = als.recommend(0, user_items[0], N=2)

bpr = BayesianPersonalizedRanking(
    factors=8, learning_rate=0.05, iterations=20, random_state=0
)
bpr.fit(user_items, show_progress=False)
```

Run it inside the environment with `pixi run python example.py`.

## Benchmarks

Measured by `pixi run bench` on an Intel Xeon E5-2697 v4 machine running Linux
6.8.0-136-generic. These are best-of-three wall times; upstream was allowed to
use its default OpenMP thread count, with BLAS limited to one thread as upstream
recommends. “Upstream / Mojo” above 1 means Mojo was faster.

| Case | mojo-implicit | implicit 0.7.3 | Upstream / Mojo |
|---|---:|---:|---:|
| ALS.fit exact, 8k x 2.5k, 120k nnz, f=24, 2 it | 31.78 ms | 175.65 ms | 5.53x |
| BPR.fit, 15k x 4k, 225k nnz, f=32, 8 epochs | 196.17 ms | 223.72 ms | 1.14x |
| recommend, 400 x 10k candidates, f=64, N=20 | 64.83 ms | 105.28 ms | 1.62x |

In this run Mojo is 5.53x faster for exact ALS fitting, 1.14x faster for BPR
fitting, and 1.62x faster for batch recommendation.

Reproduce the table with:

```bash
pixi run bench
```

## How it works

NumPy owns all allocations. CSR `indptr` and `indices` are normalized to
contiguous `int64`; confidence values, factors, Gram matrices, and scratch
buffers cross the C ABI as row-major contiguous `float64` buffers. The Python
wrapper passes their addresses as 64-bit integers through `ctypes`, and the
single Mojo compilation unit reconstructs mutable pointers using
`AnyOrigin[mut=True]`.

ALS builds each weighted normal equation from a shared factor Gram matrix and
solves it in Mojo with an in-place Cholesky factorization. `mi_least_squares_range`
solves a contiguous row span with its own scratch buffers, and the shim fans the
rows out across a thread pool; the result is bitwise identical to a serial pass
because the rows never interact. BPR generates reproducible sample indices in
NumPy; updates run in sample order under ordered user/item row locks, so the
factors do not depend on the worker count. Recommendation and similarity calls
use SIMD scoring over a flat position range that the shim splits across the same
pool; Python handles sparse filters and stable top-N selection.
