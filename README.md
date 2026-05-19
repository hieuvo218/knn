# KNearestNeighbors

A KNN classifier supporting two index backends: **KD-Tree** (exact) and **LSH** (approximate). Designed for high-dimensional datasets like MNIST.


Tech stack:
- Frontend: React + Vite
- Backend: Spring Boot Java 17
- ML: Python + Flask + scikit-learn kNN
- Database: PostgreSQL 15
- DevOps: Docker Compose

## Start fresh
Open 3 terminal

at terminal 1, run

```bash
docker compose down -v
docker compose up --build
```

at terminal 2, run

```bash
docker compose --profile import run --rm mnist-importer
```

at terminal 3, run

```bash
cd frontend
npm install
npm run dev
```

---

## Index Backends

### KD-Tree (exact)
Builds a binary tree by recursively splitting the feature space. Guarantees exact nearest neighbors but degrades on high-dimensional data (curse of dimensionality).

**Best for:** Low-to-medium dimensionality (< 20 dims), small datasets, or when accuracy is critical.

```python
knn = KNearestNeighbors(
    num_neighbors=5,
    p=2,
    index="kdtree",
    leaf_size=30       # nodes per leaf; increase to reduce tree depth, decrease for faster queries
)
```

| Parameter   | Default | Effect |
|-------------|---------|--------|
| `leaf_size` | `30`    | Higher → smaller tree, faster build, slower query. Lower → deeper tree, faster query on small N. |
| `p`         | `2`     | Minkowski order. `p=2` is Euclidean, `p=1` is Manhattan. |

---

### LSH — Locality Sensitive Hashing (approximate)
Projects data onto random hyperplanes and groups points by their binary hash. Only searches within matching buckets, making queries sublinear in N.

**Best for:** High-dimensional data (e.g. MNIST's 784 dims), large datasets (50k+), when speed matters more than exact results.

```python
knn = KNearestNeighbors(
    num_neighbors=5,
    p=2,
    index="lsh",
    lsh_n_planes=10,   # hyperplanes per table
    lsh_n_tables=5     # number of independent hash tables
)
```

| Parameter      | Default | Effect |
|----------------|---------|--------|
| `lsh_n_planes` | `10`    | Controls bucket granularity. More planes → smaller buckets → fewer candidates → faster but lower recall. |
| `lsh_n_tables` | `5`     | Number of independent hash tables. More tables → better recall, higher memory, slower build. |
| `p`            | `2`     | Distance metric used when ranking candidates. |

---

## Tuning LSH for Performance

LSH involves a **speed vs. recall tradeoff** controlled by `n_planes` and `n_tables`.

### Mental model
- Each table hashes every point into a bucket using `n_planes` random hyperplanes → `2^n_planes` possible buckets.
- At query time, the same hash is computed and only points in matching buckets across all tables are considered.
- **More tables** = more chances to find true neighbors (better recall).
- **More planes** = fewer points per bucket (faster query, higher miss rate).

### MNIST-specific starting point (N=60k, dim=784)

| Goal | `n_planes` | `n_tables` | Approx. candidates/query |
|------|-----------|-----------|--------------------------|
| Fast, ~90% recall | 12 | 5 | ~200–400 |
| Balanced | 10 | 8 | ~400–800 |
| High recall (~98%) | 8  | 15 | ~1000–2000 |

### Tuning steps

1. **Start with `n_planes=10, n_tables=5`** as a baseline.
2. **Measure recall** by comparing LSH predictions to brute-force KNN on a small validation set.
3. If recall is low → **increase `n_tables`** first (additive improvement, no bucket size change).
4. If query is too slow → **increase `n_planes`** to shrink buckets.
5. Avoid `n_planes > 16` — buckets become so small that empty-bucket fallback triggers frequently.

### Empty bucket fallback
When no candidates are found in any bucket, the implementation falls back to brute-force search over all points. This is a safety net — if it triggers often, your `n_planes` is too high or `n_tables` too low.

---

## Preprocessing MNIST

```python
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

X_train, X_test, y_train, y_test = preprocess_mnist(
    X_train, X_test, y_train, y_test,
    flatten=True,     # output shape: (N, 784)
    normalize=True    # scale pixels to [0, 1]
)
```

---

## Full Usage Example

```python
# 1. Preprocess
X_train, X_test, y_train, y_test = preprocess_mnist(
    X_train, X_test, y_train, y_test, flatten=True, normalize=True
)

# 2. Fit
knn = KNearestNeighbors(
    num_neighbors=5,
    p=2,
    index="lsh",
    lsh_n_planes=10,
    lsh_n_tables=8
)
knn.fit(X_train, y_train)

# 3. Predict
predictions = knn.predict(X_test)

# 4. Evaluate
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Choosing Between KD-Tree and LSH

| | KD-Tree | LSH |
|---|---|---|
| Accuracy | Exact | Approximate |
| Query speed on MNIST | Slow (high-dim) | Fast |
| Memory | Low | Moderate (scales with `n_tables`) |
| Tuning required | Minimal | Yes |
| Best dataset size | < 10k | 10k+ |

---

## Model Tuning (UI Feature)

The Model Dashboard includes a **Tuning** section for hyperparameter optimization. It evaluates multiple k values on a validation set and ranks results by accuracy, latency, and F1-score.

### Current Scope

**Parameters tuned:**
- `k` (n_neighbors): number of nearest neighbors to consider

**Parameters NOT currently tuned (hardcoded defaults):**
- **LSH**: `n_planes=10`, `n_tables=5` — these significantly impact speed vs. recall tradeoff but are fixed during tuning
- **KD-Tree**: `leaf_size=30` — affects tree structure and query performance but not exposed for tuning

### Known Limitations

1. **LSH tuning is incomplete**: Only varies k while holding n_planes and n_tables constant. To fully optimize LSH performance, users must manually adjust these in the backend code (`knn_core.py:245`).

2. **KD-Tree leaf_size not tunable**: The leaf_size parameter (default 30) affects build time and query performance but is not exposed for optimization.

3. **No multi-parameter grid search**: The tuner does not support grid search over combinations of LSH parameters (k × n_planes × n_tables), which would be computationally expensive but valuable for production tuning.

### Tuning Workflow

```
1. Set sample count (train/val split)
2. Choose method (kd-tree or lsh)
3. Specify k values to test (e.g., "1,3,5,7")
4. Run tune
5. View results table and chart
6. Activate best model with "Use this k" button
```

**To manually tune LSH parameters:**
1. Edit `backend/ml-service/knn_core.py` line 370: adjust `n_planes` and `n_tables` in LSHIndexClassifier constructor
2. Re-run tuning to evaluate the new configuration
3. Rebuild cache: POST `/cache/rebuild` to refresh the model