from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class KNNModel:
    version: int
    k: int
    method: str
    model: object
    sample_count: int


_MODEL_CACHE: Dict[tuple, KNNModel] = {}


def normalize_X(X_uint8: np.ndarray) -> np.ndarray:
    return np.asarray(X_uint8, dtype=np.float32) / 255.0


def validate_pixels(pixels):
    if not isinstance(pixels, list) or len(pixels) != 784:
        raise ValueError("pixels must be a list of 784 integers")
    arr = np.asarray(pixels, dtype=np.float32)
    if np.any(arr < 0) or np.any(arr > 255):
        raise ValueError("pixels must be in [0, 255]")
    return arr.reshape(1, -1) / 255.0


def majority_vote(labels: np.ndarray, distances: np.ndarray | None = None) -> tuple[int, float]:
    labels = labels.astype(np.int64, copy=False)
    if distances is None:
        counts = np.bincount(labels, minlength=10).astype(np.float32)
    else:
        distances = distances.astype(np.float32, copy=False)
        weights = 1.0 / (distances + 1e-6)
        counts = np.bincount(labels, weights=weights, minlength=10).astype(np.float32)
    pred = int(np.argmax(counts))
    confidence = float(counts[pred] / max(1e-6, counts.sum()))
    return pred, confidence


class KdTreeNode:
    def __init__(
        self,
        point: np.ndarray,
        index=None,
        indices: np.ndarray | None = None,
        left: "KdTreeNode" | None = None,
        right: "KdTreeNode" | None = None,
        axis=None,
    ):
        self.left = left
        self.right = right
        self.point = point
        self.axis = axis
        self.index = index
        self.indices = indices


class KdTree:
    def __init__(self, X: np.ndarray, leaf_size: int = 10):
        self.X = X
        self.k = self.X.shape[1]
        self.leaf_size = int(leaf_size)
        self.root = self._build(self.X, np.arange(len(X)), depth=0)

    def _build(self, X: np.ndarray, indices: np.ndarray, depth: int):
        if len(X) == 0:
            return None

        if len(X) <= self.leaf_size:
            return KdTreeNode(point=X, indices=indices)

        axis = depth % self.k
        sorted_indices = np.argsort(X[:, axis])
        X_sorted = X[sorted_indices]
        indices_sorted = indices[sorted_indices]

        median = len(X_sorted) // 2
        X_1 = X_sorted[:median]
        X_2 = X_sorted[median + 1 :]

        return KdTreeNode(
            point=X_sorted[median],
            index=indices_sorted[median],
            left=self._build(X_1, indices_sorted[:median], depth + 1),
            right=self._build(X_2, indices_sorted[median + 1 :], depth + 1),
            axis=axis,
        )

    def query(self, x: np.ndarray, p: int = 1, metric: str = "Minkowski", k: int = 1):
        import heapq

        best = []

        def _distance(x1: np.ndarray, x2: np.ndarray, p: int = 1, metric: str = "Minkowski"):
            if metric == "Minkowski":
                return np.linalg.norm(x=x1 - x2, ord=p)
            return None

        def _search(node: KdTreeNode | None):
            if node is None:
                return

            if node.left is None and node.right is None:
                points = node.point if len(node.point.shape) > 1 else node.point.reshape(1, -1)
                for i, point in enumerate(points):
                    d = _distance(x, point, p=p, metric=metric)
                    idx = node.indices[i]

                    if len(best) < k:
                        heapq.heappush(best, (-d, idx))
                    elif -best[0][0] > d:
                        heapq.heapreplace(best, (-d, idx))
                return

            d = _distance(x, node.point, p, metric)
            if len(best) < k:
                heapq.heappush(best, (-d, node.index))
            elif -best[0][0] > d:
                heapq.heapreplace(best, (-d, node.index))

            axis = node.axis
            diff = x[axis] - node.point[axis]

            if diff < 0:
                _search(node.left)
                if len(best) < k or abs(diff) < -best[0][0]:
                    _search(node.right)
            else:
                _search(node.right)
                if len(best) < k or abs(diff) < -best[0][0]:
                    _search(node.left)

        _search(self.root)

        result_indices = sorted(
            [idx for _, idx in best],
            key=lambda i: _distance(x, self.X[i], p=p, metric=metric),
        )
        return np.array(result_indices[:k])


class ExactVectorKNN:
    def __init__(self, n_neighbors: int = 3, leaf_size: int = 30, chunk_size: int = 8192):
        self.n_neighbors = int(n_neighbors)
        self.leaf_size = int(leaf_size)
        self.chunk_size = int(chunk_size)
        self.X = None
        self.y = None
        self._tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.uint8)
        self._tree = KdTree(self.X, leaf_size=self.leaf_size)
        return self

    def _nearest_one(self, q: np.ndarray):
        k = min(self.n_neighbors, len(self.y))
        if k == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        indices = self._tree.query(q, p=2, k=k)
        if len(indices) == 0:
            return indices, np.array([], dtype=np.float32)
        distances = np.linalg.norm(self.X[indices] - q, ord=2, axis=1).astype(np.float32)
        order = np.argsort(distances)
        return indices[order], distances[order]

    def predict_one(self, q: np.ndarray):
        idx, distances = self._nearest_one(q)
        labels = self.y[idx]
        return majority_vote(labels, distances)

    def predict(self, Xq: np.ndarray):
        Xq = np.atleast_2d(np.asarray(Xq, dtype=np.float32))
        return np.asarray([self.predict_one(q)[0] for q in Xq], dtype=np.uint8)

    def predict_proba(self, Xq: np.ndarray):
        Xq = np.atleast_2d(np.asarray(Xq, dtype=np.float32))
        rows = []
        for q in Xq:
            idx, distances = self._nearest_one(q)
            labels = self.y[idx].astype(np.int64, copy=False)
            weights = 1.0 / (distances.astype(np.float32) + 1e-6)
            counts = np.bincount(labels, weights=weights, minlength=10).astype(np.float32)
            rows.append(counts / max(1e-6, counts.sum()))
        return np.vstack(rows)


class LSHIndex:
    def __init__(self, n_planes: int = 10, n_tables: int = 5, random_state: int = 42):
        self.n_planes = int(n_planes)
        self.n_tables = int(n_tables)
        self.rng = np.random.default_rng(random_state)
        self.planes = []
        self.tables = []
        self.X = None

    def fit(self, X: np.ndarray):
        self.X = X
        dim = X.shape[1]
        self.planes = self.rng.standard_normal((self.n_tables, self.n_planes, dim)).astype(X.dtype)

        for t in range(self.n_tables):
            projections = X @ self.planes[t].T
            bits = (projections >= 0).astype(np.uint8)
            table = {}
            for i, b in enumerate(bits):
                key = b.tobytes()
                table.setdefault(key, []).append(i)
            self.tables.append(table)
        return self

    def query(self, x: np.ndarray, k: int, p: int = 2) -> np.ndarray:
        candidates = set()
        for t, table in enumerate(self.tables):
            bits = (x @ self.planes[t].T >= 0).astype(np.uint8)
            key = bits.tobytes()
            candidates.update(table.get(key, []))

        if not candidates:
            candidates = set(range(len(self.X)))

        cands = np.array(list(candidates), dtype=np.int64)
        if len(cands) == 0:
            return np.array([], dtype=np.int64)
        dists = np.linalg.norm(self.X[cands] - x, ord=p, axis=1)

        top_k_idx = np.argpartition(dists, min(k, len(dists)) - 1)[:k]
        top_k = cands[top_k_idx[np.argsort(dists[top_k_idx])]]

        return top_k


class LSHIndexClassifier:
    def __init__(self, n_neighbors: int = 3, n_planes: int = 10, n_tables: int = 5, seed: int = 42):
        self.n_neighbors = int(n_neighbors)
        self.n_planes = int(n_planes)
        self.n_tables = int(n_tables)
        self.seed = int(seed)
        self.index = None
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.uint8)
        self.index = LSHIndex(self.n_planes, self.n_tables, self.seed).fit(self.X)
        return self

    def _nearest_one(self, q: np.ndarray):
        k = min(self.n_neighbors, len(self.y))
        if k == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        indices = self.index.query(q, k=k, p=2)
        if len(indices) == 0:
            return indices, np.array([], dtype=np.float32)
        distances = np.linalg.norm(self.X[indices] - q, ord=2, axis=1).astype(np.float32)
        order = np.argsort(distances)
        return indices[order], distances[order]

    def predict_one(self, q: np.ndarray):
        idx, distances = self._nearest_one(q)
        labels = self.y[idx]
        return majority_vote(labels, distances)

    def predict(self, Xq: np.ndarray):
        Xq = np.atleast_2d(np.asarray(Xq, dtype=np.float32))
        return np.asarray([self.predict_one(q)[0] for q in Xq], dtype=np.uint8)

    def predict_proba(self, Xq: np.ndarray):
        Xq = np.atleast_2d(np.asarray(Xq, dtype=np.float32))
        rows = []
        for q in Xq:
            idx, distances = self._nearest_one(q)
            labels = self.y[idx].astype(np.int64, copy=False)
            weights = 1.0 / (distances.astype(np.float32) + 1e-6)
            counts = np.bincount(labels, weights=weights, minlength=10).astype(np.float32)
            rows.append(counts / max(1e-6, counts.sum()))
        return np.vstack(rows)


def build_model(version: int, X_uint8: np.ndarray, y: np.ndarray, k: int, method: str):
    method = "lsh" if method == "lsh" else "kd_tree"
    key = (int(version), int(k), method)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    X = normalize_X(X_uint8)
    if method == "lsh":
        model = LSHIndexClassifier(n_neighbors=k).fit(X, y)
    else:
        model = ExactVectorKNN(n_neighbors=k).fit(X, y)

    wrapped = KNNModel(version=version, k=int(k), method=method, model=model, sample_count=len(y))
    _MODEL_CACHE.clear()
    _MODEL_CACHE[key] = wrapped
    return wrapped


def predict_digit(version: int, X_uint8: np.ndarray, y: np.ndarray, pixels: List[int], k: int, method: str):
    query = validate_pixels(pixels)
    model = build_model(version, X_uint8, y, int(k), method)
    start = time.perf_counter()

    if hasattr(model.model, "predict_one"):
        pred, confidence = model.model.predict_one(query[0])
    else:
        pred = int(model.model.predict(query)[0])
        try:
            proba = model.model.predict_proba(query)[0]
            confidence = float(proba[pred])
        except Exception:
            confidence = 1.0

    latency_ms = max(1, int((time.perf_counter() - start) * 1000))
    return int(pred), float(confidence), latency_ms


def balanced_indices(y: np.ndarray, total: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.arange(10)
    per_label = max(1, total // 10)
    remainder = max(0, total - per_label * 10)
    selected = []
    for label in labels:
        idx = np.where(y == label)[0]
        if len(idx) == 0:
            continue
        take = min(len(idx), per_label + (1 if label < remainder else 0))
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def tune_knn(version: int, X_uint8: np.ndarray, y: np.ndarray, sample_count: int, method: str, k_values: List[int]):
    # Currently only tunes k (n_neighbors). LSH has additional parameters (n_planes, n_tables)
    # and kd-tree has leaf_size that could be tuned for better performance optimization.
    method = "lsh" if method == "lsh" else "kd_tree"
    sample_count = int(sample_count or 500)
    sample_count = max(20, min(sample_count, len(y)))
    val_count = max(10, sample_count // 5)
    total_needed = min(len(y), sample_count + val_count)

    idx = balanced_indices(y, total_needed, seed=42)
    if len(idx) < 20:
        raise RuntimeError("Not enough samples to tune. Import data first.")

    train_idx = idx[:min(sample_count, len(idx) - 10)]
    val_idx = idx[len(train_idx):]
    if len(val_idx) == 0:
        val_idx = train_idx

    X_train = normalize_X(X_uint8[train_idx])
    y_train = y[train_idx]
    X_val = normalize_X(X_uint8[val_idx])
    y_val = y[val_idx]

    results = []
    for k in sorted(set(int(v) for v in k_values if int(v) > 0)):
        k_eff = min(k, len(y_train))
        if method == "lsh":
            # LSH parameters n_planes=10, n_tables=5 are hardcoded; consider as tuning candidates for future work
            model = LSHIndexClassifier(n_neighbors=k_eff).fit(X_train, y_train)
        else:
            # KdTree leaf_size=30 is hardcoded; could be tuned for different dataset sizes
            model = ExactVectorKNN(n_neighbors=k_eff).fit(X_train, y_train)

        start = time.perf_counter()
        preds = model.predict(X_val)
        total_latency = (time.perf_counter() - start) * 1000
        avg_latency = total_latency / max(1, len(y_val))
        acc = float(accuracy_score(y_val, preds))
        f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        precision = float(precision_score(y_val, preds, average="macro", zero_division=0))
        recall = float(recall_score(y_val, preds, average="macro", zero_division=0))
        results.append({
            "k": int(k),
            "method": method,
            "accuracy": acc,
            "f1Score": f1,
            "precision": precision,
            "recall": recall,
            "avgLatencyMs": float(avg_latency),
            "trainingSamples": int(len(y_train)),
            "evaluatedSamples": int(len(y_val)),
            "datasetVersion": int(version),
        })

    ranked = sorted(results, key=lambda r: (-r["accuracy"], r["avgLatencyMs"], -r["f1Score"]))[:5]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    return {
        "jobId": str(uuid.uuid4()),
        "datasetVersion": int(version),
        "sampleCount": int(sample_count),
        "method": method,
        "topResults": ranked,
    }
