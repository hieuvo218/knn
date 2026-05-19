import time
import uuid
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


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


class ExactVectorKNN:
    def __init__(self, n_neighbors: int = 3, chunk_size: int = 8192):
        self.n_neighbors = int(n_neighbors)
        self.chunk_size = int(chunk_size)
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.uint8)
        return self

    def _nearest_one(self, q: np.ndarray):
        k = min(self.n_neighbors, len(self.y))
        best_d = np.full(k, np.inf, dtype=np.float32)
        best_i = np.full(k, -1, dtype=np.int64)

        for start in range(0, self.X.shape[0], self.chunk_size):
            block = self.X[start:start + self.chunk_size]
            diff = block - q
            dists = np.einsum("ij,ij->i", diff, diff, optimize=True)

            if len(dists) <= k:
                cand_local = np.arange(len(dists))
            else:
                cand_local = np.argpartition(dists, k - 1)[:k]

            cand_d = dists[cand_local]
            cand_i = cand_local + start

            merged_d = np.concatenate([best_d, cand_d])
            merged_i = np.concatenate([best_i, cand_i])
            keep = np.argpartition(merged_d, k - 1)[:k]
            best_d = merged_d[keep]
            best_i = merged_i[keep]

        order = np.argsort(best_d)
        return best_i[order], np.sqrt(best_d[order])

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


class FastLSHClassifier:
    def __init__(
        self,
        n_neighbors: int = 3,
        n_planes: int = 12,
        n_tables: int = 8,
        candidate_min: int = 256,
        candidate_max: int = 4096,
        seed: int = 42,
    ):
        self.n_neighbors = int(n_neighbors)
        self.n_planes = int(n_planes)
        self.n_tables = int(n_tables)
        self.candidate_min = int(candidate_min)
        self.candidate_max = int(candidate_max)
        self.seed = int(seed)
        self.planes = None
        self.tables = []
        self.X = None
        self.y = None
        self.rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.uint8)
        dim = self.X.shape[1]
        self.rng = np.random.default_rng(self.seed)
        self.planes = self.rng.standard_normal((self.n_tables, self.n_planes, dim)).astype(np.float32)
        self.tables = []

        for t in range(self.n_tables):
            projections = self.X @ self.planes[t].T
            bits = projections >= 0
            codes = self._bits_to_codes(bits)
            table = {}
            for idx, code in enumerate(codes):
                table.setdefault(int(code), []).append(idx)
            self.tables.append(table)
        return self

    def _bits_to_codes(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint64)
        shifts = np.arange(bits.shape[1], dtype=np.uint64)
        return np.sum(bits << shifts, axis=1, dtype=np.uint64)

    def _code_one(self, q: np.ndarray, table_idx: int) -> int:
        bits = (q @ self.planes[table_idx].T) >= 0
        return int(self._bits_to_codes(bits.reshape(1, -1))[0])

    def _candidate_indices(self, q: np.ndarray) -> np.ndarray:
        candidates = set()
        mask_values = [1 << i for i in range(self.n_planes)]

        for t, table in enumerate(self.tables):
            code = self._code_one(q, t)
            candidates.update(table.get(code, []))

        if len(candidates) < self.candidate_min:
            for t, table in enumerate(self.tables):
                code = self._code_one(q, t)
                for mask in mask_values:
                    candidates.update(table.get(code ^ mask, []))
                    if len(candidates) >= self.candidate_max:
                        break
                if len(candidates) >= self.candidate_max:
                    break

        if not candidates:
            size = min(self.candidate_max, self.X.shape[0])
            return self.rng.choice(self.X.shape[0], size=size, replace=False).astype(np.int64)

        arr = np.fromiter(candidates, dtype=np.int64)
        if len(arr) > self.candidate_max:
            arr = self.rng.choice(arr, size=self.candidate_max, replace=False).astype(np.int64)
        return arr

    def _nearest_one(self, q: np.ndarray):
        cands = self._candidate_indices(q)
        k = min(self.n_neighbors, len(cands))
        block = self.X[cands]
        diff = block - q
        sq_dists = np.einsum("ij,ij->i", diff, diff, optimize=True)
        if len(sq_dists) <= k:
            local = np.arange(len(sq_dists))
        else:
            local = np.argpartition(sq_dists, k - 1)[:k]
        order = np.argsort(sq_dists[local])
        nearest = cands[local[order]]
        distances = np.sqrt(sq_dists[local[order]])
        return nearest, distances

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
        model = FastLSHClassifier(n_neighbors=k).fit(X, y)
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
            model = FastLSHClassifier(n_neighbors=k_eff).fit(X_train, y_train)
        else:
            model = ExactVectorKNN(n_neighbors=k_eff).fit(X_train, y_train)

        start = time.perf_counter()
        preds = model.predict(X_val)
        total_latency = (time.perf_counter() - start) * 1000
        avg_latency = total_latency / max(1, len(y_val))
        acc = float(accuracy_score(y_val, preds))
        f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
        results.append({
            "k": int(k),
            "method": method,
            "accuracy": acc,
            "f1Score": f1,
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