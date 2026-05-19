import argparse
from pathlib import Path

import numpy as np

from machine_learning import KNearestNeighbors
from utility import preprocess_mnist


def _find_npz_array(data, candidates):
    for key in candidates:
        if key in data:
            return data[key], key
    return None, None


def load_mnist_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"MNIST NPZ not found: {path}")

    data = np.load(path, allow_pickle=False)
    X, x_key = _find_npz_array(data, ["X", "x", "images", "pixels", "arr_0"])
    y, y_key = _find_npz_array(data, ["y", "Y", "labels", "label", "target", "arr_1"])

    if X is None or y is None:
        raise ValueError(
            f"Could not find X/y in {path}. Keys present: {list(data.keys())}"
        )

    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if y.dtype.kind in ("U", "S", "O"):
        y = y.astype(np.int64)

    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")

    return X, y


def split_dataset(X, y, train_size, test_size, seed):
    total = train_size + test_size
    if total > len(y):
        raise ValueError(f"Requested {total} samples, but dataset has {len(y)}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[:total]
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def classification_metrics(y_true, y_pred, num_classes=10):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    accuracy = float(np.mean(y_true == y_pred))
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)

    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
    }


def run_eval(X_train, X_test, y_train, y_test, args, index_name):
    model = KNearestNeighbors(
        num_neighbors=args.k,
        p=args.p,
        index=index_name,
        leaf_size=args.leaf_size,
        lsh_n_planes=args.lsh_planes,
        lsh_n_tables=args.lsh_tables,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return classification_metrics(y_test, preds, num_classes=10)


def format_metrics(name, metrics):
    return (
        f"{name:7} | "
        f"acc={metrics['accuracy']:.4f} | "
        f"prec={metrics['precision']:.4f} | "
        f"rec={metrics['recall']:.4f} | "
        f"f1={metrics['f1']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate KNN on MNIST using kdtree and lsh indices.")
    parser.add_argument("--data", default="data\\import\\mnist.npz", help="Path to MNIST .npz file")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--leaf-size", type=int, default=30)
    parser.add_argument("--lsh-planes", type=int, default=10)
    parser.add_argument("--lsh-tables", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    X, y = load_mnist_npz(data_path)
    X_train, X_test, y_train, y_test = split_dataset(X, y, args.train_size, args.test_size, args.seed)
    X_train, X_test, y_train, y_test = preprocess_mnist(X_train, X_test, y_train, y_test, flatten=True, normalize=True)

    print(f"Dataset: {data_path} | train={len(y_train)} test={len(y_test)} | k={args.k}")
    print("Index   | Accuracy | Precision | Recall | F1")

    kdtree_metrics = run_eval(X_train, X_test, y_train, y_test, args, "kdtree")
    lsh_metrics = run_eval(X_train, X_test, y_train, y_test, args, "lsh")

    print(format_metrics("kdtree", kdtree_metrics))
    print(format_metrics("lsh", lsh_metrics))


if __name__ == "__main__":
    main()
