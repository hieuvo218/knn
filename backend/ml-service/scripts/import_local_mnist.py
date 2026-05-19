import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import sys
import time
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2.extras

from db import get_conn


def normalize_to_uint8_flat(X):
    X = np.asarray(X)

    if X.ndim == 4 and X.shape[-1] == 1:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    if X.ndim != 2:
        raise ValueError(f"Expected X with shape (N, 784) or (N, 28, 28), got {X.shape}")

    if X.shape[1] != 784:
        raise ValueError(f"Expected 784 pixels per sample, got shape {X.shape}")

    if np.issubdtype(X.dtype, np.floating):
        finite = np.isfinite(X)
        if not finite.all():
            raise ValueError("X contains NaN or infinite values")

        max_value = float(X.max())
        min_value = float(X.min())

        if min_value >= 0.0 and max_value <= 1.0:
            X = X * 255.0

        X = np.rint(X)
    else:
        X = X.astype(np.int16, copy=False)

    X = np.clip(X, 0, 255).astype(np.uint8)
    return X


def normalize_labels(y):
    y = np.asarray(y).reshape(-1)
    y = y.astype(np.int16)

    if y.min() < 0 or y.max() > 9:
        raise ValueError(f"Labels must be in [0, 9], got min={y.min()}, max={y.max()}")

    return y.astype(np.uint8)


def find_npz_array(data, candidates):
    for key in candidates:
        if key in data:
            return data[key], key
    return None, None


def load_from_npz(npz_path):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)

    X, x_key = find_npz_array(data, ["X", "x", "images", "pixels", "arr_0"])
    y, y_key = find_npz_array(data, ["y", "Y", "labels", "label", "target", "arr_1"])

    if X is None or y is None:
        raise ValueError(
            f"Could not find X/y in NPZ. Found keys: {list(data.keys())}. "
            "Expected keys like X/y, images/labels, pixels/label, or arr_0/arr_1."
        )

    X = normalize_to_uint8_flat(X)
    y = normalize_labels(y)

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: len(X)={len(X)}, len(y)={len(y)}")

    print(f"Loaded NPZ from {npz_path}")
    print(f"X key: {x_key}, y key: {y_key}")
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")

    return X, y


def parse_array_cell(value):
    if isinstance(value, list):
        arr = value
    else:
        text = str(value).strip()

        if text.startswith("{") and text.endswith("}"):
            # PostgreSQL array literal: {0,0,255,...}
            inner = text[1:-1].strip()
            if not inner:
                arr = []
            else:
                arr = [int(x) for x in inner.split(",")]
        elif text.startswith("[") and text.endswith("]"):
            # JSON list or Python list
            try:
                arr = json.loads(text)
            except Exception:
                arr = literal_eval(text)
        else:
            # fallback: comma-separated or whitespace-separated values
            if "," in text:
                arr = [int(float(x)) for x in text.split(",") if x.strip()]
            else:
                arr = [int(float(x)) for x in text.split() if x.strip()]

    arr = np.asarray(arr)
    if arr.shape[0] != 784:
        raise ValueError(f"Expected 784 pixels, got {arr.shape[0]}")
    return arr


def load_from_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    label_col = None
    for candidate in ["label", "y", "target", "true_label"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        raise ValueError(f"Could not find label column. Found columns: {list(df.columns)[:20]}")

    array_col = None
    for candidate in ["image_array", "pixels", "array", "image", "features"]:
        if candidate in df.columns:
            array_col = candidate
            break

    if array_col is not None:
        X_raw = np.stack(df[array_col].map(parse_array_cell).to_numpy())
    else:
        ignored = {"id", "label", "y", "target", "true_label"}
        pixel_cols = [col for col in df.columns if col not in ignored]

        if len(pixel_cols) < 784:
            raise ValueError(
                "CSV must contain either an image_array/pixels column or 784 pixel columns. "
                f"Found only {len(pixel_cols)} pixel-like columns."
            )

        pixel_cols = pixel_cols[:784]
        X_raw = df[pixel_cols].to_numpy()

    y_raw = df[label_col].to_numpy()

    X = normalize_to_uint8_flat(X_raw)
    y = normalize_labels(y_raw)

    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: len(X)={len(X)}, len(y)={len(y)}")

    print(f"Loaded CSV from {csv_path}")
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")

    return X, y


def load_dataset(npz_path, csv_path):
    npz_path = Path(npz_path) if npz_path else None
    csv_path = Path(csv_path) if csv_path else None

    if npz_path and npz_path.exists():
        return load_from_npz(npz_path), "mnist_npz"

    if csv_path and csv_path.exists():
        return load_from_csv(csv_path), "mnist_csv"

    raise FileNotFoundError(
        "No local MNIST file found. Put your files at:\n"
        "  data/import/mnist.npz\n"
        "  data/import/mnist.csv\n"
        "or pass --npz /path/to/file.npz / --csv /path/to/file.csv"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="/app/data/import/mnist.npz")
    parser.add_argument("--csv", default="/app/data/import/mnist.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    (X, y), source = load_dataset(args.npz, args.csv)

    if args.limit and args.limit > 0:
        X = X[: args.limit]
        y = y[: args.limit]

    print(f"Final dataset: {len(y)} samples from {source}")
    print("Label distribution:")
    labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(labels, counts):
        print(f"  {int(label)}: {int(count)}")

    rows = [
        (X[i].astype(int).tolist(), int(y[i]), source, "accepted")
        for i in range(len(y))
    ]

    start = time.time()

    with get_conn() as conn:
        with conn.cursor() as cur:
            if args.force:
                print("Force mode: clearing old app data")
                cur.execute("DELETE FROM tuning_results")
                cur.execute("DELETE FROM tuning_jobs")
                cur.execute("DELETE FROM feedback_samples")
                cur.execute("DELETE FROM predictions")
                cur.execute("DELETE FROM digit_samples")

            cur.execute(
                """
                SELECT COUNT(*)
                FROM digit_samples
                WHERE source IN ('mnist_npz', 'mnist_csv')
                """
            )
            existing = cur.fetchone()[0]

            if existing > 0 and not args.force:
                print(f"Local MNIST dataset already imported: {existing} rows.")
                print("Use --force if you really want to clear and re-import.")
                return

            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO digit_samples(pixels, label, source, status, accepted_at)
                VALUES %s
                """,
                rows,
                template="(%s, %s, %s, %s, NOW())",
                page_size=1000,
            )

            cur.execute(
                """
                UPDATE dataset_state
                SET version = version + 1,
                    updated_at = NOW()
                WHERE id = 1
                """
            )

    elapsed = time.time() - start
    print(f"Imported {len(rows)} rows into PostgreSQL in {elapsed:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"IMPORT FAILED: {exc}", file=sys.stderr)
        raise
