import argparse
import sys
import time

import numpy as np
import psycopg2.extras
from sklearn.datasets import fetch_openml, load_digits
from sklearn.preprocessing import MinMaxScaler

from db import get_conn


def load_mnist(limit: int):
    try:
        print("Downloading/loading MNIST from OpenML...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.uint8)
        y = mnist.target.astype(np.uint8)
        source = "mnist"
    except Exception as exc:
        print(f"OpenML failed: {exc}", file=sys.stderr)
        print("Falling back to sklearn load_digits resized/padded to 28x28 for local dev.")
        digits = load_digits()
        X8 = digits.images.astype(np.float32)
        y = digits.target.astype(np.uint8)
        X8 = MinMaxScaler(feature_range=(0, 255)).fit_transform(X8.reshape(len(X8), -1)).reshape(len(X8), 8, 8)
        X = np.zeros((len(X8), 28, 28), dtype=np.uint8)
        X[:, 10:18, 10:18] = X8.astype(np.uint8)
        X = X.reshape(len(X), 784)
        source = "sklearn_digits_fallback"

    if limit and limit > 0:
        X = X[:limit]
        y = y[:limit]
    return X, y, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=70000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    X, y, source = load_mnist(args.limit)
    print(f"Loaded {len(y)} samples from {source}")

    rows = [(X[i].astype(int).tolist(), int(y[i]), source, "accepted") for i in range(len(y))]
    start = time.time()
    with get_conn() as conn:
        with conn.cursor() as cur:
            if args.force:
                print("Force mode: clearing digit_samples and feedback/predictions/tuning data")
                cur.execute("DELETE FROM tuning_results")
                cur.execute("DELETE FROM tuning_jobs")
                cur.execute("DELETE FROM predictions")
                cur.execute("DELETE FROM feedback_samples")
                cur.execute("DELETE FROM digit_samples")

            cur.execute("SELECT COUNT(*) FROM digit_samples WHERE source IN ('mnist', 'sklearn_digits_fallback')")
            existing = cur.fetchone()[0]
            if existing > 0 and not args.force:
                print(f"Base dataset already exists: {existing} rows. Use --force to reload.")
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
            cur.execute("UPDATE dataset_state SET version = version + 1, updated_at = NOW() WHERE id = 1")

    elapsed = time.time() - start
    print(f"Imported {len(rows)} rows in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
