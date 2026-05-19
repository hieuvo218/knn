import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from db import current_dataset_version, fetch_accepted_samples

CACHE_DIR = Path(os.getenv("MNIST_CACHE_DIR", "/app/data/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_PATH = CACHE_DIR / "active_cache.json"
IMPORT_NPZ = Path(os.getenv("MNIST_IMPORT_NPZ", "/app/data/import/mnist.npz"))
IMPORT_CSV = Path(os.getenv("MNIST_IMPORT_CSV", "/app/data/import/mnist.csv"))
IMPORT_SCRIPT = Path(__file__).resolve().parent / "scripts" / "import_local_mnist.py"

_lock = threading.Lock()
_loaded = {
    "version": None,
    "ids": None,
    "X": None,
    "y": None,
}


def _cache_file(version: int) -> Path:
    return CACHE_DIR / f"dataset_v{version}.npz"


def _read_active_version():
    if not ACTIVE_PATH.exists():
        return None
    try:
        data = json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))
        return int(data["datasetVersion"]), Path(data["path"])
    except Exception:
        return None


def _write_active(version: int, path: Path, sample_count: int):
    ACTIVE_PATH.write_text(json.dumps({
        "datasetVersion": version,
        "path": str(path),
        "sampleCount": sample_count,
    }, indent=2), encoding="utf-8")


def _resolve_import_paths() -> tuple[Path, Path]:
    if IMPORT_NPZ.exists() or IMPORT_CSV.exists():
        return IMPORT_NPZ, IMPORT_CSV

    repo_root = Path(__file__).resolve().parents[2]
    fallback_dir = repo_root / "data" / "import"
    fallback_npz = fallback_dir / "mnist.npz"
    fallback_csv = fallback_dir / "mnist.csv"
    return fallback_npz, fallback_csv


def _seed_local_mnist() -> bool:
    if not IMPORT_SCRIPT.exists():
        return False
    npz_path, csv_path = _resolve_import_paths()
    if not npz_path.exists() and not csv_path.exists():
        return False

    command = [
        sys.executable,
        str(IMPORT_SCRIPT),
        "--npz",
        str(npz_path),
        "--csv",
        str(csv_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        message = details if details else "Unknown error"
        raise RuntimeError(f"Local MNIST import failed: {message}") from exc
    return True


def rebuild_cache(version: int) -> Path:
    rows = fetch_accepted_samples()
    if not rows:
        seeded = _seed_local_mnist()
        if seeded:
            version = current_dataset_version()
            rows = fetch_accepted_samples()
    if not rows:
        raise RuntimeError(
            "No accepted samples found in database. "
            "Add MNIST data to /app/data/import or run the MNIST importer."
        )

    ids = np.array([int(row["id"]) for row in rows], dtype=np.int64)
    X = np.array([row["pixels"] for row in rows], dtype=np.uint8)
    y = np.array([int(row["label"]) for row in rows], dtype=np.uint8)

    if X.ndim != 2 or X.shape[1] != 784:
        raise RuntimeError(f"Expected X shape (N, 784), got {X.shape}")

    final_path = _cache_file(version)
    tmp_path = final_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, ids=ids, X=X, y=y, dataset_version=version)

    # Verify before atomic swap.
    check = np.load(tmp_path)
    assert int(check["dataset_version"]) == version
    assert check["X"].shape[1] == 784

    tmp_path.replace(final_path)
    _write_active(version, final_path, len(y))
    return final_path


def ensure_cache() -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    with _lock:
        db_version = current_dataset_version()
        if _loaded["version"] == db_version and _loaded["X"] is not None:
            return _loaded["version"], _loaded["ids"], _loaded["X"], _loaded["y"]

        active = _read_active_version()
        path = None
        if active is not None:
            active_version, active_path = active
            if active_version == db_version and active_path.exists():
                path = active_path

        if path is None:
            path = rebuild_cache(db_version)

        data = np.load(path)
        _loaded["version"] = int(data["dataset_version"])
        _loaded["ids"] = data["ids"]
        _loaded["X"] = data["X"]
        _loaded["y"] = data["y"]

        return _loaded["version"], _loaded["ids"], _loaded["X"], _loaded["y"]


def cache_status() -> Dict:
    db_version = current_dataset_version()
    active = _read_active_version()
    active_version = active[0] if active else None
    active_path = str(active[1]) if active else None
    return {
        "dbVersion": db_version,
        "cacheVersion": active_version,
        "cachePath": active_path,
        "fresh": active_version == db_version,
        "loadedVersion": _loaded["version"],
        "loadedSamples": None if _loaded["y"] is None else int(len(_loaded["y"])),
    }
