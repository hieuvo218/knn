import json
import os
import threading
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from db import current_dataset_version, fetch_accepted_samples

CACHE_DIR = Path(os.getenv("MNIST_CACHE_DIR", "/app/data/cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_PATH = CACHE_DIR / "active_cache.json"

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


def rebuild_cache(version: int) -> Path:
    rows = fetch_accepted_samples()
    if not rows:
        raise RuntimeError("No accepted samples found in database. Run the MNIST importer first.")

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
