import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from cache import cache_status, ensure_cache, rebuild_cache
from db import current_dataset_version
from knn_core import predict_digit, tune_knn

app = Flask(__name__)
CORS(app)


def error_response(exc, status=400):
    return jsonify({"error": exc.__class__.__name__, "message": str(exc)}), status


@app.get("/health")
def health():
    try:
        status = cache_status()
        return jsonify({"status": "ok", "cache": status})
    except Exception as exc:
        return error_response(exc, 500)


@app.post("/cache/rebuild")
def rebuild():
    try:
        version = current_dataset_version()
        path = rebuild_cache(version)
        return jsonify({"status": "rebuilt", "datasetVersion": version, "path": str(path)})
    except Exception as exc:
        return error_response(exc, 500)


@app.post("/predict")
def predict_post():
    try:
        body = request.get_json(force=True)
        pixels = body.get("pixels")
        k = int(body.get("k", 3))
        method = body.get("method", "kd_tree")
        if method not in ("kd_tree", "lsh"):
            raise ValueError("method must be kd_tree or lsh")

        version, ids, X, y = ensure_cache()
        pred, confidence, latency_ms = predict_digit(version, X, y, pixels, k, method)
        return jsonify({
            "predictedLabel": pred,
            "confidence": confidence,
            "latencyMs": latency_ms,
            "datasetVersion": int(version),
            "sampleCount": int(len(y)),
        })
    except Exception as exc:
        return error_response(exc, 400)


@app.get("/predict")
def predict_get():
    try:
        pixels = [int(x) for x in request.args.get("pixels", "").split(",") if x.strip()]
        k = int(request.args.get("k", 3))
        method = request.args.get("method", "kd_tree")
        version, ids, X, y = ensure_cache()
        pred, confidence, latency_ms = predict_digit(version, X, y, pixels, k, method)
        return jsonify({
            "predictedLabel": pred,
            "confidence": confidence,
            "latencyMs": latency_ms,
            "datasetVersion": int(version),
            "sampleCount": int(len(y)),
        })
    except Exception as exc:
        return error_response(exc, 400)


@app.post("/tune")
def tune_post():
    try:
        body = request.get_json(force=True)
        sample_count = int(body.get("sampleCount") or body.get("sample_count") or 500)
        method = body.get("method", "kd_tree")
        if method not in ("kd_tree", "lsh"):
            raise ValueError("method must be kd_tree or lsh")
        k_values = body.get("kValues") or body.get("k_values") or [1, 3, 5, 7]
        version, ids, X, y = ensure_cache()
        return jsonify(tune_knn(version, X, y, sample_count, method, k_values))
    except Exception as exc:
        return error_response(exc, 400)


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
