import os
from models.rf_nuestro import RandomForest 
import queue
import threading
import pickle
from flask import Flask, jsonify, request
import numpy as np
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from threading import Semaphore, Lock

MAX_CONCURRENT_REQUESTS = 10
MAX_QUEUE_SIZE = 50
MAX_WORKERS = 8

app = Flask(__name__)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["100 per minute"]
)

semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
task_queue = queue.Queue(MAX_QUEUE_SIZE)
model_lock = Lock()

with open("./models/modelo.pkl", "rb") as f:
    model = pickle.load(f)

def predict_threadsafe(features):
    with model_lock:
        return model.predict(features)

def get_model_params():
    if model is None:
        raise ValueError("The model is not valid")
    params = model.get_params()
    if not isinstance(params, dict):
        raise TypeError("get_params() didn't return a valid dictionary")
    return {
        "team": "Feliz Navidad",
        "model": type(model).__name__,
        "n_estimators": params.get("n_estimators", "unknown"),
        "max_depth": params.get("max_depth", "unknown"),
        "random_state": params.get("random_state", "unknown")
    }

def worker():
    while True:
        try:
            func, args, kwargs, result_queue = task_queue.get()
            with semaphore:
                try:
                    result = func(*args, **kwargs)
                    result_queue.put(result)
                except Exception as e:
                    result_queue.put(e)
                finally:
                    task_queue.task_done()
        except Exception:
            continue

for _ in range(MAX_WORKERS):
    t = threading.Thread(target=worker, daemon=True)
    t.start()

@app.route('/health')
@limiter.limit("50 per minute")
def health():
    return jsonify({
        "status": "ok",
    }), 200

@app.route('/predict', methods=['POST'])
@limiter.limit("50 per minute")
def predict():
    labels = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    try:
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({"error": "Invalid JSON, expected key 'features'"}), 400

        values = data["features"]
        if not isinstance(values, list) or not all(isinstance(x, (int, float)) for x in values):
            return jsonify({"error": "Expected a list of numeric values"}), 400

        X = np.array([values])
        if np.any(X < 0) or np.any(X > 10):
            return jsonify({"error": "Values out of range (0–10)"}), 422

        result_queue = queue.Queue()
        task_queue.put((predict_threadsafe, (X,), {}, result_queue))
        try:
            prediction = result_queue.get(timeout=10)
        except queue.Empty:
            return jsonify({"error": "Server busy, try again later"}), 503

        if isinstance(prediction, Exception):
            raise prediction

        label = labels[prediction[0]]
        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

@app.route('/info', methods=['GET'])
@limiter.limit("50 per minute")
def info():
    result_queue = queue.Queue()
    try:
        task_queue.put_nowait((get_model_params, (), {}, result_queue))
    except queue.Full:
        return jsonify({"status": "error", "message": "Server busy, please try again later"}), 503

    try:
        result = result_queue.get(timeout=10)
    except queue.Empty:
        return jsonify({"status": "error", "message": "Server busy, please try again later"}), 503

    if isinstance(result, Exception):
        return jsonify({"status": "error", "message": f"Internal error: {str(result)}"}), 500

    return jsonify(result), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)