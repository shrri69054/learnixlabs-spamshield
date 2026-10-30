"""REST API for live spam predictions.

Usage:  python train.py   (once)
        python app.py
"""
import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).parent / "models" / "spam_model.joblib"

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run `python train.py` first.")
        _model = joblib.load(MODEL_PATH)
    return _model


def classify(text: str) -> dict:
    model = get_model()
    pred = int(model.predict([text])[0])
    if hasattr(model, "predict_proba"):
        spam_prob = float(model.predict_proba([text])[0][1])
    else:
        spam_prob = float(pred)
    return {"label": "spam" if pred == 1 else "ham",
            "spam_probability": round(spam_prob, 4)}


@app.get("/")
def index():
    return jsonify({"service": "SpamShield", "endpoints": {
        "GET /health": "service status",
        "POST /predict": "JSON body {\"message\": \"...\"}",
        "POST /predict/batch": "JSON body {\"messages\": [\"...\", \"...\"]}"}})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists()})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Provide a non-empty string in 'message'."}), 400
    try:
        result = classify(message)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"message": message, **result})


@app.post("/predict/batch")
def predict_batch():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages")
    if (not isinstance(messages, list) or not messages
            or not all(isinstance(m, str) and m.strip() for m in messages)):
        return jsonify({"error": "Provide a non-empty list of strings in 'messages'."}), 400
    try:
        results = [{"message": m, **classify(m)} for m in messages]
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
