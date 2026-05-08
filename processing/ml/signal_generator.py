import os
import joblib

MODEL_PATH = "processing/ml/model.pkl"

model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded")
else:
    print("⚠️ Model not found — running in NO-PREDICTION mode")


def generate_signal(features):
    if model is None:
        return {
            "signal": "NO_MODEL",
            "action": "HOLD"
        }

    pred = model.predict([[
        features["price"],
        features["return"],
        features["moving_avg_10"],
        features["volatility"],
        features["is_anomaly"]
    ]])[0]

    return {
        "signal": int(pred),
        "action": "BUY" if pred == 1 else "SELL"
    }