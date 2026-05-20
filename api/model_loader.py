import os
import joblib

MODEL_PATH = "processing/ml/model.pkl"

model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    else:
        print("⚠️  model.pkl not found — API running in NO-PREDICTION mode")

def get_model():
    return model