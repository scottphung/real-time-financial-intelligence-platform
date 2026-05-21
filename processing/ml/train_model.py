import joblib
import json
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from processing.ml.build_dataset import load_data, build_dataset


# -------------------------
# FEATURE COLUMNS
# -------------------------
FEATURES = [
    "price",
    "return",
    "moving_avg_10",
    "volatility",
    "is_anomaly"
]

MODEL_PATH = "processing/ml/model.pkl"
METRICS_PATH = "processing/ml/model_metrics.json"


def train():

    # -------------------------
    # LOAD DATA
    # -------------------------
    df = load_data()
    X, y = build_dataset(df)

    if X is None or len(X) < 10:
        print("❌ Not enough data to train model")
        return

    # -------------------------
    # TRAIN / TEST SPLIT
    # (time-based, not random — preserves temporal order)
    # -------------------------
    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    print(f"📊 Training on {len(X_train)} rows, testing on {len(X_test)} rows")

    # -------------------------
    # XGBOOST MODEL
    # -------------------------
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        scale_pos_weight=6
    )

    model.fit(X_train, y_train)

    # -------------------------
    # EVALUATION
    # -------------------------
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, zero_division=0)
    recall    = recall_score(y_test, preds, zero_division=0)
    f1        = f1_score(y_test, preds, zero_division=0)
    cm        = confusion_matrix(y_test, preds).tolist()

    # feature importance
    importance = dict(zip(FEATURES, model.feature_importances_.tolist()))

    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall:    {recall:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"📊 Confusion Matrix: {cm}")
    print(f"🔍 Feature Importance: {importance}")

    # -------------------------
    # SAVE MODEL
    # -------------------------
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

    # -------------------------
    # SAVE METRICS (for dashboard)
    # -------------------------
    metrics = {
        "accuracy":          round(acc, 4),
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1_score":          round(f1, 4),
        "confusion_matrix":  cm,
        "feature_importance": importance,
        "train_rows":        len(X_train),
        "test_rows":         len(X_test),
        "model_type":        "XGBClassifier"
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"📁 Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    train()