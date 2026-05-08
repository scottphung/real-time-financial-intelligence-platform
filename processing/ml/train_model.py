from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

from processing.ml.build_dataset import load_data, build_dataset


def train():

    # -------------------------
    # LOAD DATA
    # -------------------------
    df = load_data()

    X, y = build_dataset(df)

    # -------------------------
    # SAFETY CHECK
    # -------------------------
    if X is None or len(X) < 10:
        print("❌ Not enough data to train model")
        return

    # -------------------------
    # TRAIN / TEST SPLIT
    # -------------------------
    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # -------------------------
    # MODEL
    # -------------------------
    model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("📈 Model Accuracy:", acc)

    # -------------------------
    # SAVE MODEL
    # -------------------------
    joblib.dump(model, "processing/ml/model.pkl")

    print("💾 Model saved to processing/ml/model.pkl")


if __name__ == "__main__":
    train()