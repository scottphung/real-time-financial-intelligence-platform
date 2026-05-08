import pandas as pd
import glob


# -------------------------
# LOAD DATA FROM DATA LAKE
# -------------------------
def load_data():
    files = glob.glob("data_lake/crypto/**/*.parquet", recursive=True)

    if not files:
        print("❌ No data found in data_lake")
        return None

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    return df


# -------------------------
# FEATURE + LABEL BUILDING
# -------------------------
def build_dataset(df):

    if df is None or len(df) == 0:
        print("❌ Empty dataset")
        return None, None

    # -------------------------
    # FIX: timestamp normalization
    # -------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()]

    # sort for time series correctness
    df = df.sort_values("timestamp")

    # -------------------------
    # BASIC FEATURE ENGINEERING
    # -------------------------
    df["future_price"] = df["price"].shift(-1)
    df["return"] = df["price"].pct_change()

    df["moving_avg_3"] = df["price"].rolling(window=3, min_periods=1).mean()

    df["volatility"] = df["price"].rolling(window=3, min_periods=1).std().fillna(0)

    df["is_anomaly"] = (abs(df["return"]) > 0.002).astype(int)

    # -------------------------
    # TARGET LABEL
    # -------------------------
    df["target"] = (df["future_price"] > df["price"]).astype(int)

    # drop last row (no future price)
    df = df.dropna()

    # -------------------------
    # SAFETY CHECK
    # -------------------------
    if len(df) < 5:
        print(f"⚠️ Not enough data after processing: {len(df)} rows")
        return None, None

    features = [
        "price",
        "return",
        "moving_avg_3",
        "volatility",
        "is_anomaly"
    ]

    X = df[features]
    y = df["target"]

    return X, y


# -------------------------
# DEBUG MODE
# -------------------------
if __name__ == "__main__":
    df = load_data()

    X, y = build_dataset(df)

    if X is not None:
        print("📊 Dataset ready")
        print("Features shape:", X.shape)
        print("Target distribution:")
        print(y.value_counts())