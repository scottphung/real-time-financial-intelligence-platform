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
    # TIMESTAMP NORMALIZATION
    # -------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df[df["timestamp"].notna()]
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -------------------------
    # KEEP ONLY COLUMNS WE NEED
    # -------------------------
    df = df[["timestamp", "price", "return", "moving_avg_10", "volatility", "is_anomaly"]].copy()

    # -------------------------
    # FILL NULLS BEFORE FEATURE ENGINEERING
    # -------------------------
    df["return"] = df["return"].fillna(0)
    df["moving_avg_10"] = df["moving_avg_10"].fillna(df["price"])
    df["volatility"] = df["volatility"].fillna(0)
    df["is_anomaly"] = df["is_anomaly"].fillna(0).astype(int)

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------
    df["future_price"] = df["price"].shift(-5)
    df["volatility"] = df["price"].rolling(window=3, min_periods=1).std().fillna(0)

    # -------------------------
    # TARGET LABEL
    # -------------------------
    df["target"] = (df["future_price"] > df["price"]).astype(int)

    # Drop only the last row (no future price) — not all rows
    df = df.iloc[:-5]

    # -------------------------
    # SAFETY CHECK
    # -------------------------
    if len(df) < 10:
        print(f"⚠️ Not enough data after processing: {len(df)} rows")
        return None, None

    features = [
        "price",
        "return",
        "moving_avg_10",
        "volatility",
        "is_anomaly"
    ]

    X = df[features]
    y = df["target"]

    print(f"✅ Dataset built: {len(df)} rows, {len(features)} features")
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