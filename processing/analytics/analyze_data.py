import duckdb
import pandas as pd
import glob


def load_data():
    files = glob.glob("data_lake/crypto/**/*.parquet", recursive=True)

    if not files:
        print("No data found")
        return None

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df


def basic_insights(df):
    print("\n📊 MARKET SUMMARY")

    print("Total records:", len(df))

    print("Avg price:", df["price"].mean())

    print("Max price:", df["price"].max())

    print("Volatility avg:", df["volatility"].mean())

    print("\n🚨 Anomalies detected:", df["is_anomaly"].sum())


def trend_analysis(df):
    print("\n📈 TREND ANALYSIS")

    # 🔥 FORCE CONSISTENT TYPE
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df = df.sort_values("timestamp")

    print("Latest price:", df["price"].iloc[-1])
    print("Price change:", df["price"].iloc[-1] - df["price"].iloc[0])


if __name__ == "__main__":
    df = load_data()

    if df is not None:
        basic_insights(df)
        trend_analysis(df)