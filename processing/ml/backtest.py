import pandas as pd
import glob
from processing.ml.build_dataset import load_data
from processing.ml.signal_generator import generate_signal


def run_backtest():
    df = load_data()

    if df is None:
        return

    df = df.sort_values("timestamp")

    capital = 10_000
    position = 0
    entry_price = 0

    equity_curve = []

    for _, row in df.iterrows():

        features = {
            "price": row["price"],
            "return": row["return"],
            "moving_avg_10": row["moving_avg_10"],
            "volatility": row["volatility"],
            "is_anomaly": row["is_anomaly"]
        }

        signal = generate_signal(features)

        price = row["price"]

        # BUY
        if signal == "BUY 📈" and position == 0:
            position = capital / price
            entry_price = price
            capital = 0

        # SELL
        elif signal == "SELL 📉" and position > 0:
            capital = position * price
            position = 0

        # track portfolio value
        portfolio_value = capital + position * price
        equity_curve.append(portfolio_value)

    final_value = equity_curve[-1]

    print("\n📊 BACKTEST RESULTS")
    print("Start capital: 10,000")
    print("Final value:", round(final_value, 2))
    print("Return %:", round((final_value - 10000) / 10000 * 100, 2))


if __name__ == "__main__":
    run_backtest()