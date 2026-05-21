import pandas as pd
import numpy as np
import json
import glob
from processing.ml.build_dataset import load_data
from processing.ml.signal_generator import generate_signal


BACKTEST_PATH = "processing/ml/backtest_results.json"


def run_backtest():

    # -------------------------
    # LOAD DATA
    # -------------------------
    df = load_data()

    if df is None or len(df) < 10:
        print("❌ Not enough data to backtest")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df[df["timestamp"].notna()]
    df = df.sort_values("timestamp").reset_index(drop=True)

    # fill nulls
    df["return"] = df["return"].fillna(0)
    df["moving_avg_10"] = df["moving_avg_10"].fillna(df["price"])
    df["volatility"] = df["volatility"].fillna(0)
    df["is_anomaly"] = df["is_anomaly"].fillna(0).astype(int)

    # -------------------------
    # SIMULATION
    # -------------------------
    capital     = 10_000.0
    position    = 0.0
    entry_price = 0.0

    equity_curve = []
    trades       = []

    for _, row in df.iterrows():
        features = {
            "price":         row["price"],
            "return":        row["return"],
            "moving_avg_10": row["moving_avg_10"],
            "volatility":    row["volatility"],
            "is_anomaly":    row["is_anomaly"]
        }

        result = generate_signal(features)
        action = result.get("action", "HOLD")
        price  = float(row["price"])

        # BUY
        if action == "BUY" and position == 0 and capital > 0:
            position    = capital / price
            entry_price = price
            capital     = 0.0
            trades.append({"type": "BUY", "price": price})

        # SELL
        elif action == "SELL" and position > 0:
            capital  = position * price
            pnl      = (price - entry_price) / entry_price * 100
            position = 0.0
            trades.append({"type": "SELL", "price": price, "pnl_pct": round(pnl, 4)})

        portfolio_value = capital + position * price
        equity_curve.append({
            "timestamp": str(row["timestamp"]),
            "value":     round(portfolio_value, 2)
        })

    # -------------------------
    # QUANT METRICS
    # -------------------------
    values = [e["value"] for e in equity_curve]
    final_value = values[-1]
    total_return = (final_value - 10_000) / 10_000 * 100

    # daily returns from equity curve
    equity_series = pd.Series(values)
    pct_returns   = equity_series.pct_change().dropna()

    # Sharpe ratio (annualized, assuming ~252 trading periods)
    if pct_returns.std() > 0:
        sharpe = (pct_returns.mean() / pct_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # max drawdown
    rolling_max  = equity_series.cummax()
    drawdown     = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100  # as percentage

    # win rate from completed trades
    sell_trades = [t for t in trades if t["type"] == "SELL"]
    if sell_trades:
        wins     = [t for t in sell_trades if t["pnl_pct"] > 0]
        win_rate = len(wins) / len(sell_trades) * 100
    else:
        win_rate = 0.0

    # -------------------------
    # PRINT RESULTS
    # -------------------------
    print("\n📊 BACKTEST RESULTS")
    print(f"  Start Capital:  $10,000.00")
    print(f"  Final Value:    ${final_value:,.2f}")
    print(f"  Total Return:   {total_return:.2f}%")
    print(f"  Sharpe Ratio:   {sharpe:.4f}")
    print(f"  Max Drawdown:   {max_drawdown:.2f}%")
    print(f"  Total Trades:   {len(sell_trades)}")
    print(f"  Win Rate:       {win_rate:.1f}%")

    # -------------------------
    # SAVE RESULTS
    # -------------------------
    results = {
        "start_capital":  10000.0,
        "final_value":    round(final_value, 2),
        "total_return":   round(total_return, 4),
        "sharpe_ratio":   round(float(sharpe), 4),
        "max_drawdown":   round(float(max_drawdown), 4),
        "total_trades":   len(sell_trades),
        "win_rate":       round(win_rate, 2),
        "equity_curve":   equity_curve[-200:]  # last 200 points for dashboard
    }

    with open(BACKTEST_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {BACKTEST_PATH}")


if __name__ == "__main__":
    run_backtest()