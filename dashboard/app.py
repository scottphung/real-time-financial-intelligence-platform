import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
import random
import os

# -------------------------
# CONFIG
# -------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
REFRESH_INTERVAL = 10  # seconds

st.set_page_config(
    page_title="Financial Intelligence Platform",
    page_icon="📈",
    layout="wide"
)

# -------------------------
# INIT SESSION STATE
# -------------------------
if "price_history" not in st.session_state:
    st.session_state.price_history = []

if "signal_history" not in st.session_state:
    st.session_state.signal_history = []

if "last_price" not in st.session_state:
    st.session_state.last_price = None


# -------------------------
# FETCH LIVE PRICE
# -------------------------
def fetch_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin", "vs_currencies": "usd"}
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        return float(data["bitcoin"]["usd"])
    except Exception:
        # fallback: simulate small price movement if rate limited
        if st.session_state.last_price:
            return st.session_state.last_price * (1 + random.uniform(-0.0005, 0.0005))
        return 76000.0


# -------------------------
# COMPUTE FEATURES
# -------------------------
def compute_features(price):
    history = st.session_state.price_history

    if len(history) == 0:
        return_pct = 0.0
        moving_avg = price
        volatility = 0.0
    else:
        prev_price = history[-1]["price"]
        return_pct = (price - prev_price) / prev_price if prev_price != 0 else 0.0
        prices = [h["price"] for h in history[-10:]] + [price]
        moving_avg = sum(prices) / len(prices)
        if len(prices) >= 2:
            mean = moving_avg
            volatility = (sum((p - mean) ** 2 for p in prices) / len(prices)) ** 0.5
        else:
            volatility = 0.0

    is_anomaly = 1 if abs(return_pct) > 0.002 else 0

    return {
        "coin": "bitcoin",
        "price": price,
        "return_pct": return_pct,
        "moving_avg_3": moving_avg,
        "volatility": volatility,
        "is_anomaly": is_anomaly
    }


# -------------------------
# CALL FASTAPI
# -------------------------
def get_signal(features):
    try:
        r = requests.post(API_URL, json=features, timeout=5)
        return r.json()
    except Exception as e:
        return {"action": "API_DOWN", "confidence": 0.0, "signal": -1}


# -------------------------
# HEADER
# -------------------------
st.title("📈 Real-Time Financial Intelligence Platform")
st.caption("Live BTC/USD · ML Signal Generation · Powered by FastAPI + Kafka")

st.divider()

# -------------------------
# MAIN LAYOUT
# -------------------------
col1, col2, col3, col4 = st.columns(4)

price_box = col1.empty()
signal_box = col2.empty()
confidence_box = col3.empty()
time_box = col4.empty()

st.divider()

chart_title = st.empty()
chart_box = st.empty()

st.divider()

table_title = st.empty()
table_box = st.empty()

# -------------------------
# LIVE LOOP
# -------------------------
while True:

    # --- fetch ---
    price = fetch_price()
    st.session_state.last_price = price
    features = compute_features(price)
    result = get_signal(features)

    action = result.get("action", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    now = datetime.now().strftime("%H:%M:%S")

    # --- update history ---
    st.session_state.price_history.append({
        "time": now,
        "price": price
    })

    st.session_state.signal_history.append({
        "time": now,
        "price": f"${price:,.0f}",
        "signal": action,
        "confidence": f"{confidence * 100:.1f}%",
        "return": f"{features['return_pct'] * 100:.4f}%",
        "volatility": f"{features['volatility']:.2f}"
    })

    # keep last 50 ticks only
    st.session_state.price_history = st.session_state.price_history[-50:]
    st.session_state.signal_history = st.session_state.signal_history[-50:]

    # --- metric cards ---
    price_box.metric(
        label="BTC/USD Price",
        value=f"${price:,.0f}",
        delta=f"{features['return_pct'] * 100:.4f}%"
    )

    signal_emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
    signal_box.metric(
        label="ML Signal",
        value=f"{signal_emoji} {action}"
    )

    confidence_box.metric(
        label="Model Confidence",
        value=f"{confidence * 100:.1f}%"
    )

    time_box.metric(
        label="Last Updated",
        value=now
    )

    # --- price chart ---
    chart_title.markdown("### 📊 Live Price Chart (Last 50 Ticks)")
    if len(st.session_state.price_history) > 1:
        chart_df = pd.DataFrame(st.session_state.price_history)
        chart_df = chart_df.set_index("time")
        chart_box.line_chart(chart_df["price"])

    # --- signal table ---
    table_title.markdown("### 🧠 Signal History")
    if st.session_state.signal_history:
        table_df = pd.DataFrame(st.session_state.signal_history[::-1])
        table_box.dataframe(table_df, use_container_width=True, hide_index=True)

    time.sleep(REFRESH_INTERVAL)