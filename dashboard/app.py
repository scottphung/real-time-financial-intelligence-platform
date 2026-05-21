import streamlit as st
import requests
import pandas as pd
import time
import json
import os
from datetime import datetime
import random
import altair as alt

# -------------------------
# CONFIG
# -------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
METRICS_PATH = "processing/ml/model_metrics.json"
REFRESH_INTERVAL = 10

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
    except Exception:
        return {"action": "API_DOWN", "confidence": 0.0, "signal": -1}


# -------------------------
# LOAD MODEL METRICS
# -------------------------
def load_metrics():
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


# -------------------------
# TABS
# -------------------------
tab1, tab2 = st.tabs(["📡 Live Signals", "🧠 Model Metrics"])


# ========================
# TAB 1 — LIVE SIGNALS
# ========================
with tab1:
    st.title("📈 Real-Time Financial Intelligence Platform")
    st.caption("Live BTC/USD · ML Signal Generation · Powered by FastAPI + Kafka")
    st.divider()

    # fetch and update on every rerun
    price = fetch_price()
    st.session_state.last_price = price
    features = compute_features(price)
    result = get_signal(features)

    action     = result.get("action", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    now        = datetime.now().strftime("%H:%M:%S")

    st.session_state.price_history.append({"time": now, "price": price})
    st.session_state.signal_history.append({
        "Time":   now,
        "Price":  f"${price:,.0f}",
        "Signal": action,
        "Conf%":  f"{confidence * 100:.1f}%",
        "Ret%":   f"{features['return_pct'] * 100:.4f}%",
        "Vol":    f"{features['volatility']:.2f}"
    })

    st.session_state.price_history  = st.session_state.price_history[-50:]
    st.session_state.signal_history = st.session_state.signal_history[-50:]

    signal_emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BTC/USD Price", f"${price:,.0f}", f"{features['return_pct']*100:.4f}%")
    col2.metric("ML Signal", f"{signal_emoji} {action}")
    col3.metric("Model Confidence", f"{confidence*100:.1f}%")
    col4.metric("Last Updated", now)

    st.divider()
    st.markdown("### 📊 Live Price Chart (Last 50 Ticks)")
    if len(st.session_state.price_history) > 1:
        chart_df = pd.DataFrame(st.session_state.price_history)
        chart_df["tick"] = range(len(chart_df))
        chart = alt.Chart(chart_df).mark_line(color="#4c9be8").encode(
            x=alt.X("tick:Q", title="Tick", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("price:Q", title="BTC Price (USD)", scale=alt.Scale(zero=False))
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.markdown("### 🧠 Signal History")
    if st.session_state.signal_history:
        table_df = pd.DataFrame(st.session_state.signal_history[::-1])
        st.dataframe(table_df, use_container_width=True, hide_index=True)


# ========================
# TAB 2 — MODEL METRICS
# ========================
with tab2:
    st.title("🧠 Model Performance Metrics")
    st.caption("XGBoost classifier trained on live Parquet data lake")
    st.divider()

    metrics = load_metrics()

    if metrics is None:
        st.warning("No model metrics found. Run `python -m processing.ml.train_model` first.")
    else:
        st.markdown(f"**Model:** `{metrics.get('model_type', 'Unknown')}`")
        st.markdown(f"**Train rows:** {metrics.get('train_rows')} &nbsp;|&nbsp; **Test rows:** {metrics.get('test_rows')}")
        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{metrics['accuracy']*100:.1f}%")
        m2.metric("Precision", f"{metrics['precision']*100:.1f}%")
        m3.metric("Recall",    f"{metrics['recall']*100:.1f}%")
        m4.metric("F1 Score",  f"{metrics['f1_score']*100:.1f}%")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Confusion Matrix")
            cm = metrics["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual SELL", "Actual BUY"],
                columns=["Predicted SELL", "Predicted BUY"]
            )
            st.dataframe(cm_df, use_container_width=True)

        with col_b:
            import altair as alt
            st.markdown("### Feature Importance")
            fi = metrics["feature_importance"]
            fi_df = pd.DataFrame(
                list(fi.items()),
                columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=False)
            chart = alt.Chart(fi_df).mark_bar(color="#4c9be8").encode(
                x=alt.X("Feature:N", sort="-y", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Importance:Q", title="Importance Score")
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)


# -------------------------
# AUTO REFRESH
# -------------------------
time.sleep(REFRESH_INTERVAL)
st.rerun()