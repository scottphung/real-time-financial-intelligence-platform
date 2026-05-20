# 📈 Real-Time Financial Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Kafka](https://img.shields.io/badge/Kafka-7.4.0-black?logo=apachekafka)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A **production-grade real-time financial intelligence platform** that ingests live crypto market data, processes it through a Kafka streaming pipeline, engineers features, trains an ML model, and serves live BUY/SELL signals through a FastAPI inference service and Streamlit dashboard.

> Built to demonstrate real-world data engineering, ML pipelines, and streaming architecture — not just scripts.

---

## 🖥️ Live Dashboard

![Dashboard Screenshot](assets/dashboard_screenshot.png)

> Live BTC/USD price · ML signal · Model confidence · Signal history table — all updating in real time.

---

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                        │
│              CoinGecko API → Kafka Producer                  │
└──────────────────────────┬──────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                     KAFKA BROKER                             │
│                  Topic: financial-stream                     │
└──────────────────────────┬──────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                   STREAM PROCESSOR                           │
│         Feature Engineering (returns, MA, volatility,        │
│         anomaly detection) → Parquet Data Lake               │
└──────────────┬────────────────────────────┬─────────────────┘
│                            │
▼                            ▼
┌──────────────────────┐      ┌─────────────────────────────┐
│   PARQUET DATA LAKE  │      │      ML TRAINING PIPELINE    │
│  Partitioned by      │─────▶│  Logistic Regression         │
│  year/month/day      │      │  87% Accuracy · model.pkl    │
└──────────────────────┘      └──────────────┬──────────────┘
│
▼
┌─────────────────────────────┐
│   FastAPI INFERENCE SERVICE  │
│   GET  /health               │
│   POST /predict              │
└──────────────┬──────────────┘
│
▼
┌─────────────────────────────┐
│   STREAMLIT DASHBOARD        │
│   Live Price · Signal · Chart│
└─────────────────────────────┘
---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Data Ingestion | CoinGecko API, Kafka Producer |
| Message Broker | Apache Kafka 7.4.0 (Dockerized) |
| Stream Processing | Python Kafka Consumer |
| Feature Engineering | Pandas, NumPy |
| Data Lake | Parquet (partitioned by date) |
| ML Training | scikit-learn Logistic Regression |
| Inference API | FastAPI + Uvicorn + Pydantic |
| Dashboard | Streamlit |
| Containerization | Docker, Docker Compose |
| Language | Python 3.11 |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop running

### 1. Clone the repo
```bash
git clone https://github.com/scottphung/real-time-financial-intelligence-platform.git
cd real-time-financial-intelligence-platform
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 3. Start Kafka
```bash
docker-compose -f streaming/kafka/docker-compose.yml up -d
```

### 4. Start the producer (Terminal 1)
```bash
python -m ingestion.producers.crypto_producer
```

### 5. Start the stream processor (Terminal 2)
```bash
python -m processing.spark_jobs.stream_processor
```

### 6. Train the model (after 2-3 minutes of data)
```bash
python -m processing.ml.train_model
```

### 7. Start the FastAPI inference service (Terminal 3)
```bash
uvicorn api.main:app --reload --port 8000
```

### 8. Launch the dashboard (Terminal 4)
```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` to see the live dashboard.
Open `http://localhost:8000/docs` to explore the API.

---

## 📡 API Reference

### `GET /health`
Returns API and model status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### `POST /predict`
Returns a BUY/SELL signal with confidence score.

**Request:**
```json
{
  "coin": "bitcoin",
  "price": 76848.0,
  "return_pct": 0.002,
  "moving_avg_3": 76800.0,
  "volatility": 13.0,
  "is_anomaly": 0
}
```

**Response:**
```json
{
  "coin": "bitcoin",
  "signal": 1,
  "action": "BUY",
  "confidence": 0.8714,
  "timestamp": "2026-05-19T12:19:51Z"
}
```

---

## 🧠 How It Works

**1. Ingestion** — A Kafka producer polls CoinGecko every 5 seconds for live BTC/USD prices and publishes events to a Kafka topic.

**2. Stream Processing** — A Kafka consumer reads events in real time and engineers features: percentage returns, 10-period moving average, rolling volatility, and anomaly flags.

**3. Data Lake** — Processed events are written to a partitioned Parquet data lake (`year/month/day`) for efficient time-series querying.

**4. ML Training** — A logistic regression model is trained on the lake data to predict whether the next price tick will be higher or lower. Achieves 87% accuracy on held-out data.

**5. Inference API** — A FastAPI service loads the trained model at startup and serves predictions via a REST endpoint with Pydantic-validated request/response schemas.

**6. Dashboard** — A Streamlit dashboard polls the API every 10 seconds, displays live price, BUY/SELL signal, model confidence, a live price chart, and a scrolling signal history table.

---

## 📁 Project Structure
real-time-financial-intelligence-platform/
│
├── ingestion/
│   └── producers/
│       └── crypto_producer.py       # Kafka producer (CoinGecko)
│
├── streaming/
│   └── kafka/
│       └── docker-compose.yml       # Kafka + Zookeeper
│
├── processing/
│   ├── spark_jobs/
│   │   └── stream_processor.py      # Kafka consumer + feature engineering
│   ├── storage/
│   │   └── parquet_writer.py        # Append-only parquet writer
│   └── ml/
│       ├── build_dataset.py         # Feature + label builder
│       ├── train_model.py           # Model training pipeline
│       ├── signal_generator.py      # Inference logic
│       └── backtest.py              # Strategy backtesting
│
├── api/
│   ├── main.py                      # FastAPI app
│   ├── schemas.py                   # Pydantic models
│   └── model_loader.py              # Model lifecycle
│
├── dashboard/
│   └── app.py                       # Streamlit live dashboard
│
├── data_lake/                        # Partitioned parquet storage
├── assets/                           # Screenshots for README
├── requirements.txt
└── README.md
---

## 🔮 Roadmap

- [ ] Docker Compose full stack (one command startup)
- [ ] Upgrade ML model to XGBoost / Random Forest
- [ ] Add Sharpe ratio and drawdown analytics
- [ ] Airflow orchestration for scheduled retraining
- [ ] AWS deployment (ECS + S3 data lake)
- [ ] Multi-asset support (ETH, SOL)
- [ ] CI/CD pipeline with GitHub Actions

---

## 👤 Author

**Scott Phung**
Master's in Computer Science · Finance Background
[GitHub](https://github.com/scottphung) · [LinkedIn](#)