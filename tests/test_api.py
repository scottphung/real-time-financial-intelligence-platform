import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_returns_correct_schema():
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data
    assert data["status"] == "ok"


def test_predict_endpoint_returns_200():
    payload = {
        "coin": "bitcoin",
        "price": 45000.0,
        "return_pct": 0.001,
        "moving_avg_3": 44900.0,
        "volatility": 100.0,
        "is_anomaly": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_endpoint_returns_correct_schema():
    payload = {
        "coin": "bitcoin",
        "price": 45000.0,
        "return_pct": 0.001,
        "moving_avg_3": 44900.0,
        "volatility": 100.0,
        "is_anomaly": 0
    }
    response = client.post("/predict", json=payload)
    data = response.json()
    assert "coin" in data
    assert "signal" in data
    assert "action" in data
    assert "confidence" in data
    assert "timestamp" in data


def test_predict_endpoint_invalid_payload_returns_422():
    response = client.post("/predict", json={"coin": "bitcoin"})
    assert response.status_code == 422


def test_predict_endpoint_negative_price_returns_422():
    payload = {
        "coin": "bitcoin",
        "price": -1.0,
        "return_pct": 0.001,
        "moving_avg_3": 44900.0,
        "volatility": 100.0,
        "is_anomaly": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422