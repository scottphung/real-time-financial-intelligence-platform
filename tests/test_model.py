import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing.ml.signal_generator import generate_signal


def test_generate_signal_no_model_returns_hold():
    """When model is None, signal should be NO_MODEL / HOLD."""
    import processing.ml.signal_generator as sg
    original_model = sg.model
    sg.model = None

    result = generate_signal({
        "price": 45000,
        "return": 0.001,
        "moving_avg_10": 44900,
        "volatility": 100,
        "is_anomaly": 0
    })

    sg.model = original_model

    assert result["signal"] == "NO_MODEL"
    assert result["action"] == "HOLD"


def test_generate_signal_returns_dict():
    """generate_signal must always return a dict with signal and action keys."""
    result = generate_signal({
        "price": 45000,
        "return": 0.001,
        "moving_avg_10": 44900,
        "volatility": 100,
        "is_anomaly": 0
    })
    assert isinstance(result, dict)
    assert "signal" in result
    assert "action" in result


def test_generate_signal_valid_actions():
    """Action must be one of the valid values."""
    result = generate_signal({
        "price": 45000,
        "return": 0.001,
        "moving_avg_10": 44900,
        "volatility": 100,
        "is_anomaly": 0
    })
    assert result["action"] in ["BUY", "SELL", "HOLD", "NO_MODEL"]