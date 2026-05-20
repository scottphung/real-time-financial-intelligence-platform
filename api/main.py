from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from api.schemas import PredictRequest, PredictResponse, HealthResponse
from api.model_loader import load_model, get_model


# -------------------------
# LIFESPAN (startup/shutdown)
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once on server startup
    load_model()
    yield
    # Runs on shutdown (cleanup goes here if needed)
    print("🛑 API shutting down")


# -------------------------
# APP
# -------------------------
app = FastAPI(
    title="Financial Intelligence API",
    description="Real-time ML signal generation for crypto assets",
    version="1.0.0",
    lifespan=lifespan
)


# -------------------------
# ENDPOINTS
# -------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Check API and model status."""
    return HealthResponse(
        status="ok",
        model_loaded=get_model() is not None
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Given feature inputs for a coin, return a BUY/SELL signal
    and the model's confidence score.
    """
    model = get_model()

    if model is None:
        return PredictResponse(
            coin=request.coin,
            signal=0,
            action="NO_MODEL",
            confidence=0.0,
            timestamp=datetime.now(timezone.utc)
        )

    features = [[
        request.price,
        request.return_pct,
        request.moving_avg_3,
        request.volatility,
        request.is_anomaly
    ]]

    try:
        pred = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0]
        confidence = round(float(proba[pred]), 4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return PredictResponse(
        coin=request.coin,
        signal=pred,
        action="BUY" if pred == 1 else "SELL",
        confidence=confidence,
        timestamp=datetime.now(timezone.utc)
    )