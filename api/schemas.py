from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime


# -------------------------
# REQUEST
# -------------------------
class PredictRequest(BaseModel):
    coin: str = Field(..., example="bitcoin")
    price: float = Field(..., gt=0, example=45000.0)
    return_pct: float = Field(..., example=0.002)
    moving_avg_3: float = Field(..., example=44800.0)
    volatility: float = Field(..., ge=0, example=120.5)
    is_anomaly: int = Field(..., ge=0, le=1, example=0)


# -------------------------
# RESPONSE
# -------------------------
class PredictResponse(BaseModel):
    coin: str
    signal: int                          # 1 = BUY, 0 = SELL
    action: Literal["BUY", "SELL", "HOLD", "NO_MODEL"]
    confidence: float                    # probability of BUY signal
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"