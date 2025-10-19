from pydantic import BaseModel
from typing import List, Dict, Optional


class ForecastPoint(BaseModel):
    date: str
    discharge: float


class BasinInfo(BaseModel):
    gauge_id: str
    distance_km: float
    method: str


class GlofasOut(BaseModel):
    current: float
    forecast: List[ForecastPoint]


class PredictionOut(BaseModel):
    ok: bool
    basin: Optional[BasinInfo] = None
    glofas: Optional[GlofasOut] = None
    thresholds: Optional[Dict[str, float]] = None
    error: Optional[str] = None
