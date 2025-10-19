from fastapi import APIRouter, Query
from ..services.predictor_service import get_predictor
from ..models.predictor_models import (
    PredictionOut,
    BasinInfo,
    GlofasOut,
    ForecastPoint,
)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.get("", response_model=PredictionOut)
def predict(
    q: str = Query(
        ..., description="Free-text place (e.g., 'Surrey, BC')"
    )
):
    p = get_predictor()
    out = p.predict_by_query(q)
    if not out:
        return PredictionOut(
            ok=False,
            error="Could not produce forecast for this query"
        )
    return PredictionOut(
        ok=True,
        basin=BasinInfo(**out["basin"]),
        glofas=GlofasOut(current=out["glofas"]["current"],
                         forecast=[
                             ForecastPoint(**f)
                             for f in out["glofas"]["forecast"]
                         ]),
        thresholds={str(k): v for k, v in out["thresholds"].items()},
    )


@router.get("/by-coords", response_model=PredictionOut)
def predict_by_coords(lat: float, lon: float):
    p = get_predictor()
    out = p.predict_by_coords(lat, lon)
    if not out:
        return PredictionOut(
            ok=False,
            error="Could not produce forecast for these coordinates"
        )
    return PredictionOut(
        ok=True,
        basin=BasinInfo(**out["basin"]),
        glofas=GlofasOut(current=out["glofas"]["current"],
                         forecast=[
                             ForecastPoint(**f)
                             for f in out["glofas"]["forecast"]
                         ]),
        thresholds={str(k): v for k, v in out["thresholds"].items()},
    )
