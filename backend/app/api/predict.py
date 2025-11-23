"""
FastAPI router for flood prediction endpoints.

This module provides REST API endpoints for querying flood predictions
by location name or geographic coordinates. It uses the Google Flood API
to retrieve gauge information, discharge forecasts, and severity assessments.
"""

from fastapi import APIRouter, Query
from ..services.predictor_service import get_predictor
from ..models.predictor_models import (
    PredictionOut,
    BasinInfo,
    GlofasOut,
    ForecastPoint,
    Coordinates,
)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.get("", response_model=PredictionOut)
def predict(
    q: str = Query(
        None, description="Free-text place (e.g., 'Surrey, BC')"
    ),
    lat: float = Query(
        None, description="Latitude in decimal degrees"
    ),
    lon: float = Query(
        None, description="Longitude in decimal degrees"
    )
):
    """
    Get flood prediction for a location.

    This endpoint accepts either:
    - A location string (city name, address, etc.) via 'q' parameter
    - Geographic coordinates via 'lat' and 'lon' parameters

    Both methods use the same flow: KDTree to find gauge → Google Flood API
    → 7-day discharge forecast with severity assessment based on return
    period thresholds (2-year, 5-year, and 20-year).

    Args:
        q: Free-text location query (e.g., "Surrey, BC", "Vancouver, BC")
        lat: Latitude in decimal degrees (e.g., 49.1913)
        lon: Longitude in decimal degrees (e.g., -122.8490)

    Returns:
        PredictionOut: Prediction response containing:
            - Basin/gauge information
            - Current and forecasted discharge values
            - Return period thresholds
            - Maximum severity level across forecast period
            - Geographic coordinates

    Examples:
        GET /predict?q=Surrey,BC
        GET /predict?lat=49.1913&lon=-122.8490
    """
    p = get_predictor(use_google_api=True)
    # Determine which method to use based on provided parameters
    if q:
        # Location string provided - geocode then use KDTree
        out = p.predict_by_query(q)
        error_msg = "Could not produce forecast for this query"
    elif lat is not None and lon is not None:
        # Coordinates provided - use KDTree directly
        out = p.predict_by_coords(lat, lon)
        error_msg = "Could not produce forecast for these coordinates"
    else:
        return PredictionOut(
            ok=False,
            error=(
                "Either 'q' (location) or both 'lat' and 'lon' "
                "must be provided"
            )
        )
    if not out:
        return PredictionOut(ok=False, error=error_msg)

    # Build response with all required fields
    response_data = {
        "ok": True,
        "basin": BasinInfo(**out["basin"]),
        "glofas": GlofasOut(
            current=out["glofas"]["current"],
            forecast=[
                ForecastPoint(**f)
                for f in out["glofas"]["forecast"]
            ]
        ),
        # Convert threshold keys to strings for JSON serialization
        "thresholds": {str(k): v for k, v in out["thresholds"].items()},
    }
    # Add optional fields if present in the response
    if "max_severity" in out:
        response_data["max_severity"] = out["max_severity"]
    if "coordinates" in out and out["coordinates"]:
        response_data["coordinates"] = Coordinates(**out["coordinates"])
    return PredictionOut(**response_data)
