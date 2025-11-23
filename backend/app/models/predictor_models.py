"""
Pydantic models for flood prediction API requests and responses.

This module defines the data structures used for flood prediction,
including forecast points, basin information, and prediction outputs.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional


class ForecastPoint(BaseModel):
    """
    A single forecast point in the discharge prediction timeline.

    Attributes:
        date: Date string in ISO format (YYYY-MM-DD)
        discharge: Predicted discharge value in cubic meters per second
        severity: Flood severity level based on return period thresholds
        level: Alias for severity (maintained for backward compatibility)
    """
    date: str
    discharge: float
    # "normal", "warning", "danger", "extreme"
    severity: Optional[str] = None
    # Same as severity, maintained for compatibility
    level: Optional[str] = None


class BasinInfo(BaseModel):
    """
    Information about the river basin and gauge used for prediction.

    Attributes:
        gauge_id: Unique identifier for the gauge station
        distance_km: Distance from query location to gauge in kilometers
        method: Method used to find the gauge (e.g., "Google Flood API")
        name: Optional name of the gauge or basin
    """
    gauge_id: str
    distance_km: float
    method: str
    name: Optional[str] = None


class GlofasOut(BaseModel):
    """
    GLOFAS (Global Flood Awareness System) forecast data.

    Attributes:
        current: Current discharge value in cubic meters per second
        forecast: List of forecast points for the prediction period
    """
    current: float
    forecast: List[ForecastPoint]


class Coordinates(BaseModel):
    """
    Geographic coordinates for a location.

    Attributes:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
    """
    lat: float
    lon: float


class PredictionOut(BaseModel):
    """
    Complete flood prediction response from the API.

    Attributes:
        ok: Whether the prediction was successful
        basin: Information about the basin and gauge used
        glofas: GLOFAS forecast data including current and predicted values
        thresholds: Return period thresholds in m³/s
                   (keys: "2", "5", "20" for 2-year, 5-year, 20-year)
        max_severity: Maximum severity level across the forecast period
        coordinates: Geographic coordinates of the queried location
        error: Error message if prediction failed (ok=False)
    """
    ok: bool
    basin: Optional[BasinInfo] = None
    glofas: Optional[GlofasOut] = None
    thresholds: Optional[Dict[str, float]] = None
    # "normal", "warning", "danger", "extreme"
    max_severity: Optional[str] = None
    coordinates: Optional[Coordinates] = None
    error: Optional[str] = None
