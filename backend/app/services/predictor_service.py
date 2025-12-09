from __future__ import annotations
import os
from typing import Optional, Dict, Tuple, List
from datetime import datetime
import logging

import pandas as pd
import numpy as np
import requests
from scipy.spatial import KDTree
import geopandas as gpd
from shapely.geometry import Point

from .google_flood_service import GoogleFloodService

logger = logging.getLogger(__name__)


class ValidatedFloodPredictor:
    """
    A service that provides flood predictions using Google Flood API.

    This class provides flood prediction capabilities by:
    - Using KDTree for efficient gauge lookup from local data
    - Using Google Flood API to get gauge information and discharge forecasts
    - Comparing forecasts with return periods (2, 5, 20 years)
    - Falling back to Open-Meteo API if Google API is unavailable

    The class exposes two main public methods:
    - predict_by_query(): Get prediction by location name/address
    - predict_by_coords(): Get prediction by latitude/longitude

    Both methods return JSON-friendly dictionaries containing:
    - Basin/gauge information
    - Current and forecasted discharge values
    - Return period thresholds
    - Severity assessments
    """
    # Simple cache for common locations to reduce geocoding calls
    _location_cache: Dict[str, Dict] = {
        "Vancouver, BC": {
            "lat": 49.2827,
            "lon": -123.1207,
            "name": "Vancouver, BC"
        },
    }

    def __init__(self, data_dir: str = None, use_google_api: bool = True):
        self.use_google_api = use_google_api
        if use_google_api:
            self.google_flood_service = GoogleFloodService()
            # Still initialize data_dir for fallback if Google API fails
            if data_dir is None:
                data_dir = os.getenv(
                    "FLOOD_DATA_DIR",
                    os.path.join(
                        os.path.dirname(__file__), "..", "..", "data"
                    )
                )
            self.data_dir = os.path.abspath(data_dir)
            # Don't load data yet, only if fallback is needed
            self._data_loaded = False
        else:
            # Fallback to original implementation
            if data_dir is None:
                raise ValueError(
                    "data_dir is required when use_google_api is False"
                )
            self.data_dir = os.path.abspath(data_dir)
            self.locations_df = self._load_locations()
            self.return_periods_df = self._load_return_periods()
            self.basin_gdf = self._load_basins()
            self._build_indexes()
            self._data_loaded = True

    # -------- file helpers --------
    def _p(self, *parts):
        """
        Helper method to construct file paths within the data directory.

        Args:
            *parts: Path components to join
        Returns:
            Full path string within the data directory
        """
        return os.path.join(self.data_dir, *parts)

    def _load_locations(self) -> pd.DataFrame:
        """
        Load gauge location data from CSV file.

        Returns:
            DataFrame containing gauge locations with columns:
            gauge_id, latitude, longitude
        Raises:
            FileNotFoundError: If the locations CSV file is not found
        """
        path = self._p("global_hybas_locations.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        return pd.read_csv(path, dtype={"gauge_id": str})

    def _load_return_periods(self) -> pd.DataFrame:
        """
        Load return period threshold data from CSV file.

        Filters out gauges with unreasonable return period values (< 1).

        Returns:
            DataFrame containing return period thresholds with columns:
            gauge_id, return_period_2, return_period_5, return_period_20
        Raises:
            FileNotFoundError: If the return periods CSV file is not found
        """
        path = self._p("global_return_periods.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        df = pd.read_csv(path, dtype={"gauge_id": str})
        # Filter out unreasonable values (keep only >= 1)
        return df[df["return_period_2"] >= 1].copy()

    def _load_basins(self):
        """
        Load basin shapefile data using geopandas.

        Returns:
            GeoDataFrame containing basin polygons, or None if file not found
            or loading fails
        """
        shp = self._p("BasinATLAS_v10_lev07.shp")
        if not os.path.exists(shp):
            return None
        try:
            return gpd.read_file(shp)
        except Exception:
            return None

    def _build_indexes(self):
        coords = self.locations_df[["latitude", "longitude"]].values
        self.location_tree = KDTree(coords)
        self.rp_data: Dict[str, Dict[int, float]] = {}

        for _, row in self.return_periods_df.iterrows():
            gid = row["gauge_id"]
            rp = {}
            for col in self.return_periods_df.columns:
                if col.startswith("return_period_"):
                    try:
                        years = int(col.split("_")[-1])
                        val = float(row[col])
                        if pd.notna(val) and val > 0:
                            rp[years] = val
                    except Exception:
                        pass
            if rp:
                self.rp_data[gid] = rp

    # -------- public entry points --------
    def _ensure_fallback_data_loaded(self):
        """Load data files if not already loaded (for fallback)"""
        if not self._data_loaded:
            try:
                self.locations_df = self._load_locations()
                self.return_periods_df = self._load_return_periods()
                self.basin_gdf = self._load_basins()
                self._build_indexes()
                self._data_loaded = True
            except Exception as e:
                logger.error(f"Failed to load fallback data: {e}")
                return False
        return True

    def predict_by_query(self, location: str) -> Optional[Dict]:
        """
        Get flood prediction for a location by name or address.

        This method:
        1. Geocodes the location string to coordinates
        2. Uses KDTree to find the nearest gauge ID from local data
        3. Attempts to retrieve forecast from Google Flood API
        4. Falls back to Open-Meteo API if Google API fails

        Args:
            location: Location string (e.g. "Vancouver, BC")

        Returns:
            Dictionary containing:
                - basin: Gauge/basin information
                - glofas: Current and forecasted discharge values
                - thresholds: Return period thresholds (2, 5, 20 years)
                - max_severity: Maximum severity across forecast period
                - coordinates: Geographic coordinates
            Returns None if geocoding or prediction fails
        """
        if self.use_google_api:
            # search KDTree to find gauge ID from local data
            if not self._ensure_fallback_data_loaded():
                logger.error("Failed to load data files for gauge lookup")
                return None
            coords = self._geocode(location)

            if not coords:
                logger.error(f"Could not geocode location: {location}")
                return None
            lat, lon = coords["lat"], coords["lon"]
            logger.info(
                f"Using KDTree to find gauge for location: {location} "
                f"at ({lat}, {lon})"
            )
            # Use KDTree to find the gauge ID
            basin = self._find_basin(lat, lon)
            if not basin:
                logger.warning(f"Could not find gauge for {location}")
                # Fallback to Open-Meteo
                return self._predict_core(lat, lon)
            gauge_id = basin["gauge_id"]
            logger.info(f"Found gauge ID via KDTree: {gauge_id}")

            # Now try to use Google Flood API with this gauge ID
            try:
                # Get gauge info from Google Flood API
                gauge_info = self.google_flood_service.get_gauge(gauge_id)
                # Get forecast from Google Flood API
                forecast_data = (
                    self.google_flood_service.query_gauge_forecasts(
                        gauge_id, forecast_days=7
                    )
                )

                if gauge_info and forecast_data:
                    logger.info(
                        f"Successfully retrieved data from Google Flood API "
                        f"for gauge {gauge_id}"
                    )

                    # Get thresholds from Google API or use local data
                    thresholds = (
                        self.google_flood_service.get_return_period_thresholds(
                            gauge_info
                        )
                    )

                    # If Google API doesn't have thresholds, use local data
                    if not any(thresholds.values()):
                        thresholds = self.rp_data.get(gauge_id, {})
                        # Convert to Google API format
                        thresholds = {
                            "warning": thresholds.get(2),
                            "danger": thresholds.get(5),
                            "extreme": thresholds.get(20)
                        }

                    # Compare forecast with thresholds
                    forecast_points = (
                        self.google_flood_service
                        .compare_forecast_with_thresholds(
                            forecast_data, thresholds
                        )
                    )

                    if forecast_points:
                        # Format response
                        google_result = {
                            "gauge": {
                                "gauge_id": gauge_id,
                                "name": gauge_info.get("name", ""),
                                "location": gauge_info.get("location", {}),
                                "metadata": gauge_info
                            },
                            "forecast": forecast_points,
                            "thresholds": thresholds,
                            "current_discharge": (
                                forecast_points[0]["discharge"]
                                if forecast_points else None
                            ),
                            "max_severity": max(
                                [fp["severity"] for fp in forecast_points],
                                key=lambda x: [
                                    "normal", "warning", "danger", "extreme"
                                ].index(x)
                            ) if forecast_points else "normal",
                            "coordinates": {"lat": lat, "lon": lon}
                        }
                        return self._format_google_api_response(google_result)
            except Exception as e:
                logger.warning(
                    f"Google Flood API failed for gauge {gauge_id}: {e}. "
                    f"Falling back to Open-Meteo API"
                )

            # Fallback to Open-Meteo if Google API fails
            logger.info("Using Open-Meteo API fallback")
            return self._predict_core(lat, lon)
        else:
            coords = self._geocode(location)
            if not coords:
                return None
            return self._predict_core(coords["lat"], coords["lon"])

    def predict_by_coords(
        self, lat: float, lon: float
    ) -> Optional[Dict]:
        """
        Get flood prediction for a location by geographic coordinates.
        This method:
        1. Uses KDTree to find the nearest gauge ID from local data
        2. Attempts to retrieve forecast from Google Flood API
        3. Falls back to Open-Meteo API if Google API fails
        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
        Returns:
            Dictionary containing:
                - basin: Gauge/basin information
                - glofas: Current and forecasted discharge values
                - thresholds: Return period thresholds (2, 5, 20 years)
                - max_severity: Maximum severity across forecast period
                - coordinates: Geographic coordinates
            Returns None if prediction fails
        """
        if self.use_google_api:
            # First, use KDTree to find gauge ID from local data
            if not self._ensure_fallback_data_loaded():
                logger.error("Failed to load data files for gauge lookup")
                return None

            logger.info(
                f"Using KDTree to find gauge for coordinates: ({lat}, {lon})"
            )
            # Use KDTree to find the gauge ID
            basin = self._find_basin(lat, lon)
            if not basin:
                logger.warning(
                    f"Could not find gauge for coordinates ({lat}, {lon})"
                )
                # Fallback to Open-Meteo
                return self._predict_core(lat, lon)

            gauge_id = basin["gauge_id"]
            logger.info(f"Found gauge ID via KDTree: {gauge_id}")

            # Use Google Flood API with this gauge ID
            try:
                # Get gauge info from Google Flood API
                gauge_info = self.google_flood_service.get_gauge(gauge_id)
                # Get forecast from Google Flood API
                forecast_data = (
                    self.google_flood_service.query_gauge_forecasts(
                        gauge_id, forecast_days=7
                    )
                )

                if gauge_info and forecast_data:
                    logger.info(
                        f"Successfully retrieved data from Google Flood API "
                        f"for gauge {gauge_id}"
                    )
                    # Get thresholds from Google API or use local data
                    thresholds = (
                        self.google_flood_service.get_return_period_thresholds(
                            gauge_info
                        )
                    )
                    # If Google API doesn't have thresholds, use local data
                    if not any(thresholds.values()):
                        thresholds = self.rp_data.get(gauge_id, {})
                        # Convert to Google API format
                        thresholds = {
                            "warning": thresholds.get(2),
                            "danger": thresholds.get(5),
                            "extreme": thresholds.get(20)
                        }
                    # Compare forecast with thresholds
                    forecast_points = (
                        self.google_flood_service
                        .compare_forecast_with_thresholds(
                            forecast_data, thresholds
                        )
                    )
                    if forecast_points:
                        # Format response
                        google_result = {
                            "gauge": {
                                "gauge_id": gauge_id,
                                "name": gauge_info.get("name", ""),
                                "location": gauge_info.get("location", {}),
                                "metadata": gauge_info
                            },
                            "forecast": forecast_points,
                            "thresholds": thresholds,
                            "current_discharge": (
                                forecast_points[0]["discharge"]
                                if forecast_points else None
                            ),
                            "max_severity": max(
                                [fp["severity"] for fp in forecast_points],
                                key=lambda x: [
                                    "normal", "warning", "danger", "extreme"
                                ].index(x)
                            ) if forecast_points else "normal",
                            "coordinates": {"lat": lat, "lon": lon}
                        }
                        return self._format_google_api_response(google_result)
            except Exception as e:
                logger.warning(
                    f"Google Flood API failed for gauge {gauge_id}: {e}. "
                    f"Falling back to Open-Meteo API"
                )
            # Fallback to Open-Meteo if Google API fails
            logger.info("Using Open-Meteo API fallback")
            return self._predict_core(lat, lon)
        else:
            return self._predict_core(lat, lon)

    def _format_google_api_response(self, google_result: Dict) -> Dict:
        """
        Format Google Flood API response to match the expected output format.

        Converts the Google API response structure to the standardized format
        used by the application, including:
        - Converting threshold keys from "warning"/"danger"/"extreme"
          to "2"/"5"/"20"
        - Formatting forecast points with date, discharge, and severity
        - Extracting current discharge value
        Args:
            google_result: Raw response dictionary from Google Flood API
        Returns:
            Formatted dictionary matching the expected output structure
        """
        gauge_info = google_result.get("gauge", {})
        forecast_points = google_result.get("forecast", [])
        thresholds = google_result.get("thresholds", {})

        # Format forecast to match expected structure
        forecast = [
            {
                "date": fp["date"],
                "discharge": fp["discharge"],
                "severity": fp["severity"],
                "level": fp["level"]
            }
            for fp in forecast_points
        ]

        # Format thresholds to match expected structure (years -> values)
        formatted_thresholds = {}
        if thresholds.get("warning"):
            # 2-year return period
            formatted_thresholds["2"] = thresholds["warning"]
        if thresholds.get("danger"):
            # 5-year return period
            formatted_thresholds["5"] = thresholds["danger"]
        if thresholds.get("extreme"):
            # 20-year return period
            formatted_thresholds["20"] = thresholds["extreme"]

        # Get current discharge (first forecast point or latest)
        current_discharge = google_result.get("current_discharge")
        if not current_discharge and forecast_points:
            current_discharge = forecast_points[0]["discharge"]

        return {
            "basin": {
                "gauge_id": gauge_info.get("gauge_id", ""),
                "name": gauge_info.get("name", ""),
                # Google API provides closest gauge
                "distance_km": 0.0,
                "method": "Google Flood API"
            },
            "glofas": {
                "current": current_discharge or 0.0,
                "forecast": forecast
            },
            "thresholds": formatted_thresholds,
            "max_severity": google_result.get("max_severity", "normal"),
            "coordinates": google_result.get("coordinates", {})
        }

    # -------- internals --------
    def _geocode(self, location: str, retries: int = 2):
        """
        Geocode a location string to coordinates with retry logic.

        Args:
            location: Location string to geocode
            retries: Number of retry attempts (default: 2)
        Returns:
            Dictionary with lat, lon, and name, or None if failed
        """
        # Check cache first
        location_key = location.strip()
        if location_key in self._location_cache:
            return self._location_cache[location_key]

        for attempt in range(retries + 1):
            try:
                r = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": location, "format": "json", "limit": 1},
                    headers={"User-Agent": "ValidatedFloodPredictor"},
                    timeout=20,  # Increased from 10 to 20 seconds
                )
                if r.status_code == 200 and r.json():
                    d = r.json()[0]
                    return {
                        "lat": float(d["lat"]),
                        "lon": float(d["lon"]),
                        "name": d["display_name"],
                    }
            except requests.exceptions.Timeout:
                if attempt < retries:
                    logger.warning(
                        f"Geocoding timeout for '{location}', "
                        f"retrying ({attempt + 1}/{retries})..."
                    )
                    continue
                else:
                    logger.error(
                        f"Geocoding timeout for '{location}' after "
                        f"{retries + 1} attempts"
                    )
            except requests.exceptions.RequestException as e:
                if attempt < retries:
                    logger.warning(
                        f"Geocoding error for '{location}': {e}, "
                        f"retrying ({attempt + 1}/{retries})..."
                    )
                    continue
                else:
                    logger.error(f"Geocoding failed for '{location}': {e}")
            except Exception as e:
                logger.error(f"Unexpected error geocoding '{location}': {e}")
                break
        return None

    def _calculate_severity(
        self, discharge: float, thresholds: Dict[int, float]
    ) -> Tuple[str, str]:
        """
        Calculate severity level based on discharge and thresholds.

        Returns:
            Tuple of (severity, level) where both are the same value:
            "normal", "warning" (2-year), "danger" (5-year),
            or "extreme" (20-year)
        """
        # 2-year return period
        warning_threshold = thresholds.get(2)
        # 5-year return period
        danger_threshold = thresholds.get(5)
        # 20-year return period
        extreme_threshold = thresholds.get(20)

        if extreme_threshold and discharge >= extreme_threshold:
            return ("extreme", "extreme")
        elif danger_threshold and discharge >= danger_threshold:
            return ("danger", "danger")
        elif warning_threshold and discharge >= warning_threshold:
            return ("warning", "warning")
        else:
            return ("normal", "normal")

    def _predict_core(self, lat: float, lon: float) -> Optional[Dict]:
        basin = self._find_basin(lat, lon)
        if not basin:
            logger.warning(
                f"Could not find basin for coordinates: ({lat}, {lon})"
            )
            return None

        main = self._find_main_river_coords(lat, lon)
        if not main:
            logger.warning("Could not find main river coordinates")
            return None

        glofas = self._get_glofas_data(*main)
        if not glofas:
            logger.warning("Could not retrieve discharge data from Open-Meteo")
            return None

        thresholds = self.rp_data.get(basin["gauge_id"], {})
        if not thresholds:
            gauge_id = basin['gauge_id']
            logger.warning(
                f"No return period thresholds found for gauge: {gauge_id}"
            )

        # Calculate severity for forecast points
        forecast_with_severity = []
        max_severity = "normal"
        severity_levels = ["normal", "warning", "danger", "extreme"]

        for point in glofas.get("forecast", []):
            discharge = point.get("discharge", 0)
            severity, level = self._calculate_severity(
                discharge, thresholds
            )
            point["severity"] = severity
            point["level"] = level
            forecast_with_severity.append(point)

            # Track maximum severity
            current_idx = severity_levels.index(severity)
            max_idx = severity_levels.index(max_severity)
            if current_idx > max_idx:
                max_severity = severity

        # Update glofas with severity-calculated forecast
        glofas["forecast"] = forecast_with_severity

        # Calculate severity for current discharge
        current_discharge = glofas.get("current", 0)
        current_severity, _ = self._calculate_severity(
            current_discharge, thresholds
        )
        current_idx = severity_levels.index(current_severity)
        max_idx = severity_levels.index(max_severity)
        if current_idx > max_idx:
            max_severity = current_severity

        return {
            "basin": basin,
            "glofas": glofas,
            "thresholds": thresholds,
            "max_severity": max_severity
        }

    def _find_basin(self, lat: float, lon: float) -> Optional[Dict]:
        # Prefer precise polygon match
        if self.basin_gdf is not None:
            try:
                point = Point(lon, lat)
                matches = self.basin_gdf[
                    self.basin_gdf.geometry.contains(point)
                ]
                if not matches.empty:
                    row = matches.iloc[0]
                    gid = f"hybas_{row['HYBAS_ID']}"
                    if gid in self.rp_data:
                        return {
                            "gauge_id": gid,
                            "distance_km": 0.0,
                            "method": "Point-in-Polygon",
                        }
            except Exception:
                pass
        # Fallback KDTree
        dists, idxs = self.location_tree.query([lat, lon], k=200)
        candidates = []
        for dist, idx in zip(dists, idxs):
            km = float(dist) * 111
            if km > 50:
                break
            gid = self.locations_df.iloc[int(idx)]["gauge_id"]
            if gid in self.rp_data:
                t2 = self.rp_data[gid].get(2, 0)
                candidates.append({
                    "gauge_id": gid,
                    "distance_km": km,
                    "t2": t2
                })
        if not candidates:
            return None
        majors = [c for c in candidates if c["t2"] > 1000]
        best = (
            min(majors, key=lambda c: c["distance_km"])
            if majors
            else max(candidates, key=lambda c: c["t2"])
        )
        return {
            "gauge_id": best["gauge_id"],
            "distance_km": best["distance_km"],
            "method": "KDTree-MajorRiver"
        }

    def _find_main_river_coords(
        self, lat: float, lon: float
    ) -> Optional[Tuple[float, float]]:
        # Optimized: Try center point first, then smaller grid if needed
        # This reduces API calls and improves performance
        offsets = np.linspace(-0.05, 0.05, 3)  # Reduced from 5 to 3
        grid = [(lat + dx, lon + dy) for dx in offsets for dy in offsets]
        vals: List[Tuple[float, float, float]] = []
        # Try center point first (most likely to have data)
        center_point = (lat, lon)
        try:
            r = requests.get(
                "https://flood-api.open-meteo.com/v1/flood",
                params={
                    "latitude": center_point[0],
                    "longitude": center_point[1],
                    "daily": "river_discharge",
                    "past_days": 1,
                    "forecast_days": 7
                },
                timeout=5,
            )
            if r.status_code == 200:
                q = r.json().get("daily", {}).get("river_discharge", [])
                if q:
                    v = q[-2] if len(q) > 1 else q[-1]
                    if v and v > 0:
                        # If center point works, use it
                        return center_point
        except Exception:
            pass

        # If center didn't work, try grid (but with early exit)
        for la, lo in grid:
            if (la, lo) == center_point:
                continue  # Already tried
            try:
                r = requests.get(
                    "https://flood-api.open-meteo.com/v1/flood",
                    params={
                        "latitude": la,
                        "longitude": lo,
                        "daily": "river_discharge",
                        "past_days": 1,
                        "forecast_days": 7
                    },
                    timeout=5,
                )
                if r.status_code == 200:
                    q = r.json().get("daily", {}).get("river_discharge", [])
                    if q:
                        v = q[-2] if len(q) > 1 else q[-1]
                        if v and v > 0:
                            vals.append((la, lo, float(v)))
                            # Early exit if we find a good value
                            if v > 10:  # Reasonable threshold
                                return (la, lo)
            except Exception:
                pass
        if not vals:
            return None
        # simple clustering by distance < 10 km
        clusters = []
        for la, lo, v in sorted(vals, key=lambda x: x[2], reverse=True):
            placed = False
            for c in clusters:
                import math
                dist_km = (
                    math.sqrt((la - c["lat"])**2 + (lo - c["lon"])**2) * 111
                )
                if dist_km < 10:
                    c["pts"].append((la, lo, v))
                    c["lat"] = float(np.mean([p[0] for p in c["pts"]]))
                    c["lon"] = float(np.mean([p[1] for p in c["pts"]]))
                    c["mean"] = float(np.mean([p[2] for p in c["pts"]]))
                    placed = True
                    break
            if not placed:
                clusters.append({
                    "lat": la,
                    "lon": lo,
                    "mean": v,
                    "pts": [(la, lo, v)]
                })
        best = max(clusters, key=lambda c: c["mean"])
        return (best["lat"], best["lon"]) if best["mean"] > 5 else None

    def _get_glofas_data(self, lat: float, lon: float) -> Optional[Dict]:
        try:
            r = requests.get(
                "https://flood-api.open-meteo.com/v1/flood",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "river_discharge",
                    "forecast_days": 7
                },
                timeout=10,
            )
            if r.status_code == 200:
                d = r.json().get("daily", {})
                q = d.get("river_discharge", [])
                t = d.get("time", [])
                if q and t:
                    today = datetime.utcnow().strftime("%Y-%m-%d")
                    current = float(q[-2] if len(q) > 1 else q[-1])
                    forecast = [
                        {"date": ti, "discharge": float(qi)}
                        for ti, qi in zip(t, q)
                        if ti > today
                    ][:7]
                    return {"current": current, "forecast": forecast}
        except Exception:
            pass
        return None


# Singleton factory (only load data once)
_predictor_singleton: Optional[ValidatedFloodPredictor] = None


def get_predictor(use_google_api: bool = True) -> ValidatedFloodPredictor:
    """
    Get or create the flood predictor singleton.

    Args:
        use_google_api: If True, use Google Flood API (default).
                       If False, use local data files.
    """
    global _predictor_singleton
    if _predictor_singleton is None:
        if use_google_api:
            _predictor_singleton = ValidatedFloodPredictor(
                use_google_api=True
            )
        else:
            data_dir = os.getenv(
                "FLOOD_DATA_DIR",
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "data"
                )
            )
            _predictor_singleton = ValidatedFloodPredictor(
                data_dir=os.path.abspath(data_dir),
                use_google_api=False
            )
    return _predictor_singleton
