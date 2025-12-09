import os
import requests
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class GoogleFloodService:
    """
    Service for interacting with Google Flood Forecasting API.
    
    Provides methods to get gauge information and retrieve discharge
    forecasts with return period comparisons. Used by predictor_service
    which handles gauge lookup via KDTree.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("FLOODS_API_KEY", "AIzaSyD3XKe77euWDHmzpaN2EUR6iOwciI1X-Zw")
        self.base_url = "https://floodforecasting.googleapis.com/v1"
        
    def _make_request(self, endpoint: str, method: str = "GET", params: Dict = None, json_data: Dict = None) -> Optional[Dict]:
        """Make a request to the Google Flood API"""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json"
        }
        
        request_params = params or {}
        request_params["key"] = self.api_key
        
        try:
            if method == "GET":
                response = requests.get(url, params=request_params, headers=headers, timeout=30)
            elif method == "POST":
                response = requests.post(url, params=request_params, json=json_data, headers=headers, timeout=30)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            # Log response for debugging
            if response.status_code != 200:
                logger.warning(
                    f"API returned status {response.status_code} for {endpoint}. "
                    f"Response: {response.text[:500]}"
                )      
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            error_text = e.response.text[:200] if e.response else str(e)       
            # Handle 404 specifically - gauge not found (this is expected for some gauges)
            if status_code == 404:
                # Try to parse error message
                try:
                    error_json = e.response.json()
                    error_msg = error_json.get("error", {}).get("message", "")
                    if "Requested entity was not found" in error_msg:
                        logger.info(
                            f"Gauge not found in Google Flood API: {endpoint}. "
                            f"This is expected - not all gauges are available in the API."
                        )
                    else:
                        logger.warning(
                            f"Google Flood API endpoint not found: {endpoint}"
                        )
                except Exception:
                    logger.warning(
                        f"Google Flood API returned 404 for {endpoint}"
                    )
            else:
                logger.error(
                    f"HTTP error for {endpoint}: {status_code} - {error_text}"
                )
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {endpoint}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in _make_request: {e}")
            return None
    
    def get_gauge(self, gauge_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific gauge.
        
        Args:
            gauge_id: The unique identifier for the gauge
            
        Returns:
            Dictionary containing gauge metadata including location, thresholds, etc.
        """
        # Try the standard REST endpoint first
        endpoint = f"gauges/{gauge_id}"
        response = self._make_request(endpoint, method="GET")
        
        return response
    
    def query_gauge_forecasts(self, gauge_id: str, forecast_days: int = 7) -> Optional[Dict]:
        """
        Query discharge forecasts for a specific gauge.
        
        Uses the correct Google Flood API endpoint: gauges:queryGaugeForecasts
        
        Args:
            gauge_id: The unique identifier for the gauge
            forecast_days: Number of days to forecast (default: 7, may not be used by API)
            
        Returns:
            Dictionary containing forecast data with dates and discharge values
        """
        # Use the correct endpoint structure: gauges:queryGaugeForecasts
        # According to Google API, only gaugeIds is needed as query parameter
        endpoint = "gauges:queryGaugeForecasts"
        params = {
            "gaugeIds": gauge_id
            # Note: forecastDays parameter may not be supported in query params
            # The API likely returns a default forecast period
        }
        response = self._make_request(endpoint, method="GET", params=params)
        return response
    
    def get_return_period_thresholds(self, gauge_info: Dict) -> Dict[str, float]:
        """
        Extract return period thresholds from gauge information.
        
        Args:
            gauge_info: Dictionary containing gauge metadata
            
        Returns:
            Dictionary mapping return period names to threshold values
            {
                "warning": <2-year threshold>,
                "danger": <5-year threshold>,
                "extreme": <20-year threshold>
            }
        """
        thresholds = {
            "warning": None,  # 2-year return period
            "danger": None,   # 5-year return period
            "extreme": None   # 20-year return period
        }
        
        # Try to extract thresholds from gauge model metadata
        if "model" in gauge_info and "thresholds" in gauge_info["model"]:
            threshold_data = gauge_info["model"]["thresholds"]
            
            # Look for return period thresholds
            if "returnPeriod2Years" in threshold_data:
                thresholds["warning"] = float(threshold_data["returnPeriod2Years"])
            if "returnPeriod5Years" in threshold_data:
                thresholds["danger"] = float(threshold_data["returnPeriod5Years"])
            if "returnPeriod20Years" in threshold_data:
                thresholds["extreme"] = float(threshold_data["returnPeriod20Years"])
                
            # Alternative field names
            if "warningLevel" in threshold_data:
                thresholds["warning"] = float(threshold_data["warningLevel"])
            if "dangerLevel" in threshold_data:
                thresholds["danger"] = float(threshold_data["dangerLevel"])
            if "extremeLevel" in threshold_data:
                thresholds["extreme"] = float(threshold_data["extremeLevel"])
        
        # Try alternative paths in the response structure
        if not any(thresholds.values()):
            if "thresholds" in gauge_info:
                threshold_data = gauge_info["thresholds"]
                thresholds["warning"] = threshold_data.get("2", threshold_data.get("warning"))
                thresholds["danger"] = threshold_data.get("5", threshold_data.get("danger"))
                thresholds["extreme"] = threshold_data.get("20", threshold_data.get("extreme"))
        
        return thresholds
    
    def compare_forecast_with_thresholds(self, forecast_data: Dict, thresholds: Dict[str, float]) -> List[Dict]:
        """
        Compare forecasted discharge values with return period thresholds.
        
        Args:
            forecast_data: Dictionary containing forecast data from Google Flood API
            thresholds: Dictionary with warning, danger, and extreme thresholds
            
        Returns:
            List of forecast points with severity assessment
        """
        forecast_points = []
        
        # Handle Google Flood API response structure
        # Response format: {"forecasts": {"gauge_id": {"forecasts": [...]}}}
        if "forecasts" in forecast_data:
            # Get the first gauge's forecasts (we only query one gauge)
            gauge_forecasts = forecast_data["forecasts"]
            gauge_id = list(gauge_forecasts.keys())[0] if gauge_forecasts else None
            
            if gauge_id and "forecasts" in gauge_forecasts[gauge_id]:
                # Get the most recent forecast (last one in the list)
                all_forecasts = gauge_forecasts[gauge_id]["forecasts"]
                if all_forecasts:
                    latest_forecast = all_forecasts[-1]
                    forecast_ranges = latest_forecast.get("forecastRanges", [])
                    
                    # Convert forecastRanges to our format
                    for range_data in forecast_ranges:
                        start_time = range_data.get("forecastStartTime", "")
                        value = range_data.get("value", 0)
                        
                        # Use the start time as the date
                        if start_time:
                            # Parse ISO format: "2025-11-23T00:00:00Z" -> "2025-11-23"
                            date_str = start_time.split("T")[0]
                            
                            # Determine severity
                            severity = "normal"
                            level = "normal"
                            
                            if thresholds.get("extreme") and value >= thresholds["extreme"]:
                                severity = "extreme"
                                level = "extreme"
                            elif thresholds.get("danger") and value >= thresholds["danger"]:
                                severity = "danger"
                                level = "danger"
                            elif thresholds.get("warning") and value >= thresholds["warning"]:
                                severity = "warning"
                                level = "warning"
                            
                            forecast_points.append({
                                "date": date_str,
                                "discharge": float(value),
                                "severity": severity,
                                "level": level,
                                "warning_threshold": thresholds["warning"],
                                "danger_threshold": thresholds["danger"],
                                "extreme_threshold": thresholds["extreme"]
                            })
                    
                    return forecast_points
        
        # Fallback: Handle other response structures
        if "forecast" in forecast_data:
            forecast = forecast_data["forecast"]
        else:
            forecast = forecast_data
            
        # Handle different possible response structures
        if "daily" in forecast:
            dates = forecast.get("time", [])
            discharges = forecast.get("river_discharge", [])
        elif "points" in forecast:
            points = forecast["points"]
            dates = [p.get("date") for p in points]
            discharges = [p.get("discharge") for p in points]
        elif isinstance(forecast, list):
            dates = [f.get("date") for f in forecast]
            discharges = [f.get("discharge") for f in forecast]
        else:
            logger.warning("Unknown forecast data structure")
            return []
        
        # Compare each forecast point with thresholds
        for date, discharge in zip(dates, discharges):
            if discharge is None:
                continue
                
            discharge_value = float(discharge)
            severity = "normal"
            level = "normal"
            
            # Determine severity level
            if thresholds.get("extreme") and discharge_value >= thresholds["extreme"]:
                severity = "extreme"
                level = "extreme"
            elif thresholds.get("danger") and discharge_value >= thresholds["danger"]:
                severity = "danger"
                level = "danger"
            elif thresholds.get("warning") and discharge_value >= thresholds["warning"]:
                severity = "warning"
                level = "warning"
            
            forecast_points.append({
                "date": date,
                "discharge": discharge_value,
                "severity": severity,
                "level": level,
                "warning_threshold": thresholds["warning"],
                "danger_threshold": thresholds["danger"],
                "extreme_threshold": thresholds["extreme"]
            })
        
        return forecast_points

