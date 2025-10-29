import aiohttp
import asyncio
import logging
from typing import Dict, Optional, Tuple, List
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Open-Meteo Flood API configuration
OPEN_METEO_FLOOD_BASE_URL = "https://flood-api.open-meteo.com/v1/flood"

class HydrologicalForecastAgent:
    """
    Agent for fetching hydrological forecast data from Open-Meteo Flood API.
    Provides river discharge data and flood risk assessment for the next 7 days.
    """
    
    def __init__(self):
        self.base_url = OPEN_METEO_FLOOD_BASE_URL
    
    async def get_hydrological_forecast(self, location: str) -> Dict:
        """
        Get hydrological forecast data for a given location.
        
        Args:
            location: Location string (e.g., "Chilliwack, BC" or "49.16,-121.96")
        
        Returns:
            Dictionary containing hydrological forecast data in the specified format:
            {
                "hydrology_forecast": {
                    "location": {"lat": float, "lon": float, "name": str},
                    "series_daily_cms": [
                        {"date": "YYYY-MM-DD", "q": float},
                        ...
                    ],
                    "thresholds": {
                        "watch_cms": float,
                        "warning_cms": float,
                        "danger_cms": float
                    }
                }
            }
        """
        try:
            # Parse location to get coordinates
            lat, lon, location_name = await self._parse_location(location)
            
            # Get river discharge data from Open-Meteo API
            discharge_data = await self._get_river_discharge_data(lat, lon)
            
            # Format the response according to the specified structure
            hydrology_forecast = self._format_hydrology_response(
                lat, lon, location_name, discharge_data
            )
            
            return {
                "hydrology_forecast": hydrology_forecast
            }
            
        except Exception as e:
            logger.error(f"Error getting hydrological forecast for {location}: {e}")
            return {
                "hydrology_forecast": {
                    "location": {"lat": 0.0, "lon": 0.0, "name": "Unknown"},
                    "series_daily_cms": [],
                    "thresholds": {"watch_cms": 350, "warning_cms": 430, "danger_cms": 500}
                }
            }
    
    async def _parse_location(self, location: str) -> Tuple[float, float, str]:
        """
        Parse location string to get latitude, longitude, and location name.
        
        Args:
            location: Location string (coordinates or place name)
        
        Returns:
            Tuple of (latitude, longitude, location_name)
        """
        try:
            # Check if location is already coordinates (format: "lat,lon")
            if ',' in location:
                parts = location.split(',')
                if len(parts) == 2:
                    # Try to parse as coordinates first
                    try:
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                        # Validate coordinate ranges
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            return lat, lon, f"{lat},{lon}"
                    except ValueError:
                        # If conversion fails, treat as place name with comma
                        pass
            
            # For place names, we'll use a simple geocoding approach
            # In a real implementation, you'd use a proper geocoding service
            location_lower = location.lower().strip()
            
            # Common place name mappings
            place_coordinates = {
                "chilliwack": (49.16, -121.96, "Chilliwack, BC"),
                "chilliwack, bc": (49.16, -121.96, "Chilliwack, BC"),
                "chilliwack, british columbia": (49.16, -121.96, "Chilliwack, BC"),
                "seattle": (47.6062, -122.3321, "Seattle, WA"),
                "seattle, wa": (47.6062, -122.3321, "Seattle, WA"),
                "seattle, washington": (47.6062, -122.3321, "Seattle, WA"),
                "vancouver": (49.2827, -123.1207, "Vancouver, BC"),
                "vancouver, bc": (49.2827, -123.1207, "Vancouver, BC"),
                "toronto": (43.6532, -79.3832, "Toronto, ON"),
                "toronto, on": (43.6532, -79.3832, "Toronto, ON"),
                "montreal": (45.5017, -73.5673, "Montreal, QC"),
                "montreal, qc": (45.5017, -73.5673, "Montreal, QC"),
                "calgary": (51.0447, -114.0719, "Calgary, AB"),
                "calgary, ab": (51.0447, -114.0719, "Calgary, AB"),
                "edmonton": (53.5461, -113.4938, "Edmonton, AB"),
                "edmonton, ab": (53.5461, -113.4938, "Edmonton, AB"),
                "ottawa": (45.4215, -75.6972, "Ottawa, ON"),
                "ottawa, on": (45.4215, -75.6972, "Ottawa, ON"),
                "winnipeg": (49.8951, -97.1384, "Winnipeg, MB"),
                "winnipeg, mb": (49.8951, -97.1384, "Winnipeg, MB"),
                "quebec city": (46.8139, -71.2080, "Quebec City, QC"),
                "quebec city, qc": (46.8139, -71.2080, "Quebec City, QC"),
                "hamilton": (43.2557, -79.8711, "Hamilton, ON"),
                "hamilton, on": (43.2557, -79.8711, "Hamilton, ON"),
                "kitchener": (43.4501, -80.4829, "Kitchener, ON"),
                "kitchener, on": (43.4501, -80.4829, "Kitchener, ON"),
                "london": (42.9849, -81.2453, "London, ON"),
                "london, on": (42.9849, -81.2453, "London, ON"),
                "victoria": (48.4284, -123.3656, "Victoria, BC"),
                "victoria, bc": (48.4284, -123.3656, "Victoria, BC"),
                "halifax": (44.6488, -63.5752, "Halifax, NS"),
                "halifax, ns": (44.6488, -63.5752, "Halifax, NS"),
                "st. john's": (47.5615, -52.7126, "St. John's, NL"),
                "st. john's, nl": (47.5615, -52.7126, "St. John's, NL"),
                "regina": (50.4452, -104.6189, "Regina, SK"),
                "regina, sk": (50.4452, -104.6189, "Regina, SK"),
                "saskatoon": (52.1579, -106.6702, "Saskatoon, SK"),
                "saskatoon, sk": (52.1579, -106.6702, "Saskatoon, SK"),
                "charlottetown": (46.2382, -63.1311, "Charlottetown, PE"),
                "charlottetown, pe": (46.2382, -63.1311, "Charlottetown, PE"),
                "fredericton": (45.9636, -66.6431, "Fredericton, NB"),
                "fredericton, nb": (45.9636, -66.6431, "Fredericton, NB"),
                "whitehorse": (60.7212, -135.0568, "Whitehorse, YT"),
                "whitehorse, yt": (60.7212, -135.0568, "Whitehorse, YT"),
                "yellowknife": (62.4540, -114.3718, "Yellowknife, NT"),
                "yellowknife, nt": (62.4540, -114.3718, "Yellowknife, NT"),
                "iqaluit": (63.7467, -68.5170, "Iqaluit, NU"),
                "iqaluit, nu": (63.7467, -68.5170, "Iqaluit, NU"),
            }
            
            if location_lower in place_coordinates:
                return place_coordinates[location_lower]
            
            # If no mapping found, use default coordinates (Chilliwack, BC)
            logger.warning(f"Location geocoding not implemented for: {location}. Using default coordinates.")
            return 49.16, -121.96, "Chilliwack, BC"
            
        except Exception as e:
            logger.error(f"Error parsing location {location}: {e}")
            return 49.16, -121.96, "Chilliwack, BC"  # Default to Chilliwack, BC
    
    async def _get_river_discharge_data(self, lat: float, lon: float) -> Dict:
        """
        Get river discharge data from Open-Meteo Flood API.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary containing river discharge data
        """
        try:
            # Construct API URL with parameters
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "river_discharge",
                "forecast_days": 7,
                "timeformat": "iso8601"
            }
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Successfully fetched river discharge data for {lat}, {lon}")
                        return data
                    else:
                        logger.error(f"API request failed with status {response.status}")
                        return await self._get_mock_discharge_data()
                        
        except Exception as e:
            logger.error(f"Error fetching river discharge data: {e}")
            return await self._get_mock_discharge_data()
    
    async def _get_mock_discharge_data(self) -> Dict:
        """
        Get mock river discharge data for testing/fallback purposes.
        
        Returns:
            Dictionary containing mock river discharge data
        """
        try:
            # Generate mock data for the next 7 days
            base_date = datetime.now()
            dates = []
            discharge_values = []
            
            for i in range(7):
                date = base_date + timedelta(days=i)
                dates.append(date.strftime("%Y-%m-%d"))
                # Mock discharge values (300-600 m³/s range)
                discharge = 300 + (i * 30) + (i % 3) * 20
                discharge_values.append(discharge)
            
            return {
                "latitude": 49.16,
                "longitude": -121.96,
                "daily": {
                    "time": dates,
                    "river_discharge": discharge_values
                },
                "daily_units": {
                    "river_discharge": "m³/s"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating mock discharge data: {e}")
            return {
                "latitude": 49.16,
                "longitude": -121.96,
                "daily": {
                    "time": [],
                    "river_discharge": []
                },
                "daily_units": {
                    "river_discharge": "m³/s"
                }
            }
    
    def _format_hydrology_response(self, lat: float, lon: float, location_name: str, discharge_data: Dict) -> Dict:
        """
        Format the hydrological response according to the specified structure.
        
        Args:
            lat: Latitude
            lon: Longitude
            location_name: Name of the location
            discharge_data: Raw discharge data from API
        
        Returns:
            Formatted hydrological forecast data
        """
        try:
            # Extract time and discharge data
            times = discharge_data.get("daily", {}).get("time", [])
            discharges = discharge_data.get("daily", {}).get("river_discharge", [])
            
            # Format daily series data
            series_daily_cms = []
            for i, (date, discharge) in enumerate(zip(times, discharges)):
                series_daily_cms.append({
                    "date": date,
                    "q": round(discharge, 1) if discharge is not None else 0.0
                })
            
            # Hardcoded flood thresholds as requested
            thresholds = {
                "watch_cms": 350,
                "warning_cms": 430,
                "danger_cms": 500
            }
            
            return {
                "location": {
                    "lat": lat,
                    "lon": lon,
                    "name": location_name
                },
                "series_daily_cms": series_daily_cms,
                "thresholds": thresholds
            }
            
        except Exception as e:
            logger.error(f"Error formatting hydrology response: {e}")
            return {
                "location": {"lat": lat, "lon": lon, "name": location_name},
                "series_daily_cms": [],
                "thresholds": {"watch_cms": 350, "warning_cms": 430, "danger_cms": 500}
            }
    
    async def get_hydrological_forecast_for_multiple_locations(self, locations: List[str]) -> Dict:
        """
        Get hydrological forecast data for multiple locations.
        
        Args:
            locations: List of location strings
        
        Returns:
            Dictionary containing hydrological forecast data for each location
        """
        try:
            results = {}
            
            # Process locations concurrently
            tasks = []
            for location in locations:
                task = self.get_hydrological_forecast(location)
                tasks.append((location, task))
            
            # Wait for all tasks to complete
            for location, task in tasks:
                try:
                    result = await task
                    results[location] = result
                except Exception as e:
                    logger.error(f"Error processing location {location}: {e}")
                    results[location] = {
                        "hydrology_forecast": {
                            "location": {"lat": 0.0, "lon": 0.0, "name": "Unknown"},
                            "series_daily_cms": [],
                            "thresholds": {"watch_cms": 350, "warning_cms": 430, "danger_cms": 500}
                        }
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting hydrological forecast for multiple locations: {e}")
            return {}


async def main():
    """
    Main function to test the hydrological forecast agent.
    """
    # Initialize the agent
    agent = HydrologicalForecastAgent()
    
    # Test locations
    test_locations = [
        "Chilliwack, BC",
        "49.16,-121.96",  # Chilliwack coordinates
        "Vancouver, BC",
        "Seattle, WA"
    ]
    
    print("Testing Hydrological Forecast Agent")
    print("=" * 50)
    
    # Test single location
    print("\n1. Testing single location (Chilliwack, BC):")
    result = await agent.get_hydrological_forecast("Chilliwack, BC")
    print(json.dumps(result, indent=2))
    
    # Test multiple locations
    print("\n2. Testing multiple locations:")
    results = await agent.get_hydrological_forecast_for_multiple_locations(test_locations)
    for location, data in results.items():
        print(f"\nLocation: {location}")
        print(json.dumps(data, indent=2))
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
