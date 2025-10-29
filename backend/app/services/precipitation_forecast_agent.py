import aiohttp
import asyncio
import logging
from urllib import parse
from typing import Dict, Optional, Tuple, List
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# MSN Weather API configuration
MSN_WEATHER_API_KEY = "j5i4gDqHL6nGYwx5wi5kRhXjtf2c5qgFX9fzfk0TOo"
MSN_WEATHER_BASE_URL = "https://api.msn.com/weather"

class PrecipitationForecastAgent:
    """
    Agent for fetching precipitation forecast data from MSN Weather API.
    Provides precipitation data in the specified format for flood prediction.
    """
    
    def __init__(self, api_key: str = MSN_WEATHER_API_KEY):
        self.api_key = api_key
        self.base_url = MSN_WEATHER_BASE_URL
    
    async def get_precipitation_forecast(self, location: str) -> Dict:
        """
        Get precipitation forecast data for a given location.
        
        Args:
            location: Location string (e.g., "Seattle, WA" or "47.6062,-122.3321")
        
        Returns:
            Dictionary containing precipitation data in the specified format:
            {
                "precip": {
                    "past72h_mm": float,
                    "max6h_past72h_mm": float,
                    "qpf_6h_max_next72h_mm": float,
                    "qpf_24h_max_next72h_mm": float,
                    "qpf_sum_next7d_mm": float
                }
            }
        """
        try:
            # Parse location to get coordinates
            lat, lon = await self._parse_location(location)
            
            # Get historical and forecast data
            historical_data = await self._get_historical_precipitation(lat, lon)
            forecast_data = await self._get_forecast_precipitation(lat, lon)
            
            # Calculate precipitation metrics
            precip_data = self._calculate_precipitation_metrics(historical_data, forecast_data)
            
            return {
                "precip": precip_data
            }
            
        except Exception as e:
            logger.error(f"Error getting precipitation forecast for {location}: {e}")
            return {
                "precip": {
                    "past72h_mm": 0.0,
                    "max6h_past72h_mm": 0.0,
                    "qpf_6h_max_next72h_mm": 0.0,
                    "qpf_24h_max_next72h_mm": 0.0,
                    "qpf_sum_next7d_mm": 0.0
                }
            }
    
    async def _parse_location(self, location: str) -> Tuple[float, float]:
        """
        Parse location string to get latitude and longitude.
        
        Args:
            location: Location string (coordinates or place name)
        
        Returns:
            Tuple of (latitude, longitude)
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
                            return lat, lon
                    except ValueError:
                        # If conversion fails, treat as place name with comma
                        pass
            
            # For place names, we'll use a simple geocoding approach
            # In a real implementation, you'd use a proper geocoding service
            # For now, return mock coordinates based on common place names
            location_lower = location.lower().strip()
            
            # Common place name mappings
            place_coordinates = {
                "seattle": (47.6062, -122.3321),
                "seattle, wa": (47.6062, -122.3321),
                "seattle, washington": (47.6062, -122.3321),
                "new york": (40.7128, -74.0060),
                "new york, ny": (40.7128, -74.0060),
                "new york, new york": (40.7128, -74.0060),
                "los angeles": (34.0522, -118.2437),
                "los angeles, ca": (34.0522, -118.2437),
                "chicago": (41.8781, -87.6298),
                "chicago, il": (41.8781, -87.6298),
                "houston": (29.7604, -95.3698),
                "houston, tx": (29.7604, -95.3698),
                "phoenix": (33.4484, -112.0740),
                "phoenix, az": (33.4484, -112.0740),
                "philadelphia": (39.9526, -75.1652),
                "philadelphia, pa": (39.9526, -75.1652),
                "san antonio": (29.4241, -98.4936),
                "san antonio, tx": (29.4241, -98.4936),
                "san diego": (32.7157, -117.1611),
                "san diego, ca": (32.7157, -117.1611),
                "dallas": (32.7767, -96.7970),
                "dallas, tx": (32.7767, -96.7970),
                "san jose": (37.3382, -121.8863),
                "san jose, ca": (37.3382, -121.8863),
            }
            
            if location_lower in place_coordinates:
                return place_coordinates[location_lower]
            
            # If no mapping found, use default coordinates
            logger.warning(f"Location geocoding not implemented for: {location}. Using default coordinates.")
            return 47.6062, -122.3321  # Seattle coordinates as default
            
        except Exception as e:
            logger.error(f"Error parsing location {location}: {e}")
            return 47.6062, -122.3321  # Default to Seattle
    
    async def _get_historical_precipitation(self, lat: float, lon: float) -> List[Dict]:
        """
        Get historical precipitation data for the past 72 hours.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            List of historical precipitation data points
        """
        try:
            # For now, return mock historical data
            # In a real implementation, you'd call the MSN Weather API for historical data
            logger.info(f"Fetching historical precipitation for {lat}, {lon}")
            
            # Mock historical data (past 72 hours)
            historical_data = []
            base_time = datetime.now() - timedelta(hours=72)
            
            for i in range(72):  # 72 hours of data
                timestamp = base_time + timedelta(hours=i)
                # Mock precipitation values (0-5mm per hour)
                precipitation = max(0, (i % 24 - 12) * 0.5) if i % 24 > 12 else 0
                historical_data.append({
                    "timestamp": timestamp.isoformat(),
                    "precipitation_mm": precipitation
                })
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error getting historical precipitation: {e}")
            return []
    
    async def _get_forecast_precipitation(self, lat: float, lon: float) -> List[Dict]:
        """
        Get forecast precipitation data for the next 7 days.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            List of forecast precipitation data points
        """
        try:
            # For now, return mock forecast data
            # In a real implementation, you'd call the MSN Weather API for forecast data
            logger.info(f"Fetching forecast precipitation for {lat}, {lon}")
            
            # Mock forecast data (next 7 days)
            forecast_data = []
            base_time = datetime.now()
            
            for i in range(168):  # 7 days * 24 hours
                timestamp = base_time + timedelta(hours=i)
                # Mock precipitation values (0-8mm per hour)
                precipitation = max(0, (i % 48 - 24) * 0.3) if i % 48 > 24 else 0
                forecast_data.append({
                    "timestamp": timestamp.isoformat(),
                    "precipitation_mm": precipitation
                })
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error getting forecast precipitation: {e}")
            return []
    
    def _calculate_precipitation_metrics(self, historical_data: List[Dict], forecast_data: List[Dict]) -> Dict:
        """
        Calculate precipitation metrics from historical and forecast data.
        
        Args:
            historical_data: List of historical precipitation data points
            forecast_data: List of forecast precipitation data points
        
        Returns:
            Dictionary containing calculated precipitation metrics
        """
        try:
            # Calculate past 72 hours total precipitation
            past72h_mm = sum(point.get("precipitation_mm", 0) for point in historical_data)
            
            # Calculate maximum 6-hour precipitation in past 72 hours
            max6h_past72h_mm = 0
            for i in range(len(historical_data) - 5):
                six_hour_total = sum(
                    historical_data[j].get("precipitation_mm", 0) 
                    for j in range(i, i + 6)
                )
                max6h_past72h_mm = max(max6h_past72h_mm, six_hour_total)
            
            # Calculate QPF (Quantitative Precipitation Forecast) for next 72 hours
            next72h_data = forecast_data[:72]  # First 72 hours of forecast
            
            # Maximum 6-hour QPF in next 72 hours
            qpf_6h_max_next72h_mm = 0
            for i in range(len(next72h_data) - 5):
                six_hour_total = sum(
                    next72h_data[j].get("precipitation_mm", 0) 
                    for j in range(i, i + 6)
                )
                qpf_6h_max_next72h_mm = max(qpf_6h_max_next72h_mm, six_hour_total)
            
            # Maximum 24-hour QPF in next 72 hours
            qpf_24h_max_next72h_mm = 0
            for i in range(len(next72h_data) - 23):
                twenty_four_hour_total = sum(
                    next72h_data[j].get("precipitation_mm", 0) 
                    for j in range(i, i + 24)
                )
                qpf_24h_max_next72h_mm = max(qpf_24h_max_next72h_mm, twenty_four_hour_total)
            
            # Sum of QPF for next 7 days
            qpf_sum_next7d_mm = sum(point.get("precipitation_mm", 0) for point in forecast_data)
            
            return {
                "past72h_mm": round(past72h_mm, 1),
                "max6h_past72h_mm": round(max6h_past72h_mm, 1),
                "qpf_6h_max_next72h_mm": round(qpf_6h_max_next72h_mm, 1),
                "qpf_24h_max_next72h_mm": round(qpf_24h_max_next72h_mm, 1),
                "qpf_sum_next7d_mm": round(qpf_sum_next7d_mm, 1)
            }
            
        except Exception as e:
            logger.error(f"Error calculating precipitation metrics: {e}")
            return {
                "past72h_mm": 0.0,
                "max6h_past72h_mm": 0.0,
                "qpf_6h_max_next72h_mm": 0.0,
                "qpf_24h_max_next72h_mm": 0.0,
                "qpf_sum_next7d_mm": 0.0
            }
    
    async def get_precipitation_for_multiple_locations(self, locations: List[str]) -> Dict:
        """
        Get precipitation forecast data for multiple locations.
        
        Args:
            locations: List of location strings
        
        Returns:
            Dictionary containing precipitation data for each location
        """
        try:
            results = {}
            
            # Process locations concurrently
            tasks = []
            for location in locations:
                task = self.get_precipitation_forecast(location)
                tasks.append((location, task))
            
            # Wait for all tasks to complete
            for location, task in tasks:
                try:
                    result = await task
                    results[location] = result
                except Exception as e:
                    logger.error(f"Error processing location {location}: {e}")
                    results[location] = {
                        "precip": {
                            "past72h_mm": 0.0,
                            "max6h_past72h_mm": 0.0,
                            "qpf_6h_max_next72h_mm": 0.0,
                            "qpf_24h_max_next72h_mm": 0.0,
                            "qpf_sum_next7d_mm": 0.0
                        }
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting precipitation for multiple locations: {e}")
            return {}


async def main():
    """
    Main function to test the precipitation forecast agent.
    """
    # Initialize the agent
    agent = PrecipitationForecastAgent()
    
    # Test locations
    test_locations = [
        "Seattle, WA",
        "47.6062,-122.3321",  # Seattle coordinates
        "New York, NY",
        "40.7128,-74.0060"    # New York coordinates
    ]
    
    print("Testing Precipitation Forecast Agent")
    print("=" * 50)
    
    # Test single location
    print("\n1. Testing single location (Seattle, WA):")
    result = await agent.get_precipitation_forecast("Seattle, WA")
    print(json.dumps(result, indent=2))
    
    # Test multiple locations
    print("\n2. Testing multiple locations:")
    results = await agent.get_precipitation_for_multiple_locations(test_locations)
    for location, data in results.items():
        print(f"\nLocation: {location}")
        print(json.dumps(data, indent=2))
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
