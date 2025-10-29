import aiohttp
import asyncio
import logging
from urllib import parse
from typing import List, Dict, Optional, Tuple
import json

logger = logging.getLogger(__name__)

# MSN Weather API configuration
MSN_WEATHER_API_KEY = "j5i4gDqHL6nGYwx5wi5kRhXjtf2c5qgFX9fzfk0TOo"
MSN_WEATHER_BASE_URL = "https://api.msn.com/weather"

class MSNFloodAlertsAgent:
    """
    Agent for fetching weather alerts from MSN Weather API.
    Handles both current weather conditions and weather alerts for flood prediction.
    """
    
    def __init__(self, api_key: str = MSN_WEATHER_API_KEY):
        self.api_key = api_key
        self.base_url = MSN_WEATHER_BASE_URL
    
    async def get_weather_api_point_data_async(self, session, params: Dict, req_type: str = "current") -> str:
        """
        Get weather data for a single point asynchronously.
        
        Args:
            session: aiohttp ClientSession
            params: Dictionary of API parameters
            req_type: Type of weather data request ("current", "forecast", etc.)
        
        Returns:
            Raw weather response as string
        """
        encoded_params = parse.urlencode(params)
        url = f"{self.base_url}/{req_type}?{encoded_params}"
        
        try:
            async with session.get(url) as resp:
                weather_response = await resp.text()
                return weather_response
        except Exception as e:
            logger.error(f"Error fetching weather data for point: {e}")
            return None
    
    async def get_weather_api_data_async(self, locations: List[Tuple[float, float]], 
                                       req_type: str = "current", 
                                       additional_params: Optional[Dict] = None) -> List[str]:
        """
        Get weather data for multiple locations asynchronously.
        
        Args:
            locations: List of (latitude, longitude) tuples
            req_type: Type of weather data request
            additional_params: Additional parameters to include in the request
        
        Returns:
            List of weather response strings
        """
        try:
            async with aiohttp.ClientSession() as session:
                tasks = []
                for location in locations:
                    params = {
                        "apikey": self.api_key,
                        "lat": location[0],
                        "lon": location[1]
                    }
                    if additional_params is not None:
                        params = {**params, **additional_params}
                    
                    tasks.append(
                        asyncio.ensure_future(
                            self.get_weather_api_point_data_async(session, params, req_type=req_type)
                        )
                    )
                
                weather_responses = await asyncio.gather(*tasks)
                return weather_responses
        except aiohttp.ClientConnectionError as e:
            logger.error(f"Connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in get_weather_api_data_async: {e}")
            return None
    
    def get_weather_api_data(self, locations: List[Tuple[float, float]], 
                           req_type: str = "current", 
                           additional_params: Optional[Dict] = None) -> List[str]:
        """
        Synchronous wrapper for getting weather data.
        Handles event loop creation for environments like Azure Functions.
        
        Args:
            locations: List of (latitude, longitude) tuples
            req_type: Type of weather data request
            additional_params: Additional parameters to include in the request
        
        Returns:
            List of weather response strings
        """
        try:
            loop = asyncio.get_event_loop()
            future = asyncio.ensure_future(
                self.get_weather_api_data_async(locations, req_type=req_type, additional_params=additional_params)
            )
            weather_data = loop.run_until_complete(future)
        except RuntimeError:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                future = asyncio.ensure_future(
                    self.get_weather_api_data_async(locations, req_type=req_type, additional_params=additional_params)
                )
                weather_data = loop.run_until_complete(future)
            except RuntimeError:
                raise
            except aiohttp.ClientConnectionError:
                logger.info("Connection was refused. Returning null.")
                weather_data = None
            except ValueError:
                logger.info("Too many connection sockets. Returning null.")
                weather_data = None
        except aiohttp.ClientConnectionError:
            logger.info("Connection was refused. Returning null.")
            weather_data = None
        except ValueError:
            logger.info("Too many connection sockets. Returning null.")
            weather_data = None
        
        return weather_data
    
    def _convert_to_standardized_format(self, raw_alerts: List[Dict], weather_info: Dict) -> List[Dict]:
        """
        Convert MSN Weather API alerts to standardized format.
        
        Args:
            raw_alerts: List of raw alert dictionaries from MSN API
            weather_info: Weather information containing location data
        
        Returns:
            List of standardized alert dictionaries
        """
        standardized_alerts = []
        
        for alert in raw_alerts:
            if not isinstance(alert, dict):
                continue
                
            # Extract location information
            location_name = "Unknown"
            if "source" in weather_info and "location" in weather_info["source"]:
                location_data = weather_info["source"]["location"]
                if "Name" in location_data:
                    location_name = location_data["Name"]
                elif "City" in location_data:
                    location_name = location_data["City"]
            
            # Create standardized alert format
            standardized_alert = {
                "provenance": "aggregator",
                "source_name": "MSN Weather",
                "source_url": "https://api.msn.com/weather",
                "origin_agency_hint": alert.get("origin", "MSN Weather"),
                "event": alert.get("title", alert.get("event", "Weather Alert")),
                "severity": alert.get("severity", alert.get("level", "Unknown")),
                "urgency": alert.get("urgency", "Unknown"),
                "certainty": alert.get("certainty", "Unknown"),
                "effective": alert.get("start", alert.get("effective")),
                "expires": alert.get("end", alert.get("expires")),
                "area": {
                    "type": "city",
                    "name": location_name
                },
                "cap_id": alert.get("capId"),
                "summary": alert.get("safetyGuide", alert.get("shortCap", alert.get("summary", ""))),
                "hash": self._generate_alert_hash(alert, location_name)
            }
            
            # Remove None values
            standardized_alert = {k: v for k, v in standardized_alert.items() if v is not None}
            
            standardized_alerts.append(standardized_alert)
        
        return standardized_alerts
    
    def _generate_alert_hash(self, alert: Dict, location_name: str) -> str:
        """
        Generate a unique hash for the alert.
        
        Args:
            alert: Alert dictionary
            location_name: Name of the location
        
        Returns:
            Unique hash string
        """
        import hashlib
        import datetime
        
        # Create a unique identifier from alert data
        hash_data = f"{alert.get('title', '')}_{location_name}_{alert.get('start', '')}_{alert.get('end', '')}"
        return hashlib.md5(hash_data.encode()).hexdigest()[:32]
    
    def get_weather_alerts(self, locations: List[Tuple[float, float]]) -> List[Dict]:
        """
        Get weather alerts for specified locations.
        
        Args:
            locations: List of (latitude, longitude) tuples
        
        Returns:
            List of dictionaries containing weather alerts for each location
        """
        try:
            weather_responses = self.get_weather_api_data(locations, req_type="current")
            
            if not weather_responses:
                logger.warning("No weather responses received")
                return []
            
            alerts_data = []
            for i, response in enumerate(weather_responses):
                if response is None:
                    alerts_data.append({
                        "location": locations[i],
                        "alerts": [],
                        "error": "Failed to fetch weather data"
                    })
                    continue
                
                try:
                    # Parse the JSON response
                    weather_data = json.loads(response)
                    
                    # Extract alerts from the response and convert to standardized format
                    # The response structure is: {"@odata.context": "...", "value": [{"responses": [{"weather": [{"alerts": [...]}]}]}]}
                    alerts = []
                    if "value" in weather_data and len(weather_data["value"]) > 0:
                        value_data = weather_data["value"][0]
                        if "responses" in value_data and len(value_data["responses"]) > 0:
                            weather_info = value_data["responses"][0]
                            if "weather" in weather_info and len(weather_info["weather"]) > 0:
                                weather = weather_info["weather"][0]
                                if "alerts" in weather:
                                    raw_alerts = weather["alerts"]
                                    # Convert to standardized format
                                    alerts = self._convert_to_standardized_format(raw_alerts, weather_info)
                    
                    alerts_data.append({
                        "location": locations[i],
                        "alerts": alerts,
                        "raw_response": weather_data
                    })
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON response for location {locations[i]}: {e}")
                    alerts_data.append({
                        "location": locations[i],
                        "alerts": [],
                        "error": f"JSON parsing error: {str(e)}"
                    })
                except Exception as e:
                    logger.error(f"Error processing weather data for location {locations[i]}: {e}")
                    alerts_data.append({
                        "location": locations[i],
                        "alerts": [],
                        "error": f"Processing error: {str(e)}"
                    })
            
            return alerts_data
            
        except Exception as e:
            logger.error(f"Error in get_weather_alerts: {e}")
            return []
    
    def get_flood_related_alerts(self, locations: List[Tuple[float, float]]) -> List[Dict]:
        """
        Get flood-related weather alerts for specified locations.
        Filters alerts to only include those related to flooding, heavy rain, etc.
        
        Args:
            locations: List of (latitude, longitude) tuples
        
        Returns:
            List of dictionaries containing flood-related alerts
        """
        all_alerts = self.get_weather_alerts(locations)
        flood_keywords = [
            "flood", "flooding", "flash flood", "river flood", "coastal flood",
            "heavy rain", "torrential", "downpour", "deluge", "inundation",
            "storm surge", "high water", "overflow", "breach", "levee",
            "rain", "precipitation", "storm", "severe weather", "warning"
        ]
        
        flood_alerts = []
        for alert_data in all_alerts:
            if "alerts" not in alert_data:
                continue
                
            location_flood_alerts = []
            for alert in alert_data["alerts"]:
                # Check if alert contains flood-related keywords using standardized format
                alert_text = ""
                if isinstance(alert, dict):
                    # Combine key text fields from the standardized alert format
                    text_fields = [
                        alert.get("event", ""),
                        alert.get("summary", ""),
                        alert.get("origin_agency_hint", "")
                    ]
                    alert_text = " ".join([str(field) for field in text_fields if field]).lower()
                elif isinstance(alert, str):
                    alert_text = alert.lower()
                
                # Check for flood-related keywords
                if any(keyword in alert_text for keyword in flood_keywords):
                    location_flood_alerts.append(alert)
            
            if location_flood_alerts:
                flood_alerts.append({
                    "location": alert_data["location"],
                    "flood_alerts": location_flood_alerts,
                    "total_alerts": len(alert_data["alerts"])
                })
        
        return flood_alerts
    
    
    def extract_weather_info(self, weather_data: Dict) -> Dict:
        """
        Extract comprehensive weather information from the MSN Weather API response.
        
        Args:
            weather_data: Parsed weather data from the API
        
        Returns:
            Dictionary containing extracted weather information
        """
        try:
            if "value" not in weather_data or len(weather_data["value"]) == 0:
                return {}
            
            value_data = weather_data["value"][0]
            if "responses" not in value_data or len(value_data["responses"]) == 0:
                return {}
            
            weather_info = value_data["responses"][0]
            if "weather" not in weather_info or len(weather_info["weather"]) == 0:
                return {}
            
            weather = weather_info["weather"][0]
            
            # Extract current conditions
            current = weather.get("current", {})
            nowcasting = weather.get("nowcasting", {})
            
            extracted_info = {
                "alerts": weather.get("alerts", []),
                "current_conditions": {
                    "temperature": current.get("temp"),
                    "feels_like": current.get("feels"),
                    "humidity": current.get("rh"),
                    "dew_point": current.get("dewPt"),
                    "pressure": current.get("baro"),
                    "visibility": current.get("vis"),
                    "uv_index": current.get("uv"),
                    "uv_description": current.get("uvDesc"),
                    "condition": current.get("cap"),
                    "condition_abbr": current.get("capAbbr"),
                    "wind_speed": current.get("windSpd"),
                    "wind_direction": current.get("windDir"),
                    "wind_gust": current.get("windGust"),
                    "daytime": current.get("daytime") == "d"
                },
                "precipitation": {
                    "precipitation_forecast": nowcasting.get("precipitation", []),
                    "precipitation_rate": nowcasting.get("precipitationRate", []),
                    "precipitation_accumulation": nowcasting.get("precipitationAccumulation", []),
                    "summary": nowcasting.get("summary"),
                    "short_summary": nowcasting.get("shortSummary")
                },
                "air_quality": {
                    "aqi": current.get("aqi"),
                    "aqi_severity": current.get("aqiSeverity"),
                    "aq_level": current.get("aqLevel"),
                    "primary_pollutant": current.get("primaryPollutant")
                },
                "location_info": {
                    "name": weather_info.get("source", {}).get("location", {}).get("Name"),
                    "state": weather_info.get("source", {}).get("location", {}).get("StateCode"),
                    "country": weather_info.get("source", {}).get("location", {}).get("CountryCode"),
                    "timezone": weather_info.get("source", {}).get("location", {}).get("TimezoneName"),
                    "coordinates": weather_info.get("source", {}).get("coordinates", {})
                },
                "units": value_data.get("units", {}),
                "provider": weather.get("provider", {}),
                "timestamp": current.get("created")
            }
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error extracting weather info: {e}")
            return {}


# Example usage and testing
def main():
    """
    Main function to test the MSNFloodAlertsAgent.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize the weather alerts agent
    agent = MSNFloodAlertsAgent()
    
    # Test locations (latitude, longitude)
    test_locations = [
        (40.7128, -74.0060),  # New York City
        (34.0522, -118.2437),  # Los Angeles
        (41.8781, -87.6298),  # Chicago
    ]
    
    print("🌤️  MSN Flood Alerts Agent Test")
    print("=" * 50)
    
    # Test 1: Get all weather alerts
    print("\n1. Fetching all weather alerts...")
    all_alerts = agent.get_weather_alerts(test_locations)
    
    for alert_data in all_alerts:
        location = alert_data["location"]
        alerts = alert_data.get("alerts", [])
        
        print(f"\n📍 Location: {location[0]:.4f}, {location[1]:.4f}")
        
        if "error" in alert_data:
            print(f"❌ Error: {alert_data['error']}")
        elif not alerts:
            print("✅ No active weather alerts")
        else:
            print(f"⚠️  {len(alerts)} active alert(s):")
            for i, alert in enumerate(alerts, 1):
                print(f"  Alert {i}:")
                print(json.dumps(alert, indent=4))
    
    # Test 2: Get flood-related alerts only
    print("\n2. Fetching flood-related alerts...")
    flood_alerts = agent.get_flood_related_alerts(test_locations)
    
    if flood_alerts:
        print(f"Found {len(flood_alerts)} locations with flood-related alerts:")
        for alert_data in flood_alerts:
            location = alert_data["location"]
            flood_alerts_list = alert_data["flood_alerts"]
            print(f"\n📍 Location: {location[0]:.4f}, {location[1]:.4f}")
            print(f"🌊 {len(flood_alerts_list)} flood-related alert(s):")
            for i, alert in enumerate(flood_alerts_list, 1):
                print(f"  Alert {i}:")
                print(json.dumps(alert, indent=4))
    else:
        print("✅ No flood-related alerts found for the test locations.")
    
    # Test 3: Test with a single location
    print("\n3. Testing with a single location...")
    single_location = [(40.97067001, -101.28075954)]  # Nebraska location
    single_alerts = agent.get_weather_alerts(single_location)
    
    for alert_data in single_alerts:
        location = alert_data["location"]
        alerts = alert_data.get("alerts", [])
        
        print(f"\n📍 Location: {location[0]:.4f}, {location[1]:.4f}")
        
        if "error" in alert_data:
            print(f"❌ Error: {alert_data['error']}")
        elif not alerts:
            print("✅ No active weather alerts")
        else:
            print(f"⚠️  {len(alerts)} active alert(s):")
            for i, alert in enumerate(alerts, 1):
                print(f"  Alert {i}:")
                print(json.dumps(alert, indent=4))


if __name__ == "__main__":
    main()
