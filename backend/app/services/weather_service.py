import os
import requests
from typing import Dict, List, Optional, Tuple
import logging
from .msn_flood_alerts_agent import MSNFloodAlertsAgent
from .gov_flood_alert_agent import GovFloodAlertAgent
from .precipitation_forecast_agent import PrecipitationForecastAgent
from .hydrological_forecast_agent import HydrologicalForecastAgent

logger = logging.getLogger(__name__)

class WeatherService:
    """
    Service for fetching real-time weather data from Microsoft Weather API
    and providing flood risk assessments.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MICROSOFT_WEATHER_API_KEY")
        self.base_url = "https://atlas.microsoft.com/weather"
        # Initialize the weather alerts agent
        self.alerts_agent = MSNFloodAlertsAgent()
        # Initialize the flood alert agent
        self.flood_alert_agent = GovFloodAlertAgent()
        # Initialize the precipitation forecast agent
        self.precipitation_agent = PrecipitationForecastAgent()
        # Initialize the hydrological forecast agent
        self.hydrological_agent = HydrologicalForecastAgent()
        
    async def get_current_conditions(self, location: str) -> Dict:
        """Get current weather conditions for a location"""
        try:
            # TODO: Implement actual Microsoft Weather API integration
            # For now, return mock data
            return {
                "location": location,
                "temperature": 15.5,
                "humidity": 85,
                "precipitation": 12.3,
                "wind_speed": 25.2,
                "conditions": "Heavy Rain",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return {}
    
    async def assess_flood_risk(self, weather_data: Dict) -> str:
        """Assess flood risk based on weather data"""
        try:
            # Simple risk assessment logic
            precipitation = weather_data.get("precipitation", 0)
            humidity = weather_data.get("humidity", 0)
            
            if precipitation > 20 and humidity > 80:
                return "High"
            elif precipitation > 10 and humidity > 70:
                return "Moderate"
            else:
                return "Low"
        except Exception as e:
            logger.error(f"Error assessing flood risk: {e}")
            return "Unknown"
    
    async def get_recommendations(self, risk_level: str) -> List[str]:
        """Get recommendations based on flood risk level"""
        recommendations = {
            "High": [
                "Evacuate immediately if in flood-prone areas",
                "Avoid all travel unless absolutely necessary",
                "Stay tuned to emergency broadcasts"
            ],
            "Moderate": [
                "Prepare emergency supplies",
                "Monitor local conditions closely",
                "Be ready to evacuate if conditions worsen"
            ],
            "Low": [
                "Stay informed about weather updates",
                "Prepare basic emergency kit",
                "Know your evacuation routes"
            ]
        }
        return recommendations.get(risk_level, ["Stay informed and prepared"])
    
    async def get_weather_alerts(self, locations: List[Tuple[float, float]]) -> List[Dict]:
        """
        Get weather alerts for specified locations using the MSN Weather API.
        
        Args:
            locations: List of (latitude, longitude) tuples
        
        Returns:
            List of dictionaries containing weather alerts for each location
        """
        try:
            return self.alerts_agent.get_weather_alerts(locations)
        except Exception as e:
            logger.error(f"Error fetching weather alerts: {e}")
            return []
    
    async def get_flood_alerts(self, locations: List[Tuple[float, float]]) -> List[Dict]:
        """
        Get flood-related weather alerts for specified locations.
        
        Args:
            locations: List of (latitude, longitude) tuples
        
        Returns:
            List of dictionaries containing flood-related alerts
        """
        try:
            return self.alerts_agent.get_flood_related_alerts(locations)
        except Exception as e:
            logger.error(f"Error fetching flood alerts: {e}")
            return []
    
    async def get_government_flood_alerts(self, 
                                        us_locations: List[Tuple[str, str]] = None,
                                        canada_locations: List[Tuple[str, str]] = None) -> Dict:
        """
        Get flood alerts from government sources (US and Canada).
        
        Args:
            us_locations: List of (state, county) tuples for US
            canada_locations: List of (province, city) tuples for Canada
        
        Returns:
            Dictionary containing government flood alerts
        """
        try:
            return await self.flood_alert_agent.get_comprehensive_flood_alerts(
                us_locations=us_locations,
                canada_locations=canada_locations
            )
        except Exception as e:
            logger.error(f"Error getting government flood alerts: {e}")
            return {"error": str(e)}
    
    async def get_comprehensive_weather_data(self, locations: List[Tuple[float, float]]) -> Dict:
        """
        Get comprehensive weather data including current conditions and alerts.
        
        Args:
            locations: List of (latitude, longitude) tuples
        
        Returns:
            Dictionary containing weather data, alerts, and flood risk assessment
        """
        try:
            # Get weather alerts
            alerts_data = await self.get_weather_alerts(locations)
            flood_alerts = await self.get_flood_alerts(locations)
            
            # Get current conditions (mock data for now)
            current_conditions = []
            for location in locations:
                conditions = await self.get_current_conditions(f"{location[0]},{location[1]}")
                current_conditions.append(conditions)
            
            # Assess flood risk for each location
            risk_assessments = []
            for i, conditions in enumerate(current_conditions):
                risk_level = await self.assess_flood_risk(conditions)
                recommendations = await self.get_recommendations(risk_level)
                risk_assessments.append({
                    "location": locations[i],
                    "risk_level": risk_level,
                    "recommendations": recommendations
                })
            
            return {
                "locations": locations,
                "current_conditions": current_conditions,
                "weather_alerts": alerts_data,
                "flood_alerts": flood_alerts,
                "risk_assessments": risk_assessments,
                "summary": {
                    "total_locations": len(locations),
                    "locations_with_alerts": len([a for a in alerts_data if a.get("alerts")]),
                    "locations_with_flood_alerts": len(flood_alerts),
                    "high_risk_locations": len([r for r in risk_assessments if r["risk_level"] == "High"])
                }
            }
        except Exception as e:
            logger.error(f"Error getting comprehensive weather data: {e}")
            return {
                "error": str(e),
                "locations": locations,
                "current_conditions": [],
                "weather_alerts": [],
                "flood_alerts": [],
                "risk_assessments": []
            }
    
    async def get_precipitation_forecast(self, location: str) -> Dict:
        """
        Get precipitation forecast data for a given location.
        
        Args:
            location: Location string (e.g., "Seattle, WA" or "47.6062,-122.3321")
        
        Returns:
            Dictionary containing precipitation data in the specified format
        """
        try:
            return await self.precipitation_agent.get_precipitation_forecast(location)
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
    
    async def get_precipitation_for_multiple_locations(self, locations: List[str]) -> Dict:
        """
        Get precipitation forecast data for multiple locations.
        
        Args:
            locations: List of location strings
        
        Returns:
            Dictionary containing precipitation data for each location
        """
        try:
            return await self.precipitation_agent.get_precipitation_for_multiple_locations(locations)
        except Exception as e:
            logger.error(f"Error getting precipitation for multiple locations: {e}")
            return {}
    
    async def get_hydrological_forecast(self, location: str) -> Dict:
        """
        Get hydrological forecast data for a given location.
        
        Args:
            location: Location string (e.g., "Chilliwack, BC" or "49.16,-121.96")
        
        Returns:
            Dictionary containing hydrological forecast data in the specified format
        """
        try:
            return await self.hydrological_agent.get_hydrological_forecast(location)
        except Exception as e:
            logger.error(f"Error getting hydrological forecast for {location}: {e}")
            return {
                "hydrology_forecast": {
                    "location": {"lat": 0.0, "lon": 0.0, "name": "Unknown"},
                    "series_daily_cms": [],
                    "thresholds": {"watch_cms": 350, "warning_cms": 430, "danger_cms": 500}
                }
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
            return await self.hydrological_agent.get_hydrological_forecast_for_multiple_locations(locations)
        except Exception as e:
            logger.error(f"Error getting hydrological forecast for multiple locations: {e}")
            return {}
