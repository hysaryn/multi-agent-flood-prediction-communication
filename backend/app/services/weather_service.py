import os
import requests
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class WeatherService:
    """
    Service for fetching real-time weather data from Microsoft Weather API
    and providing flood risk assessments.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MICROSOFT_WEATHER_API_KEY")
        self.base_url = "https://atlas.microsoft.com/weather"
        
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
