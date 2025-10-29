"""
Integration example showing how to use all agents together to generate a comprehensive flood risk overview.
This demonstrates the complete workflow from individual data sources to final risk assessment.
"""

import asyncio
import json
import logging
from typing import Dict, List, Tuple

# Import all the agents
try:
    # Try relative imports first (when used as a module)
    from .hydrological_forecast_agent import HydrologicalForecastAgent
    from .msn_flood_alerts_agent import MSNFloodAlertsAgent
    from .precipitation_forecast_agent import PrecipitationForecastAgent
    from .risk_overview_agent import RiskOverviewAgent
except ImportError:
    # Fall back to absolute imports (when run directly)
    from hydrological_forecast_agent import HydrologicalForecastAgent
    from msn_flood_alerts_agent import MSNFloodAlertsAgent
    from precipitation_forecast_agent import PrecipitationForecastAgent
    from risk_overview_agent import RiskOverviewAgent

logger = logging.getLogger(__name__)

class FloodRiskIntegration:
    """
    Integration class that coordinates all flood prediction agents to provide comprehensive risk assessment.
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the integration with all required agents.
        
        Args:
            openai_api_key: OpenAI API key for AI-powered risk assessment (optional)
        """
        self.hydrology_agent = HydrologicalForecastAgent()
        self.alerts_agent = MSNFloodAlertsAgent()
        self.precip_agent = PrecipitationForecastAgent()
        self.risk_agent = RiskOverviewAgent(openai_api_key)
    
    async def get_comprehensive_risk_assessment(self, location: str) -> Dict:
        """
        Get comprehensive flood risk assessment by running all agents and generating final risk overview.
        
        Args:
            location: Location string (e.g., "Chilliwack, BC" or "49.16,-121.96")
        
        Returns:
            Dictionary containing complete risk assessment with all data sources and final risk overview
        """
        try:
            # Parse location to get coordinates
            lat, lon = await self._parse_location(location)
            
            # Run all agents concurrently
            tasks = [
                self.hydrology_agent.get_hydrological_forecast(location),
                self._get_alerts_data(lat, lon),
                self.precip_agent.get_precipitation_forecast(location)
            ]
            
            # Wait for all tasks to complete
            hydrology_result, alerts_result, precip_result = await asyncio.gather(*tasks)
            
            # Extract the actual data from results
            hydrology_data = hydrology_result.get("hydrology_forecast", {})
            alerts_data = alerts_result.get("alerts", []) if alerts_result else []
            precip_data = precip_result.get("precip", {})
            
            # Prepare input data for risk overview agent
            input_data = {
                "location": {
                    "name": location,
                    "lat": lat,
                    "lon": lon
                },
                "time_utc": self._get_current_utc_time(),
                "signals": {
                    "alerts": alerts_data,
                    "hydrology_forecast": hydrology_data,
                    "precip": precip_data
                }
            }
            
            # Generate final risk overview
            risk_overview = self.risk_agent.generate_risk_overview(input_data)
            
            # Combine all results
            comprehensive_result = {
                "location": location,
                "timestamp": self._get_current_utc_time(),
                "data_sources": {
                    "hydrology": hydrology_result,
                    "alerts": alerts_result,
                    "precipitation": precip_result
                },
                "risk_assessment": risk_overview
            }
            
            return comprehensive_result
            
        except Exception as e:
            logger.error(f"Error getting comprehensive risk assessment for {location}: {e}")
            return {
                "location": location,
                "timestamp": self._get_current_utc_time(),
                "error": str(e),
                "data_sources": {},
                "risk_assessment": {
                    "risk_level": "unknown",
                    "confidence_0_1": 0.0,
                    "key_drivers": ["Analysis failed"],
                    "rationale": f"Error occurred during analysis: {str(e)}",
                    "sources_used": [],
                    "notes": "Analysis failed due to error"
                }
            }
    
    async def _get_alerts_data(self, lat: float, lon: float) -> Dict:
        """
        Get flood alerts data for the given coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dictionary containing alerts data
        """
        try:
            # Get flood-related alerts
            flood_alerts = self.alerts_agent.get_flood_related_alerts([(lat, lon)])
            
            if flood_alerts and len(flood_alerts) > 0:
                # Extract alerts from the first location
                alerts = flood_alerts[0].get("flood_alerts", [])
            else:
                alerts = []
            
            return {
                "alerts": alerts,
                "source": "MSN Weather",
                "location": {"lat": lat, "lon": lon}
            }
            
        except Exception as e:
            logger.error(f"Error getting alerts data: {e}")
            return {
                "alerts": [],
                "source": "MSN Weather",
                "location": {"lat": lat, "lon": lon},
                "error": str(e)
            }
    
    async def _parse_location(self, location: str) -> Tuple[float, float]:
        """
        Parse location string to get coordinates.
        
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
                    try:
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            return lat, lon
                    except ValueError:
                        pass
            
            # For place names, use a simple mapping
            location_lower = location.lower().strip()
            place_coordinates = {
                "chilliwack": (49.16, -121.96),
                "chilliwack, bc": (49.16, -121.96),
                "seattle": (47.6062, -122.3321),
                "seattle, wa": (47.6062, -122.3321),
                "vancouver": (49.2827, -123.1207),
                "vancouver, bc": (49.2827, -123.1207),
            }
            
            if location_lower in place_coordinates:
                return place_coordinates[location_lower]
            
            # Default to Chilliwack if not found
            logger.warning(f"Location not found: {location}. Using default coordinates.")
            return 49.16, -121.96
            
        except Exception as e:
            logger.error(f"Error parsing location {location}: {e}")
            return 49.16, -121.96  # Default to Chilliwack
    
    def _get_current_utc_time(self) -> str:
        """Get current UTC time in ISO format."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


async def main():
    """
    Main function to demonstrate the integration.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize integration (without OpenAI API key for now)
    integration = FloodRiskIntegration()
    
    # Test locations
    test_locations = [
        "Chilliwack, BC",
        "Seattle, WA",
        "49.16,-121.96"  # Chilliwack coordinates
    ]
    
    print("🌊 Flood Risk Integration Test")
    print("=" * 60)
    
    for location in test_locations:
        print(f"\n📍 Testing location: {location}")
        print("-" * 40)
        
        try:
            # Get comprehensive risk assessment
            result = await integration.get_comprehensive_risk_assessment(location)
            
            # Display results
            print(f"Location: {result['location']}")
            print(f"Timestamp: {result['timestamp']}")
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            # Display risk assessment
            risk = result.get("risk_assessment", {})
            print(f"\n🚨 Risk Level: {risk.get('risk_level', 'unknown').upper()}")
            print(f"🎯 Confidence: {risk.get('confidence_0_1', 0):.2f}")
            print(f"📋 Key Drivers:")
            for driver in risk.get('key_drivers', []):
                print(f"  • {driver}")
            print(f"💭 Rationale: {risk.get('rationale', 'N/A')}")
            print(f"📊 Sources: {', '.join(risk.get('sources_used', []))}")
            print(f"📝 Notes: {risk.get('notes', 'N/A')}")
            
            # Display data sources summary
            data_sources = result.get("data_sources", {})
            print(f"\n📈 Data Sources Summary:")
            
            # Hydrology
            hydrology = data_sources.get("hydrology", {})
            if hydrology:
                series = hydrology.get("series_daily_cms", [])
                print(f"  • Hydrology: {len(series)} days of data")
                if series:
                    latest = series[-1]
                    print(f"    Latest: {latest.get('q', 0):.1f} cms on {latest.get('date', 'N/A')}")
            
            # Alerts
            alerts = data_sources.get("alerts", {})
            alert_count = len(alerts.get("alerts", []))
            print(f"  • Alerts: {alert_count} active flood alerts")
            
            # Precipitation
            precip = data_sources.get("precipitation", {})
            if precip:
                past72h = precip.get("past72h_mm", 0)
                qpf_24h = precip.get("qpf_24h_max_next72h_mm", 0)
                print(f"  • Precipitation: {past72h:.1f}mm past 72h, {qpf_24h:.1f}mm forecast 24h max")
            
        except Exception as e:
            print(f"❌ Error processing {location}: {e}")
    
    print("\n" + "=" * 60)
    print("Integration test completed!")


if __name__ == "__main__":
    asyncio.run(main())
