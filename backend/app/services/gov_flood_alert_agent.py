import aiohttp
import asyncio
import logging
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Union
from urllib.parse import urlencode
import re

logger = logging.getLogger(__name__)

class GovFloodAlertAgent:
    """
    Agent for fetching live flood alerts from government sources for US and Canada.
    Integrates with multiple official data sources to provide comprehensive flood alert information.
    """
    
    def __init__(self):
        # US Government APIs
        self.nws_alerts_url = "https://api.weather.gov/alerts"
        self.usgs_water_data_url = "https://waterservices.usgs.gov/nwis/iv"
        self.usgs_flood_impacts_url = "https://api.waterdata.usgs.gov/nwis/iv"
        
        # Canada Government APIs (via third-party aggregators)
        self.weatherbit_alerts_url = "https://api.weatherbit.io/v2.0/alerts"
        self.weatherbit_api_key = "YOUR_WEATHERBIT_API_KEY"  # Replace with actual key
        
        # Alert severity mapping
        self.severity_mapping = {
            "extreme": 5,
            "severe": 4,
            "moderate": 3,
            "minor": 2,
            "unknown": 1
        }
        
        # Flood-related keywords for filtering
        self.flood_keywords = [
            "flood", "flooding", "flash flood", "river flood", "coastal flood",
            "storm surge", "high water", "overflow", "breach", "levee",
            "inundation", "deluge", "torrential", "downpour", "heavy rain",
            "excessive rainfall", "urban flooding", "areal flooding",
            "severe thunderstorm", "thunderstorm warning", "storm warning",
            "precipitation", "rain", "rainfall", "storm", "severe weather",
            "weather warning", "weather watch", "hydrologic", "water level",
            "stream", "river", "creek", "watershed", "runoff"
        ]
    
    async def get_us_nws_alerts(self, state: str = None, county: str = None, 
                               severity: str = None, urgency: str = None) -> List[Dict]:
        """
        Get flood alerts from US National Weather Service.
        
        Args:
            state: US state code (e.g., 'CA', 'NY', 'KS')
            county: County name (optional)
            severity: Alert severity filter (minor, moderate, severe, extreme)
            urgency: Alert urgency filter (immediate, expected, future, past)
        
        Returns:
            List of flood alert dictionaries
        """
        try:
            # Build query parameters
            params = []
            
            if state:
                params.append(f"area={state.upper()}")
            
            if severity:
                params.append(f"severity={severity}")
            
            if urgency:
                params.append(f"urgency={urgency}")
            
            # Construct URL with parameters
            if params:
                url = f"{self.nws_alerts_url}/active?{'&'.join(params)}"
            else:
                url = f"{self.nws_alerts_url}/active"
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "User-Agent": "FloodPredictionSystem/1.0 (contact@example.com)",
                    "Accept": "application/geo+json"
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        alerts = self._parse_nws_alerts(data)
                        return self._filter_flood_alerts(alerts)
                    else:
                        logger.error(f"NWS API error: {response.status} - {url}")
                        # Try to get response text for debugging
                        try:
                            error_text = await response.text()
                            logger.error(f"NWS API error response: {error_text[:200]}")
                        except:
                            pass
                        return []
        except Exception as e:
            logger.error(f"Error fetching NWS alerts: {e}")
            return []
    
    async def get_us_usgs_flood_data(self, state: str = None, huc: str = None) -> List[Dict]:
        """
        Get flood-related data from USGS Water Data API.
        
        Args:
            state: US state code
            huc: Hydrologic Unit Code
        
        Returns:
            List of flood data dictionaries
        """
        try:
            params = {
                "format": "json",
                "parameterCd": "00065,00060",  # Gage height and discharge
                "siteStatus": "active"
            }
            
            if state:
                params["stateCd"] = state
            if huc:
                params["huc"] = huc
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.usgs_water_data_url}?{urlencode(params)}"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_usgs_flood_data(data)
                    else:
                        logger.error(f"USGS API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching USGS flood data: {e}")
            return []
    
    async def get_canada_environment_alerts(self, province: str = None) -> List[Dict]:
        """
        Get flood alerts from Environment Canada via Weatherbit API.
        
        Args:
            province: Canadian province code (e.g., 'ON', 'BC', 'AB')
        
        Returns:
            List of flood alert dictionaries
        """
        try:
            params = {
                "key": self.weatherbit_api_key,
                "country": "CA"
            }
            
            if province:
                params["state"] = province
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.weatherbit_alerts_url}?{urlencode(params)}"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_weatherbit_alerts(data)
                    else:
                        logger.error(f"Weatherbit API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching Canada alerts: {e}")
            return []
    
    def _parse_nws_alerts(self, data: Dict) -> List[Dict]:
        """Parse NWS alerts JSON response."""
        alerts = []
        
        try:
            features = data.get("features", [])
            for feature in features:
                properties = feature.get("properties", {})
                geometry = feature.get("geometry", {})
                
                alert = {
                    "id": properties.get("id"),
                    "title": properties.get("headline"),
                    "description": properties.get("description"),
                    "severity": properties.get("severity", "unknown"),
                    "urgency": properties.get("urgency", "unknown"),
                    "certainty": properties.get("certainty", "unknown"),
                    "event": properties.get("event"),
                    "area_desc": properties.get("areaDesc"),
                    "effective": properties.get("effective"),
                    "expires": properties.get("expires"),
                    "onset": properties.get("onset"),
                    "ends": properties.get("ends"),
                    "status": properties.get("status"),
                    "message_type": properties.get("messageType"),
                    "category": properties.get("category"),
                    "response": properties.get("response"),
                    "sender": properties.get("sender"),
                    "sender_name": properties.get("senderName"),
                    "instruction": properties.get("instruction"),
                    "geometry": geometry,
                    "source": "NWS",
                    "country": "US"
                }
                alerts.append(alert)
        except Exception as e:
            logger.error(f"Error parsing NWS alerts: {e}")
        
        return alerts
    
    def _parse_usgs_flood_data(self, data: Dict) -> List[Dict]:
        """Parse USGS water data JSON response."""
        flood_data = []
        
        try:
            time_series = data.get("value", {}).get("timeSeries", [])
            for series in time_series:
                site_info = series.get("sourceInfo", {})
                site_code = site_info.get("siteCode", [{}])[0].get("value")
                site_name = site_info.get("siteName")
                
                values = series.get("values", [])
                if values:
                    value_data = values[0].get("value", [])
                    if value_data:
                        latest_value = value_data[-1]
                        
                        flood_info = {
                            "site_code": site_code,
                            "site_name": site_name,
                            "parameter": series.get("variable", {}).get("variableDescription"),
                            "unit": series.get("variable", {}).get("unit", {}).get("unitCode"),
                            "value": latest_value.get("value"),
                            "date_time": latest_value.get("dateTime"),
                            "qualifiers": latest_value.get("qualifiers", []),
                            "source": "USGS",
                            "country": "US"
                        }
                        flood_data.append(flood_info)
        except Exception as e:
            logger.error(f"Error parsing USGS flood data: {e}")
        
        return flood_data
    
    def _parse_weatherbit_alerts(self, data: Dict) -> List[Dict]:
        """Parse Weatherbit alerts JSON response."""
        alerts = []
        
        try:
            alert_list = data.get("alerts", [])
            for alert in alert_list:
                parsed_alert = {
                    "id": alert.get("alert_id"),
                    "title": alert.get("title"),
                    "description": alert.get("description"),
                    "severity": alert.get("severity"),
                    "urgency": alert.get("urgency"),
                    "certainty": alert.get("certainty"),
                    "event": alert.get("event"),
                    "area_desc": alert.get("areas"),
                    "effective": alert.get("effective_utc"),
                    "expires": alert.get("expires_utc"),
                    "onset": alert.get("onset_utc"),
                    "ends": alert.get("ends_utc"),
                    "status": alert.get("status"),
                    "message_type": alert.get("message_type"),
                    "category": alert.get("category"),
                    "instruction": alert.get("instruction"),
                    "source": "Environment Canada",
                    "country": "CA"
                }
                alerts.append(parsed_alert)
        except Exception as e:
            logger.error(f"Error parsing Weatherbit alerts: {e}")
        
        return alerts
    
    def _filter_flood_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Filter alerts to only include flood-related ones."""
        flood_alerts = []
        
        for alert in alerts:
            # Check if alert is flood-related
            is_flood_related = False
            matched_keywords = []
            
            # Check various text fields for flood keywords
            text_fields = [
                alert.get("title", ""),
                alert.get("description", ""),
                alert.get("event", ""),
                alert.get("area_desc", ""),
                alert.get("instruction", "")
            ]
            
            combined_text = " ".join([str(field) for field in text_fields]).lower()
            
            for keyword in self.flood_keywords:
                if keyword in combined_text:
                    is_flood_related = True
                    matched_keywords.append(keyword)
                    break
            
            if is_flood_related:
                # Add flood-specific metadata
                alert["is_flood_alert"] = True
                alert["flood_type"] = self._classify_flood_type(combined_text)
                alert["severity_score"] = self.severity_mapping.get(alert.get("severity", "unknown"), 1)
                alert["matched_keywords"] = matched_keywords
                flood_alerts.append(alert)
        
        return flood_alerts
    
    def _classify_flood_type(self, text: str) -> str:
        """Classify the type of flood based on alert text."""
        text_lower = text.lower()
        
        if "flash flood" in text_lower:
            return "flash_flood"
        elif "river flood" in text_lower:
            return "river_flood"
        elif "coastal flood" in text_lower or "storm surge" in text_lower:
            return "coastal_flood"
        elif "urban flood" in text_lower:
            return "urban_flood"
        elif "areal flood" in text_lower:
            return "areal_flood"
        else:
            return "general_flood"
    
    async def get_us_nws_flood_alerts(self, state: str = None) -> List[Dict]:
        """
        Get specifically flood-related alerts from NWS.
        
        Args:
            state: US state code (e.g., 'CA', 'NY', 'KS')
        
        Returns:
            List of flood alert dictionaries
        """
        try:
            # Get all alerts for the state
            all_alerts = await self.get_us_nws_alerts(state=state)
            
            # Filter for flood-related alerts
            flood_alerts = []
            for alert in all_alerts:
                if self._is_flood_alert(alert):
                    flood_alerts.append(alert)
            
            return flood_alerts
        except Exception as e:
            logger.error(f"Error fetching NWS flood alerts: {e}")
            return []
    
    def _is_flood_alert(self, alert: Dict) -> bool:
        """Check if an alert is flood-related."""
        text_fields = [
            alert.get("title", ""),
            alert.get("description", ""),
            alert.get("event", ""),
            alert.get("area_desc", ""),
            alert.get("instruction", "")
        ]
        
        combined_text = " ".join([str(field) for field in text_fields]).lower()
        
        return any(keyword in combined_text for keyword in self.flood_keywords)

    async def get_comprehensive_flood_alerts(self, 
                                           us_locations: List[Tuple[str, str]] = None,
                                           canada_locations: List[Tuple[str, str]] = None) -> Dict:
        """
        Get comprehensive flood alerts for both US and Canada.
        
        Args:
            us_locations: List of (state, county) tuples for US
            canada_locations: List of (province, city) tuples for Canada
        
        Returns:
            Dictionary containing all flood alerts organized by country
        """
        try:
            # Get US alerts
            us_alerts = []
            us_flood_data = []
            
            if us_locations:
                for state, county in us_locations:
                    nws_alerts = await self.get_us_nws_alerts(state=state)
                    usgs_data = await self.get_us_usgs_flood_data(state=state)
                    us_alerts.extend(nws_alerts)
                    us_flood_data.extend(usgs_data)
            else:
                # Get alerts for all states if no specific locations provided
                us_alerts = await self.get_us_nws_alerts()
                us_flood_data = await self.get_us_usgs_flood_data()
            
            # Get Canada alerts
            canada_alerts = []
            
            if canada_locations:
                provinces = list(set([loc[0] for loc in canada_locations]))
                for province in provinces:
                    alerts = await self.get_canada_environment_alerts(province=province)
                    canada_alerts.extend(alerts)
            else:
                canada_alerts = await self.get_canada_environment_alerts()
            
            # Process and categorize alerts
            processed_alerts = {
                "us": {
                    "nws_alerts": us_alerts,
                    "usgs_flood_data": us_flood_data,
                    "total_alerts": len(us_alerts),
                    "high_priority_alerts": len([a for a in us_alerts if a.get("severity_score", 0) >= 4])
                },
                "canada": {
                    "environment_canada_alerts": canada_alerts,
                    "total_alerts": len(canada_alerts),
                    "high_priority_alerts": len([a for a in canada_alerts if a.get("severity_score", 0) >= 4])
                },
                "summary": {
                    "total_us_alerts": len(us_alerts),
                    "total_canada_alerts": len(canada_alerts),
                    "total_flood_data_points": len(us_flood_data),
                    "last_updated": datetime.now(timezone.utc).isoformat()
                }
            }
            
            return processed_alerts
            
        except Exception as e:
            logger.error(f"Error getting comprehensive flood alerts: {e}")
            return {
                "error": str(e),
                "us": {"nws_alerts": [], "usgs_flood_data": [], "total_alerts": 0, "high_priority_alerts": 0},
                "canada": {"environment_canada_alerts": [], "total_alerts": 0, "high_priority_alerts": 0},
                "summary": {"total_us_alerts": 0, "total_canada_alerts": 0, "total_flood_data_points": 0, "last_updated": datetime.now(timezone.utc).isoformat()}
            }
    
    def format_alerts_for_display(self, alerts_data: Dict) -> str:
        """Format flood alerts data for display."""
        if not alerts_data or "error" in alerts_data:
            return "❌ Error retrieving flood alerts data."
        
        output = []
        output.append("🌊 FLOOD ALERT SUMMARY")
        output.append("=" * 50)
        
        # US Alerts
        us_data = alerts_data.get("us", {})
        us_alerts = us_data.get("nws_alerts", [])
        us_flood_data = us_data.get("usgs_flood_data", [])
        
        output.append(f"\n🇺🇸 UNITED STATES")
        output.append(f"   NWS Alerts: {len(us_alerts)}")
        output.append(f"   USGS Flood Data Points: {len(us_flood_data)}")
        output.append(f"   High Priority: {us_data.get('high_priority_alerts', 0)}")
        
        if us_alerts:
            output.append("\n   Active NWS Flood Alerts:")
            for i, alert in enumerate(us_alerts[:5], 1):  # Show first 5
                output.append(f"   {i}. {alert.get('title', 'Unknown Alert')}")
                output.append(f"      Severity: {alert.get('severity', 'Unknown')}")
                output.append(f"      Area: {alert.get('area_desc', 'Unknown Area')}")
                if alert.get('effective'):
                    output.append(f"      Effective: {alert.get('effective')}")
                output.append("")
        
        # Canada Alerts
        canada_data = alerts_data.get("canada", {})
        canada_alerts = canada_data.get("environment_canada_alerts", [])
        
        output.append(f"\n🇨🇦 CANADA")
        output.append(f"   Environment Canada Alerts: {len(canada_alerts)}")
        output.append(f"   High Priority: {canada_data.get('high_priority_alerts', 0)}")
        
        if canada_alerts:
            output.append("\n   Active Environment Canada Flood Alerts:")
            for i, alert in enumerate(canada_alerts[:5], 1):  # Show first 5
                output.append(f"   {i}. {alert.get('title', 'Unknown Alert')}")
                output.append(f"      Severity: {alert.get('severity', 'Unknown')}")
                output.append(f"      Area: {alert.get('area_desc', 'Unknown Area')}")
                if alert.get('effective'):
                    output.append(f"      Effective: {alert.get('effective')}")
                output.append("")
        
        # Summary
        summary = alerts_data.get("summary", {})
        output.append(f"\n📊 SUMMARY")
        output.append(f"   Total US Alerts: {summary.get('total_us_alerts', 0)}")
        output.append(f"   Total Canada Alerts: {summary.get('total_canada_alerts', 0)}")
        output.append(f"   Last Updated: {summary.get('last_updated', 'Unknown')}")
        
        return "\n".join(output)


# Example usage and testing
def main():
    """Main function to test the GovFloodAlertAgent."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Initialize the flood alert agent
    agent = GovFloodAlertAgent()
    
    # Test locations
    us_locations = [("CA", "Los Angeles"), ("NY", "New York"), ("TX", "Harris")]
    canada_locations = [("ON", "Toronto"), ("BC", "Vancouver"), ("AB", "Calgary")]
    
    print("🌊 Government Flood Alert Agent Test")
    print("=" * 50)
    
    # Test comprehensive flood alerts
    async def test_comprehensive_alerts():
        alerts_data = await agent.get_comprehensive_flood_alerts(
            us_locations=us_locations,
            canada_locations=canada_locations
        )
        print(agent.format_alerts_for_display(alerts_data))
    
    # Run the test
    asyncio.run(test_comprehensive_alerts())


if __name__ == "__main__":
    main()
