import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskOverviewAgent:
    """
    Agent for generating flood risk overview based on alerts, hydrology, and precipitation data.
    Analyzes multiple data sources to provide a comprehensive risk assessment.
    """
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        # OpenAI client will be initialized only if API key is provided
        self.client = None
        if openai_api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=openai_api_key)
            except ImportError:
                logger.warning("OpenAI library not available. Using rule-based assessment only.")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}. Using rule-based assessment only.")
    
    def generate_risk_overview(self, input_data: Dict) -> Dict:
        """
        Generate flood risk overview based on alerts, hydrology, and precipitation data.
        
        Args:
            input_data: Dictionary containing location, time, and signals data
                {
                    "location": {"name": str, "lat": float, "lon": float},
                    "time_utc": str,
                    "signals": {
                        "alerts": List[Dict],
                        "hydrology_forecast": Dict,
                        "precip": Dict
                    }
                }
        
        Returns:
            Dictionary containing risk assessment in the specified format:
            {
                "risk_level": "none|low|medium|high|extreme",
                "confidence_0_1": float,
                "key_drivers": List[str],
                "rationale": str,
                "sources_used": List[str],
                "notes": str
            }
        """
        try:
            # Extract data from input
            location = input_data.get("location", {})
            time_utc = input_data.get("time_utc", "")
            signals = input_data.get("signals", {})
            
            alerts = signals.get("alerts", [])
            hydrology = signals.get("hydrology_forecast", {})
            precip = signals.get("precip", {})
            
            # Analyze each data source
            alert_analysis = self._analyze_alerts(alerts)
            hydrology_analysis = self._analyze_hydrology(hydrology)
            precip_analysis = self._analyze_precipitation(precip)
            
            # Generate risk assessment using AI if available, otherwise use rule-based
            if self.client:
                risk_assessment = self._generate_risk_assessment_with_ai(
                    location, time_utc, alert_analysis, hydrology_analysis, precip_analysis
                )
            else:
                risk_assessment = self._generate_rule_based_assessment(
                    alert_analysis, hydrology_analysis, precip_analysis
                )
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error generating risk overview: {e}")
            return self._get_default_risk_assessment()
    
    def _analyze_alerts(self, alerts: List[Dict]) -> Dict:
        """
        Analyze flood alerts to determine alert-based risk indicators.
        
        Args:
            alerts: List of alert dictionaries
        
        Returns:
            Dictionary containing alert analysis
        """
        if not alerts:
            return {
                "has_alerts": False,
                "severity_level": "none",
                "urgency_level": "none",
                "certainty_level": "none",
                "alert_count": 0,
                "risk_indicators": []
            }
        
        # Analyze alert characteristics
        severity_levels = [alert.get("severity", "").lower() for alert in alerts]
        urgency_levels = [alert.get("urgency", "").lower() for alert in alerts]
        certainty_levels = [alert.get("certainty", "").lower() for alert in alerts]
        
        # Determine highest severity, urgency, and certainty
        max_severity = self._get_max_severity(severity_levels)
        max_urgency = self._get_max_urgency(urgency_levels)
        max_certainty = self._get_max_certainty(certainty_levels)
        
        # Generate risk indicators
        risk_indicators = []
        if max_severity in ["severe", "extreme"]:
            risk_indicators.append("High severity alerts active")
        if max_urgency in ["immediate", "urgent"]:
            risk_indicators.append("Immediate action required")
        if max_certainty in ["likely", "certain"]:
            risk_indicators.append("High certainty of flood conditions")
        
        return {
            "has_alerts": True,
            "severity_level": max_severity,
            "urgency_level": max_urgency,
            "certainty_level": max_certainty,
            "alert_count": len(alerts),
            "risk_indicators": risk_indicators,
            "alerts": alerts
        }
    
    def _analyze_hydrology(self, hydrology: Dict) -> Dict:
        """
        Analyze hydrological forecast data to determine river-based risk indicators.
        
        Args:
            hydrology: Dictionary containing hydrological forecast data
        
        Returns:
            Dictionary containing hydrology analysis
        """
        if not hydrology:
            return {
                "has_data": False,
                "risk_level": "unknown",
                "risk_indicators": [],
                "threshold_status": "unknown"
            }
        
        # Extract forecast data
        forecast = hydrology.get("forecast", {})
        thresholds = hydrology.get("thresholds", {})
        
        peak_72h = forecast.get("peak_next72h_cms", 0)
        peak_7d = forecast.get("peak_next7d_cms", 0)
        leadtime = forecast.get("leadtime_hours_to_peak", 0)
        
        watch_threshold = thresholds.get("watch_cms", 350)
        warning_threshold = thresholds.get("warning_cms", 430)
        danger_threshold = thresholds.get("danger_cms", 500)
        
        # Determine risk level based on thresholds
        risk_level = "low"
        threshold_status = "below_watch"
        risk_indicators = []
        
        if peak_72h >= danger_threshold or peak_7d >= danger_threshold:
            risk_level = "extreme"
            threshold_status = "danger"
            risk_indicators.append(f"River discharge ({peak_72h} cms) exceeds danger threshold ({danger_threshold} cms)")
        elif peak_72h >= warning_threshold or peak_7d >= warning_threshold:
            risk_level = "high"
            threshold_status = "warning"
            risk_indicators.append(f"River discharge ({peak_72h} cms) exceeds warning threshold ({warning_threshold} cms)")
        elif peak_72h >= watch_threshold or peak_7d >= watch_threshold:
            risk_level = "medium"
            threshold_status = "watch"
            risk_indicators.append(f"River discharge ({peak_72h} cms) exceeds watch threshold ({watch_threshold} cms)")
        
        # Add lead time information
        if leadtime > 0:
            risk_indicators.append(f"Peak discharge expected in {leadtime} hours")
        
        return {
            "has_data": True,
            "risk_level": risk_level,
            "risk_indicators": risk_indicators,
            "threshold_status": threshold_status,
            "peak_72h_cms": peak_72h,
            "peak_7d_cms": peak_7d,
            "leadtime_hours": leadtime,
            "thresholds": thresholds
        }
    
    def _analyze_precipitation(self, precip: Dict) -> Dict:
        """
        Analyze precipitation data to determine rainfall-based risk indicators.
        
        Args:
            precip: Dictionary containing precipitation data
        
        Returns:
            Dictionary containing precipitation analysis
        """
        if not precip:
            return {
                "has_data": False,
                "risk_level": "unknown",
                "risk_indicators": []
            }
        
        # Extract precipitation metrics
        past72h = precip.get("past72h_mm", 0)
        max6h_past72h = precip.get("max6h_past72h_mm", 0)
        qpf_6h_max = precip.get("qpf_6h_max_next72h_mm", 0)
        qpf_24h_max = precip.get("qpf_24h_max_next72h_mm", 0)
        qpf_7d_sum = precip.get("qpf_sum_next7d_mm", 0)
        
        # Define precipitation thresholds (mm)
        heavy_rain_6h = 25  # Heavy rain threshold for 6 hours
        heavy_rain_24h = 50  # Heavy rain threshold for 24 hours
        extreme_rain_6h = 50  # Extreme rain threshold for 6 hours
        extreme_rain_24h = 100  # Extreme rain threshold for 24 hours
        
        risk_level = "low"
        risk_indicators = []
        
        # Analyze past precipitation
        if past72h > 100:
            risk_level = "high"
            risk_indicators.append(f"High recent rainfall: {past72h}mm in past 72h")
        elif past72h > 50:
            risk_level = "medium"
            risk_indicators.append(f"Moderate recent rainfall: {past72h}mm in past 72h")
        
        if max6h_past72h > extreme_rain_6h:
            risk_level = "extreme"
            risk_indicators.append(f"Extreme 6-hour rainfall: {max6h_past72h}mm")
        elif max6h_past72h > heavy_rain_6h:
            risk_level = "high"
            risk_indicators.append(f"Heavy 6-hour rainfall: {max6h_past72h}mm")
        
        # Analyze forecast precipitation
        if qpf_6h_max > extreme_rain_6h:
            risk_level = "extreme"
            risk_indicators.append(f"Extreme 6-hour forecast: {qpf_6h_max}mm")
        elif qpf_6h_max > heavy_rain_6h:
            risk_level = "high"
            risk_indicators.append(f"Heavy 6-hour forecast: {qpf_6h_max}mm")
        
        if qpf_24h_max > extreme_rain_24h:
            risk_level = "extreme"
            risk_indicators.append(f"Extreme 24-hour forecast: {qpf_24h_max}mm")
        elif qpf_24h_max > heavy_rain_24h:
            risk_level = "high"
            risk_indicators.append(f"Heavy 24-hour forecast: {qpf_24h_max}mm")
        
        if qpf_7d_sum > 200:
            risk_level = "high"
            risk_indicators.append(f"High 7-day forecast: {qpf_7d_sum}mm")
        
        return {
            "has_data": True,
            "risk_level": risk_level,
            "risk_indicators": risk_indicators,
            "past72h_mm": past72h,
            "max6h_past72h_mm": max6h_past72h,
            "qpf_6h_max_next72h_mm": qpf_6h_max,
            "qpf_24h_max_next72h_mm": qpf_24h_max,
            "qpf_sum_next7d_mm": qpf_7d_sum
        }
    
    def _generate_risk_assessment_with_ai(self, location: Dict, time_utc: str, 
                                        alert_analysis: Dict, hydrology_analysis: Dict, 
                                        precip_analysis: Dict) -> Dict:
        """
        Use OpenAI to generate a comprehensive risk assessment.
        
        Args:
            location: Location information
            time_utc: UTC timestamp
            alert_analysis: Alert analysis results
            hydrology_analysis: Hydrology analysis results
            precip_analysis: Precipitation analysis results
        
        Returns:
            Dictionary containing risk assessment
        """
        try:
            # Prepare context for AI
            context = self._prepare_ai_context(location, time_utc, alert_analysis, hydrology_analysis, precip_analysis)
            
            # Create prompt for OpenAI
            prompt = f"""
You are a flood-risk analyst. Analyze the following data and provide a structured risk assessment.

Location: {location.get('name', 'Unknown')} ({location.get('lat', 0)}, {location.get('lon', 0)})
Time: {time_utc}

ALERT ANALYSIS:
- Has alerts: {alert_analysis.get('has_alerts', False)}
- Severity: {alert_analysis.get('severity_level', 'none')}
- Urgency: {alert_analysis.get('urgency_level', 'none')}
- Certainty: {alert_analysis.get('certainty_level', 'none')}
- Risk indicators: {alert_analysis.get('risk_indicators', [])}

HYDROLOGY ANALYSIS:
- Has data: {hydrology_analysis.get('has_data', False)}
- Risk level: {hydrology_analysis.get('risk_level', 'unknown')}
- Peak 72h: {hydrology_analysis.get('peak_72h_cms', 0)} cms
- Peak 7d: {hydrology_analysis.get('peak_7d_cms', 0)} cms
- Threshold status: {hydrology_analysis.get('threshold_status', 'unknown')}
- Risk indicators: {hydrology_analysis.get('risk_indicators', [])}

PRECIPITATION ANALYSIS:
- Has data: {precip_analysis.get('has_data', False)}
- Risk level: {precip_analysis.get('risk_level', 'unknown')}
- Past 72h: {precip_analysis.get('past72h_mm', 0)}mm
- Max 6h past 72h: {precip_analysis.get('max6h_past72h_mm', 0)}mm
- QPF 6h max next 72h: {precip_analysis.get('qpf_6h_max_next72h_mm', 0)}mm
- QPF 24h max next 72h: {precip_analysis.get('qpf_24h_max_next72h_mm', 0)}mm
- QPF 7d sum: {precip_analysis.get('qpf_sum_next7d_mm', 0)}mm
- Risk indicators: {precip_analysis.get('risk_indicators', [])}

Based on this analysis, provide a JSON response with the following structure:
{{
    "risk_level": "none|low|medium|high|extreme",
    "confidence_0_1": <float between 0 and 1>,
    "key_drivers": ["bullet point list of main risk factors"],
    "rationale": "1-3 sentences explaining how you combined the data and why",
    "sources_used": ["alerts: MSN Weather", "hydrology: <station/location>", "precip: <provider>"],
    "notes": "any caveats or limitations"
}}

Guidelines:
- If MSN Flood Warning/Danger alerts are active, normally imply at least medium risk unless hydrology and rainfall contradict
- If no alerts but hydrology and precipitation are high, still raise risk accordingly
- If information conflicts, explain which evidence you trusted most and why
- If data quality is low or uncertain, reduce confidence_0_1
- Consider the most severe indicators from any source
- Be conservative but realistic in assessment
"""
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a flood-risk analyst. Provide structured JSON responses for risk assessment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse response
            ai_response = response.choices[0].message.content.strip()
            
            # Try to extract JSON from response
            try:
                # Find JSON in the response
                start_idx = ai_response.find('{')
                end_idx = ai_response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = ai_response[start_idx:end_idx]
                    risk_assessment = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse AI response as JSON: {e}")
                # Fallback to rule-based assessment
                risk_assessment = self._generate_rule_based_assessment(alert_analysis, hydrology_analysis, precip_analysis)
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Fallback to rule-based assessment
            return self._generate_rule_based_assessment(alert_analysis, hydrology_analysis, precip_analysis)
    
    def _prepare_ai_context(self, location: Dict, time_utc: str, 
                          alert_analysis: Dict, hydrology_analysis: Dict, 
                          precip_analysis: Dict) -> str:
        """
        Prepare context string for AI analysis.
        
        Args:
            location: Location information
            time_utc: UTC timestamp
            alert_analysis: Alert analysis results
            hydrology_analysis: Hydrology analysis results
            precip_analysis: Precipitation analysis results
        
        Returns:
            Formatted context string
        """
        context_parts = [
            f"Location: {location.get('name', 'Unknown')} ({location.get('lat', 0)}, {location.get('lon', 0)})",
            f"Time: {time_utc}",
            "",
            "ALERT ANALYSIS:",
            f"- Has alerts: {alert_analysis.get('has_alerts', False)}",
            f"- Severity: {alert_analysis.get('severity_level', 'none')}",
            f"- Urgency: {alert_analysis.get('urgency_level', 'none')}",
            f"- Certainty: {alert_analysis.get('certainty_level', 'none')}",
            f"- Risk indicators: {alert_analysis.get('risk_indicators', [])}",
            "",
            "HYDROLOGY ANALYSIS:",
            f"- Has data: {hydrology_analysis.get('has_data', False)}",
            f"- Risk level: {hydrology_analysis.get('risk_level', 'unknown')}",
            f"- Peak 72h: {hydrology_analysis.get('peak_72h_cms', 0)} cms",
            f"- Peak 7d: {hydrology_analysis.get('peak_7d_cms', 0)} cms",
            f"- Threshold status: {hydrology_analysis.get('threshold_status', 'unknown')}",
            f"- Risk indicators: {hydrology_analysis.get('risk_indicators', [])}",
            "",
            "PRECIPITATION ANALYSIS:",
            f"- Has data: {precip_analysis.get('has_data', False)}",
            f"- Risk level: {precip_analysis.get('risk_level', 'unknown')}",
            f"- Past 72h: {precip_analysis.get('past72h_mm', 0)}mm",
            f"- Max 6h past 72h: {precip_analysis.get('max6h_past72h_mm', 0)}mm",
            f"- QPF 6h max next 72h: {precip_analysis.get('qpf_6h_max_next72h_mm', 0)}mm",
            f"- QPF 24h max next 72h: {precip_analysis.get('qpf_24h_max_next72h_mm', 0)}mm",
            f"- QPF 7d sum: {precip_analysis.get('qpf_sum_next7d_mm', 0)}mm",
            f"- Risk indicators: {precip_analysis.get('risk_indicators', [])}"
        ]
        
        return "\n".join(context_parts)
    
    def _generate_rule_based_assessment(self, alert_analysis: Dict, 
                                      hydrology_analysis: Dict, 
                                      precip_analysis: Dict) -> Dict:
        """
        Generate risk assessment using rule-based logic as fallback.
        
        Args:
            alert_analysis: Alert analysis results
            hydrology_analysis: Hydrology analysis results
            precip_analysis: Precipitation analysis results
        
        Returns:
            Dictionary containing risk assessment
        """
        # Determine risk level based on rules
        risk_level = "none"
        confidence = 0.5
        key_drivers = []
        rationale_parts = []
        sources_used = []
        
        # Check alerts
        if alert_analysis.get("has_alerts", False):
            sources_used.append("alerts: MSN Weather")
            severity = alert_analysis.get("severity_level", "none")
            urgency = alert_analysis.get("urgency_level", "none")
            certainty = alert_analysis.get("certainty_level", "none")
            
            if severity in ["severe", "extreme"] and urgency in ["immediate", "urgent"]:
                risk_level = "high"
                key_drivers.append("Active severe flood alerts with immediate urgency")
                rationale_parts.append("Severe flood alerts indicate high risk")
                confidence = 0.8
            elif severity in ["moderate", "severe"]:
                risk_level = "medium"
                key_drivers.append("Active flood alerts present")
                rationale_parts.append("Flood alerts indicate elevated risk")
                confidence = 0.7
        
        # Check hydrology
        if hydrology_analysis.get("has_data", False):
            sources_used.append("hydrology: River station")
            hydrology_risk = hydrology_analysis.get("risk_level", "low")
            peak_72h = hydrology_analysis.get("peak_72h_cms", 0)
            
            if hydrology_risk == "extreme":
                if risk_level in ["none", "low"]:
                    risk_level = "high"
                key_drivers.append(f"River discharge ({peak_72h} cms) exceeds danger threshold")
                rationale_parts.append("River levels indicate extreme flood risk")
                confidence = min(confidence + 0.2, 0.9)
            elif hydrology_risk == "high":
                if risk_level in ["none", "low"]:
                    risk_level = "medium"
                key_drivers.append(f"River discharge ({peak_72h} cms) exceeds warning threshold")
                rationale_parts.append("River levels indicate high flood risk")
                confidence = min(confidence + 0.1, 0.8)
            elif hydrology_risk == "medium":
                if risk_level == "none":
                    risk_level = "low"
                key_drivers.append(f"River discharge ({peak_72h} cms) exceeds watch threshold")
                rationale_parts.append("River levels indicate moderate flood risk")
        
        # Check precipitation
        if precip_analysis.get("has_data", False):
            sources_used.append("precip: Weather service")
            precip_risk = precip_analysis.get("risk_level", "low")
            past72h = precip_analysis.get("past72h_mm", 0)
            qpf_24h_max = precip_analysis.get("qpf_24h_max_next72h_mm", 0)
            
            if precip_risk == "extreme":
                if risk_level in ["none", "low", "medium"]:
                    risk_level = "high"
                key_drivers.append(f"Extreme precipitation forecast ({qpf_24h_max}mm in 24h)")
                rationale_parts.append("Extreme precipitation forecast indicates high flood risk")
                confidence = min(confidence + 0.2, 0.9)
            elif precip_risk == "high":
                if risk_level in ["none", "low"]:
                    risk_level = "medium"
                key_drivers.append(f"Heavy precipitation forecast ({qpf_24h_max}mm in 24h)")
                rationale_parts.append("Heavy precipitation forecast indicates elevated flood risk")
                confidence = min(confidence + 0.1, 0.8)
            elif past72h > 50:
                key_drivers.append(f"Recent heavy rainfall ({past72h}mm in 72h)")
                rationale_parts.append("Recent heavy rainfall increases flood risk")
        
        # Generate rationale
        if rationale_parts:
            rationale = " ".join(rationale_parts) + "."
        else:
            rationale = "No significant flood risk indicators detected."
        
        # Add notes
        notes = []
        if not alert_analysis.get("has_alerts", False):
            notes.append("No active flood alerts")
        if not hydrology_analysis.get("has_data", False):
            notes.append("No hydrological data available")
        if not precip_analysis.get("has_data", False):
            notes.append("No precipitation data available")
        
        if not sources_used:
            sources_used = ["No data sources available"]
        
        return {
            "risk_level": risk_level,
            "confidence_0_1": round(confidence, 2),
            "key_drivers": key_drivers,
            "rationale": rationale,
            "sources_used": sources_used,
            "notes": "; ".join(notes) if notes else "All data sources available"
        }
    
    def _get_max_severity(self, severity_levels: List[str]) -> str:
        """Get the highest severity level from a list."""
        severity_hierarchy = ["none", "low", "moderate", "severe", "extreme"]
        max_severity = "none"
        for severity in severity_levels:
            if severity in severity_hierarchy:
                if severity_hierarchy.index(severity) > severity_hierarchy.index(max_severity):
                    max_severity = severity
        return max_severity
    
    def _get_max_urgency(self, urgency_levels: List[str]) -> str:
        """Get the highest urgency level from a list."""
        urgency_hierarchy = ["none", "low", "moderate", "urgent", "immediate"]
        max_urgency = "none"
        for urgency in urgency_levels:
            if urgency in urgency_hierarchy:
                if urgency_hierarchy.index(urgency) > urgency_hierarchy.index(max_urgency):
                    max_urgency = urgency
        return max_urgency
    
    def _get_max_certainty(self, certainty_levels: List[str]) -> str:
        """Get the highest certainty level from a list."""
        certainty_hierarchy = ["none", "unlikely", "possible", "likely", "certain"]
        max_certainty = "none"
        for certainty in certainty_levels:
            if certainty in certainty_hierarchy:
                if certainty_hierarchy.index(certainty) > certainty_hierarchy.index(max_certainty):
                    max_certainty = certainty
        return max_certainty
    
    def _get_default_risk_assessment(self) -> Dict:
        """Get default risk assessment when analysis fails."""
        return {
            "risk_level": "unknown",
            "confidence_0_1": 0.0,
            "key_drivers": ["Analysis failed - insufficient data"],
            "rationale": "Unable to analyze flood risk due to data processing error.",
            "sources_used": ["Error: No data sources available"],
            "notes": "Risk assessment failed - check data sources and try again"
        }


async def main():
    """
    Main function to test the risk overview agent.
    """
    # Initialize the agent
    agent = RiskOverviewAgent()
    
    # Test with sample data
    sample_data = {
        "location": {"name": "Chilliwack, BC", "lat": 49.16, "lon": -121.96},
        "time_utc": "2025-10-18T20:05Z",
        "signals": {
            "alerts": [
                {
                    "provenance": "aggregator",
                    "source_name": "MSN Weather",
                    "source_url": "https://api.msn.com/…",
                    "origin_agency_hint": "ECCC",
                    "event": "Flood Warning",
                    "severity": "Severe",
                    "urgency": "Immediate",
                    "certainty": "Likely",
                    "effective": "2025-10-18T15:00Z",
                    "expires": "2025-10-19T06:00Z",
                    "area": {"type": "city", "name": "Chilliwack"},
                    "cap_id": None,
                    "summary": "Flooding possible along lower Vedder. Avoid low-lying areas.",
                    "hash": "floodwarning_chilliwack_20251018T1500Z"
                }
            ],
            "hydrology_forecast": {
                "station_id": "08MH001",
                "river": "Vedder",
                "forecast": {"peak_next72h_cms": 480, "peak_next7d_cms": 520, "leadtime_hours_to_peak": 36},
                "thresholds": {"watch_cms": 350, "warning_cms": 430, "danger_cms": 500}
            },
            "precip": {
                "past72h_mm": 62,
                "max6h_past72h_mm": 18,
                "qpf_6h_max_next72h_mm": 34,
                "qpf_24h_max_next72h_mm": 82,
                "qpf_sum_next7d_mm": 120
            }
        }
    }
    
    print("🌊 Risk Overview Agent Test")
    print("=" * 50)
    
    # Generate risk overview
    risk_assessment = agent.generate_risk_overview(sample_data)
    
    print("Risk Assessment:")
    print(json.dumps(risk_assessment, indent=2))
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
