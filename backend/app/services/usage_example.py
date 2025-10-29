"""
Usage example showing how to use the Risk Overview Agent with the exact input format provided.
This demonstrates the complete workflow from input data to risk assessment.
"""

import asyncio
import json
import logging
from risk_overview_agent import RiskOverviewAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def main():
    """
    Main function demonstrating how to use the Risk Overview Agent.
    """
    print("🌊 Risk Overview Agent Usage Example")
    print("=" * 50)
    
    # Initialize the risk overview agent
    # You can provide your OpenAI API key here for AI-powered analysis
    agent = RiskOverviewAgent(openai_api_key=None)  # Using rule-based analysis for this example
    
    # Sample input data in the exact format you provided
    sample_input = {
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
    
    print("📊 Input Data:")
    print(json.dumps(sample_input, indent=2))
    print("\n" + "=" * 50)
    
    # Generate risk overview
    print("🔍 Generating Risk Assessment...")
    risk_assessment = agent.generate_risk_overview(sample_input)
    
    print("\n📋 Risk Assessment Results:")
    print("=" * 50)
    
    # Display the results in a formatted way
    print(f"🚨 Risk Level: {risk_assessment['risk_level'].upper()}")
    print(f"🎯 Confidence: {risk_assessment['confidence_0_1']:.2f}")
    
    print(f"\n📋 Key Drivers:")
    for i, driver in enumerate(risk_assessment['key_drivers'], 1):
        print(f"  {i}. {driver}")
    
    print(f"\n💭 Rationale:")
    print(f"   {risk_assessment['rationale']}")
    
    print(f"\n📊 Sources Used:")
    for source in risk_assessment['sources_used']:
        print(f"   • {source}")
    
    print(f"\n📝 Notes:")
    print(f"   {risk_assessment['notes']}")
    
    print("\n" + "=" * 50)
    print("✅ Risk assessment completed!")
    
    # Also show the raw JSON output
    print("\n📄 Raw JSON Output:")
    print(json.dumps(risk_assessment, indent=2))
    
    return risk_assessment

if __name__ == "__main__":
    asyncio.run(main())


