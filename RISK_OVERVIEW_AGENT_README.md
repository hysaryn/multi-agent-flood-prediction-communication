# 🌊 Risk Overview Agent

A comprehensive flood risk assessment system that analyzes multiple data sources to provide structured risk evaluations.

## Overview

The Risk Overview Agent is designed to analyze flood risk based on three key data sources:
- **Alerts**: MSN Weather flood alerts with severity, urgency, and certainty levels
- **Hydrology**: River discharge forecasts with watch/warning/danger thresholds  
- **Precipitation**: Historical and forecast rainfall data

## Features

### 🎯 Multi-Source Analysis
- **Alert Analysis**: Evaluates flood alerts for severity, urgency, and certainty
- **Hydrology Analysis**: Compares river discharge against established thresholds
- **Precipitation Analysis**: Assesses rainfall patterns and forecasts

### 🤖 Intelligent Assessment
- **AI-Powered**: Uses OpenAI GPT for sophisticated risk analysis (optional)
- **Rule-Based Fallback**: Robust rule-based system when AI is unavailable
- **Confidence Scoring**: Provides confidence levels based on data quality

### 📊 Structured Output
Returns JSON in the exact format specified:
```json
{
  "risk_level": "none|low|medium|high|extreme",
  "confidence_0_1": 0.8,
  "key_drivers": ["Active severe flood alerts", "River discharge exceeds danger threshold"],
  "rationale": "Combined analysis explanation",
  "sources_used": ["alerts: MSN Weather", "hydrology: River station", "precip: Weather service"],
  "notes": "Data quality notes"
}
```

## Files Created

### Core Agent
- `risk_overview_agent.py` - Main risk assessment agent

### Integration Examples
- `integration_example.py` - Shows how to use all agents together
- `usage_example.py` - Demonstrates usage with exact input format

## Usage

### Basic Usage

```python
from risk_overview_agent import RiskOverviewAgent

# Initialize agent
agent = RiskOverviewAgent(openai_api_key="your-key")  # Optional

# Input data in the specified format
input_data = {
    "location": {"name": "Chilliwack, BC", "lat": 49.16, "lon": -121.96},
    "time_utc": "2025-10-18T20:05Z",
    "signals": {
        "alerts": [...],
        "hydrology_forecast": {...},
        "precip": {...}
    }
}

# Generate risk assessment
risk_assessment = agent.generate_risk_overview(input_data)
```

### Integration with Other Agents

```python
from integration_example import FloodRiskIntegration

# Initialize integration
integration = FloodRiskIntegration(openai_api_key="your-key")

# Get comprehensive assessment
result = await integration.get_comprehensive_risk_assessment("Chilliwack, BC")
```

## Risk Assessment Logic

### Alert Analysis
- **Severe + Immediate**: High risk (confidence 0.8)
- **Moderate + Urgent**: Medium risk (confidence 0.7)
- **No alerts**: No alert-based risk

### Hydrology Analysis
- **Danger threshold (500 cms)**: Extreme risk
- **Warning threshold (430 cms)**: High risk  
- **Watch threshold (350 cms)**: Medium risk

### Precipitation Analysis
- **Extreme rain (50mm/6h, 100mm/24h)**: Extreme risk
- **Heavy rain (25mm/6h, 50mm/24h)**: High risk
- **Recent heavy rainfall (>50mm/72h)**: Elevated risk

### Conflict Resolution
- When data conflicts, explains which evidence is trusted most
- Considers the most severe indicators from any source
- Reduces confidence when data quality is low

## Testing

### Run Individual Agent
```bash
cd backend/app/services
python risk_overview_agent.py
```

### Run Integration Example
```bash
cd backend/app/services
python integration_example.py
```

### Run Usage Example
```bash
cd backend/app/services
python usage_example.py
```

## Sample Output

For the provided input data, the agent correctly identifies:

- **Risk Level**: HIGH
- **Confidence**: 0.8
- **Key Drivers**:
  - Active severe flood alerts with immediate urgency
  - River discharge (480 cms) exceeds danger threshold
  - Heavy precipitation forecast (82mm in 24h)

## Dependencies

- `openai` (optional, for AI-powered analysis)
- `aiohttp` (for API calls)
- `asyncio` (for async operations)
- Standard library modules: `json`, `logging`, `datetime`, `typing`

## Configuration

The agent can be configured with:
- OpenAI API key for AI-powered analysis
- Custom thresholds for risk levels
- Data source preferences

## Error Handling

- Graceful fallback to rule-based analysis when AI is unavailable
- Robust error handling for missing or invalid data
- Comprehensive logging for debugging

## Future Enhancements

- Additional data sources (satellite imagery, social media)
- Machine learning models for improved accuracy
- Real-time data streaming
- Historical risk trend analysis

