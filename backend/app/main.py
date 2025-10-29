from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from app.services.rag_service import RAGService
from app.services.weather_service import WeatherService
from app.services.social_media_service import SocialMediaService

app = FastAPI(title="Flood Prediction Communication System", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rag_service = RAGService()
weather_service = WeatherService()
social_media_service = SocialMediaService()

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    location: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    confidence: float

class OfficialGuideResponse(BaseModel):
    content: str
    sources: List[str]

class LiveDataResponse(BaseModel):
    weather_data: dict
    flood_risk: str
    recommendations: List[str]

class SocialMediaResponse(BaseModel):
    summary: str
    sentiment: str
    key_topics: List[str]

class FloodAlertResponse(BaseModel):
    us_alerts: dict
    canada_alerts: dict
    summary: dict

@app.get("/")
async def root():
    return {"message": "Flood Prediction Communication System API"}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(message: ChatMessage):
    """Main chatbot endpoint that integrates all information sources"""
    try:
        # Get information from all sources
        official_guide = await rag_service.get_relevant_content(message.message)
        live_data = await weather_service.get_current_conditions(message.location)
        social_summary = await social_media_service.get_summary()
        
        # TODO: Integrate all sources using LLM
        response = f"Based on official guidelines, current conditions, and social media insights: {message.message}"
        
        return ChatResponse(
            response=response,
            sources=["official_guide", "weather_api", "social_media"],
            confidence=0.85
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/official-guide", response_model=OfficialGuideResponse)
async def get_official_guide(query: str = ""):
    """Get official guide content using RAG"""
    try:
        content, sources = await rag_service.get_relevant_content(query)
        return OfficialGuideResponse(content=content, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/live", response_model=LiveDataResponse)
async def get_live_data(location: str = "Hope, BC"):
    """Get live weather data and flood risk assessment"""
    try:
        weather_data = await weather_service.get_current_conditions(location)
        flood_risk = await weather_service.assess_flood_risk(weather_data)
        recommendations = await weather_service.get_recommendations(flood_risk)
        
        return LiveDataResponse(
            weather_data=weather_data,
            flood_risk=flood_risk,
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/social-media", response_model=SocialMediaResponse)
async def get_social_media_summary():
    """Get social media feed summary"""
    try:
        summary, sentiment, topics = await social_media_service.get_summary()
        return SocialMediaResponse(
            summary=summary,
            sentiment=sentiment,
            key_topics=topics
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/flood-alerts", response_model=FloodAlertResponse)
async def get_government_flood_alerts(
    us_states: str = None,
    canada_provinces: str = None
):
    """Get live flood alerts from government sources (US and Canada)"""
    try:
        # Parse location parameters
        us_locations = None
        canada_locations = None
        
        if us_states:
            # Parse comma-separated state:county pairs
            us_locations = []
            for location in us_states.split(','):
                if ':' in location:
                    state, county = location.split(':', 1)
                    us_locations.append((state.strip().upper(), county.strip()))
                else:
                    us_locations.append((location.strip().upper(), None))
        
        if canada_provinces:
            # Parse comma-separated province:city pairs
            canada_locations = []
            for location in canada_provinces.split(','):
                if ':' in location:
                    province, city = location.split(':', 1)
                    canada_locations.append((province.strip().upper(), city.strip()))
                else:
                    canada_locations.append((location.strip().upper(), None))
        
        # Get flood alerts
        alerts_data = await weather_service.get_government_flood_alerts(
            us_locations=us_locations,
            canada_locations=canada_locations
        )
        
        if "error" in alerts_data:
            raise HTTPException(status_code=500, detail=alerts_data["error"])
        
        return FloodAlertResponse(
            us_alerts=alerts_data.get("us", {}),
            canada_alerts=alerts_data.get("canada", {}),
            summary=alerts_data.get("summary", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/flood-alerts/display")
async def get_flood_alerts_display(
    us_states: str = None,
    canada_provinces: str = None
):
    """Get formatted flood alerts for display"""
    try:
        # Parse location parameters
        us_locations = None
        canada_locations = None
        
        if us_states:
            us_locations = []
            for location in us_states.split(','):
                if ':' in location:
                    state, county = location.split(':', 1)
                    us_locations.append((state.strip().upper(), county.strip()))
                else:
                    us_locations.append((location.strip().upper(), None))
        
        if canada_provinces:
            canada_locations = []
            for location in canada_provinces.split(','):
                if ':' in location:
                    province, city = location.split(':', 1)
                    canada_locations.append((province.strip().upper(), city.strip()))
                else:
                    canada_locations.append((location.strip().upper(), None))
        
        # Get flood alerts
        alerts_data = await weather_service.get_government_flood_alerts(
            us_locations=us_locations,
            canada_locations=canada_locations
        )
        
        if "error" in alerts_data:
            return {"error": alerts_data["error"]}
        
        # Format for display
        formatted_output = weather_service.flood_alert_agent.format_alerts_for_display(alerts_data)
        
        return {"formatted_alerts": formatted_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
