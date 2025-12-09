from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

from app.services.rag_service import RAGService
from app.services.weather_service import WeatherService
from app.services.social_media_service import SocialMediaService
from .services.location_service import get_location_info
from app.services.predictor_service import ValidatedFloodPredictor

from .api import predict


app = FastAPI(title="Flood Prediction Communication System", version="1.0.0")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)

# Initialize services
rag_service = RAGService(docs_path="Alert Guides Docs")
weather_service = WeatherService()
social_media_service = SocialMediaService()
predictor_service = ValidatedFloodPredictor(data_dir=DATA_DIR)

@app.get("/locate")
async def locate(q: str):
    """
    Example: /locate?q=Hope, BC  或 /locate?q=49.377,-121.441
    """
    try:
        result = await get_location_info(q)
        return result.model_dump()  # 如果是 Pydantic 模型
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Models ----------
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

class RAGAsk(BaseModel):
    query: str

# ---------- Routes ----------
@app.get("/")
async def root():
    return {"message": "Flood Prediction Communication System API"}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(message: ChatMessage):
    """Main chatbot endpoint integrating all data sources."""
    try:
        content, _ = await rag_service.get_relevant_content(message.message or "")
        response = f"Based on official guidelines: {content[:160]}..."
        return ChatResponse(
            response=response,
            sources=["official_guide"],
            confidence=0.85
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/official-guide", response_model=OfficialGuideResponse)
async def get_official_guide(query: str = ""):
    """Retrieve official guide content using RAG."""
    try:
        q = query or "What to prepare 7 days before a flood in BC?"
        content, sources = await rag_service.get_relevant_content(q)
        return OfficialGuideResponse(content=content, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/ask", response_model=OfficialGuideResponse)
async def rag_ask(req: RAGAsk):
    """Query the RAG engine with a question."""
    try:
        content, sources = await rag_service.get_relevant_content(req.query)
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
