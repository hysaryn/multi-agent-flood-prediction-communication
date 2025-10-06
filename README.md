# Multi-Agent Flood Prediction Communication System

A multi-agent system for flood prediction and risk communication that integrates:
- RAG (Retrieval-Augmented Generation) for official guide content
- Microsoft Weather Service API for real-time data
- Social media feeds for community insights
- Intelligent chatbot for integrated information delivery

## Project Structure

```
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py         # FastAPI application
│   │   ├── models/         # Data models
│   │   ├── services/       # Business logic
│   │   │   ├── rag_service.py
│   │   │   ├── weather_service.py
│   │   │   └── social_media_service.py
│   │   ├── agents/         # Multi-agent components
│   │   └── api/           # API routes
│   ├── requirements.txt
│   └── docs/              # Local documents for RAG
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── OfficialGuide/
│   │   │   ├── Live/
│   │   │   ├── SocialMedia/
│   │   │   └── Chatbot/
│   │   ├── services/
│   │   └── App.js
│   ├── package.json
│   └── public/
└── README.md
```

## Features

- **OfficialGuide**: RAG-powered content from local documents
- **Live**: Real-time weather data from Microsoft Weather API
- **Social Media**: Social media feed summarization
- **Chatbot**: Integrated information from all sources

## Getting Started

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```
