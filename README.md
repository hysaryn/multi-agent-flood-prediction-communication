# Multi-Agent Flood Prediction Communication System

A comprehensive multi-agent system for flood prediction and risk communication that integrates real-time flood forecasting, government document analysis, and intelligent action plan generation.

## Getting Started

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your API keys
```

5. Start the FastAPI server:
```bash
uvicorn app.main:app --reload
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

## Experimental Modes

The system includes three experimental modes that were compared in the research report:

1. **Single Agent Mode** (`feature-single-agent` branch)
2. **Sequential Multi-Agent Mode** (`feature/multi-agent-sequential` branch)
3. **Dynamic Multi-Agent Mode** (`feature/multi-agent-loop-2` branch)

### Running Experimental Modes

To execute any of these experimental modes, use:

```bash
python -m app.action_plan_agent
```


## Data Files

The system uses several data files for gauge lookup and threshold calculation:

- `global_hybas_locations.csv`: Gauge locations for KDTree lookup
- `global_return_periods.csv`: Return period thresholds for severity assessment
- `BasinATLAS_v10_lev07.*`: Basin shapefiles for geographic matching


## Environment Variables

Key environment variables (see `backend/env.example`):

- `FLOODS_API_KEY`: Google Flood API key
- `OPENAI_API_KEY`: OpenAI API key for agents