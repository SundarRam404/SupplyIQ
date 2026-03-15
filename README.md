# SupplyIQ Website

A recruiter-grade split deployment for your SupplyIQ project:
- `frontend/` → Next.js website for Vercel
- `backend/` → FastAPI API for Render

## What it includes
- Demand Forecasting
- Demand Risk Detection
- Contract Intelligence
- Supply Chain Decision Engine
- Scenario Simulator
- Executive KPI Dashboard
- Premium dark glassmorphism UI with hero illustration

## Local run

### Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
cp .env.example .env.local
# set NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm run dev
```

## Deploy

### Render
- Create a new Web Service from the `backend` folder
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Add env var `GROQ_API_KEY`

### Vercel
- Import the project and set root directory to `frontend`
- Add env var `NEXT_PUBLIC_API_BASE_URL=https://your-render-service.onrender.com`
- Deploy
