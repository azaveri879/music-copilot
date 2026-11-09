import os

from dotenv import load_dotenv

# Must run before the route imports, which read configuration at import time.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.analyze import router as analyze_router
from routes.chat import router as chat_router

app = FastAPI(
    title="Music Copilot API",
    description="AI-powered production assistant for Logic Pro",
    version="0.1.0",
)

# CORS — allow the React dev server and production frontend
origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Edit-Report"],
)

app.include_router(analyze_router)
app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}
