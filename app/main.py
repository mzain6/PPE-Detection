#python app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import cameras, detections, stream
from app.routers import ws_detections
from app.services.worker import worker

app = FastAPI(
    title="PPE Detection API",
    description="Enterprise-ready MVP API for PPE and fall detection (DB-agnostic).",
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(cameras.router)
app.include_router(detections.router)
app.include_router(stream.router)
app.include_router(ws_detections.router)

@app.on_event("startup")
def startup_event():
    worker.start()

@app.on_event("shutdown")
def shutdown_event():
    worker.stop()