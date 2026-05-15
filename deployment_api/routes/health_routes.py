from fastapi import APIRouter
from datetime import datetime
from utils.load_model import model

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

@router.get("/readiness")
def readiness():
    return {
        "ready": model is not None,
        "timestamp": datetime.now().isoformat()
    }