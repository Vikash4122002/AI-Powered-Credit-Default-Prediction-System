from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware
)

from routes.prediction_routes import (
    router as prediction_router
)

from routes.batch_routes import (
    router as batch_router
)

from routes.health_routes import (
    router as health_router
)

# Create FastAPI app

app = FastAPI(

    title=
    "Credit Default Prediction API",

    description=
    "Stacking Ensemble ML API "
    "for Credit Card Default Prediction",

    version="2.0.0",

    docs_url="/docs",

    redoc_url="/redoc"
)

# Enable CORS

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

# Include routes

app.include_router(

    prediction_router,

    prefix="/api/v1",

    tags=["Prediction"]
)

app.include_router(

    batch_router,

    prefix="/api/v1",

    tags=["Batch Prediction"]
)

app.include_router(

    health_router,

    tags=["Health"]
)

# Home route

@app.get("/")

def home():

    return {

        "message":

        "Credit Default Prediction API Running",

        "version":

        "2.0.0",

        "status":

        "active"
    }