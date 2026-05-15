from fastapi import (
    APIRouter,
    HTTPException
)

from schema.prediction_schema import (
    PredictionInput,
    PredictionResponse
)

from services.prediction_service import (
    predict_default
)

from utils.logger import (
    logger
)  

router = APIRouter()
@router.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionInput):
    
    try:


        result = predict_default(
            data.features,
            data.threshold
        )
        logger.info(
            f"Prediction successful: "
            f"{result['prediction']}"
        )
        return result

    except Exception as e:
        logger.error(
            f"Prediction failed: "
            f"{str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
