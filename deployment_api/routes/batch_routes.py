from fastapi import (
    APIRouter,
    HTTPException
)

from schema.batch_schema import (
    BatchPredictionInput,
    BatchPredictionResponse
)

from services.batch_service import (
    batch_predict
)

from utils.logger import (
    logger
)

router = APIRouter()

@router.post(
    "/predict_batch",
    response_model=BatchPredictionResponse
)

def predict_batch(
    data: BatchPredictionInput
):

    try:

        # Call batch service

        batch_result = batch_predict(

            data.customers,

            data.threshold
        )

        # Log success

        logger.info(

            f"Batch prediction completed | "

            f"Total={batch_result['total_customers']}"
        )

        # Return response

        return BatchPredictionResponse(

            total_customers=
            batch_result[
                'total_customers'
            ],

            successful_predictions=
            batch_result[
                'successful_predictions'
            ],

            failed_predictions=
            batch_result[
                'failed_predictions'
            ],

            predicted_defaults=
            batch_result[
                'predicted_defaults'
            ],

            results=
            batch_result[
                'results'
            ]
        )

    except Exception as e:

        logger.error(

            f"Batch prediction failed: "

            f"{str(e)}"
        )

        raise HTTPException(

            status_code=500,

            detail=str(e)
        )