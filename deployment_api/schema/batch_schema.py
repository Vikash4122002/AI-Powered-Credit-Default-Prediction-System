from pydantic import (

    BaseModel,

    Field
)

from typing import (

    List,

    Optional
)

class BatchPredictionInput(BaseModel):

    """Batch prediction input schema"""

    customers: List[List[float]] = Field(

        ...,

        description=
        "List of customers, each with 23 features",

        min_length=1
    )

    threshold: float = Field(

        0.3,

        ge=0.0,

        le=1.0,

        description=
        "Risk threshold"
    )

    model_config = {

        "json_schema_extra": {

            "example": {

                "customers": [

                    [

                        20000,
                        2,
                        2,
                        1,
                        24,
                        2,
                        2,
                        -1,
                        -1,
                        -2,
                        -2,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        689,
                        0,
                        0,
                        0,
                        0,
                        0
                    ],

                    [

                        50000,
                        1,
                        2,
                        2,
                        35,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        10000,
                        5000,
                        3000,
                        2000,
                        1000,
                        500,
                        2000,
                        1000,
                        500,
                        300,
                        200,
                        100
                    ]
                ],

                "threshold": 0.3
            }
        }
    }

class BatchResultItem(BaseModel):

    """Single batch result"""

    customer_id: int

    prediction: Optional[int]

    risk: str

    probability: float

    risk_level: str

    threshold_used: float

class BatchPredictionResponse(BaseModel):

    """Batch prediction response"""

    total_customers: int

    successful_predictions: int

    failed_predictions: int

    predicted_defaults: int

    results: List[BatchResultItem]

    model_config = {

        "json_schema_extra": {

            "example": {

                "total_customers": 2,

                "successful_predictions": 2,

                "failed_predictions": 0,

                "predicted_defaults": 1,

                "results": [

                    {

                        "customer_id": 0,

                        "prediction": 1,

                        "risk": "Default",

                        "probability": 0.8234,

                        "risk_level": "HIGH",

                        "threshold_used": 0.3
                    }
                ]
            }
        }
    }