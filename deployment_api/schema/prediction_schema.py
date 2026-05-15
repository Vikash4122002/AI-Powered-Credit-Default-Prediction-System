from pydantic import BaseModel,Field,field_validator
from typing import List, Optional
class PredictionInput(BaseModel):

    features: List[float] = Field(
        ...,
        min_length=23,
        max_length=23,
        description="23 feature values"
    )

    threshold: Optional[float] = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Risk threshold"
    )

    @field_validator("features")

    @classmethod

    def validate_features(cls, v):

        ranges = {

            "LIMIT_BAL": (0, 1000000),
            "SEX": (1, 2),
            "EDUCATION": (1, 4),
            "MARRIAGE": (1, 3),
            "AGE": (18, 100),
            "PAY_0": (-2, 9),
            "PAY_2": (-2, 9),
            "PAY_3": (-2, 9),
            "PAY_4": (-2, 9),
            "PAY_5": (-2, 9),
            "PAY_6": (-2, 9),
        }

        for i, val in enumerate(v[:11]):

            name = list(ranges.keys())[i]

            min_val, max_val = ranges[name]

            if not (min_val <= val <= max_val):

                raise ValueError(

                    f"{name}={val} "
                    f"outside range "
                    f"({min_val}-{max_val})"
                )

        return v

    class Config:

        json_schema_extra = {

            "example": {

                "features": [

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

                "threshold": 0.3
            }
        }

class PredictionResponse(BaseModel):

    prediction: int = Field(
        ...,
        description="1=Default, 0=No Default"
    )

    risk: str = Field(
        ...,
        description="Default or No Default"
    )

    probability: float = Field(
        ...,
        description="Risk probability"
    )

    risk_level: str = Field(
        ...,
        description="HIGH/MEDIUM/LOW"
    )

    threshold_used: float = Field(
        ...,
        description="Threshold used"
    )

    class Config:

        json_schema_extra = {

            "example": {

                "prediction": 1,

                "risk": "Default",

                "probability": 0.7654,

                "risk_level": "HIGH",

                "threshold_used": 0.3
            }
        }