import numpy as np

from typing import List

def preprocess_features(
    features: List[float]
) -> np.ndarray:

    arr = np.array(
        features,
        dtype=np.float64
    )

    # Check invalid values

    if (
        np.any(np.isnan(arr))
        or
        np.any(np.isinf(arr))
    ):

        raise ValueError(
            "Features contain invalid values"
        )

    # Reshape for model

    arr = arr.reshape(1, -1)

    return arr

def validate_feature_count(
    features: List[float]
) -> bool:

    return len(features) == 23

def get_feature_summary(
    features: List[float]
) -> dict:

    return {

        "mean":
        float(np.mean(features)),

        "std":
        float(np.std(features)),

        "min":
        float(np.min(features)),

        "max":
        float(np.max(features))
    }