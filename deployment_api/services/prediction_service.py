from utils.load_model import (
    model,
    scaler
)
from utils.logger import logger
from utils.preprocess import preprocess_features

def get_risk_level(probability: float):

    if probability > 0.5:
        return "HIGH"

    elif probability > 0.3:
        return "MEDIUM"

    return "LOW"

def predict_default(features, threshold=0.3):

    try:

        if model is None:
            raise ValueError("Model not loaded")

        arr = preprocess_features(features)

        arr_scaled = scaler.transform(arr)


        prob = model.predict_proba(arr_scaled)[0, 1]

        pred = 1 if prob > threshold else 0

        logger.info(
            f"Prediction={pred} | "
            f"Probability={prob:.4f}"
        )

        return {

            "prediction":
            int(pred),

            "risk":
            "Default"
            if pred == 1
            else "No Default",

            "probability":
            round(float(prob), 4),

            "risk_level":
            get_risk_level(prob),

            "threshold_used":
            threshold
        }

    except Exception as e:

        logger.error(
            f"Prediction error: {str(e)}"
        )

        raise