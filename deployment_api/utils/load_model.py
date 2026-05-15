import joblib

from pathlib import Path

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

model = None

scaler = None

def load_artifacts():

    global model

    global scaler

    try:

        base_path = (

            Path(__file__)

            .resolve()

            .parent

            .parent

            / 'models'
        )

        model_path = (

            base_path

            / 'stacking_optuna_meta.pkl'
        )

        scaler_path = (

            base_path

            / 'scaler.pkl'
        )

        # Load model

        model = joblib.load(model_path)

        # Load scaler

        scaler = joblib.load(scaler_path)

        logger.info(
            "✅ Model loaded successfully"
        )

        logger.info(
            "✅ Scaler loaded successfully"
        )

        return model, scaler

    except Exception as e:

        logger.error(

            f"❌ Failed loading artifacts: {e}"
        )

        raise

try:

    model, scaler = load_artifacts()

except Exception as e:

    logger.warning(

        f"Artifacts not loaded: {e}"
    )

    model = None

    scaler = None