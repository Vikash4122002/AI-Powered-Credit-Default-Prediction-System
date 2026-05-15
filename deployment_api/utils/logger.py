import logging
import os

# Create logs directory

log_dir = os.path.join(

    os.path.dirname(__file__),

    '..',

    'logs'
)

os.makedirs(

    log_dir,

    exist_ok=True
)

# Log file path

log_file = os.path.join(

    log_dir,

    'app.log'
)

# Configure logging

logging.basicConfig(

    filename=log_file,

    level=logging.INFO,

    format=
    '%(asctime)s | %(levelname)s | %(name)s | %(message)s',

    datefmt=
    '%Y-%m-%d %H:%M:%S'
)

# Create logger

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

# Prevent duplicate handlers

if not logger.handlers:

    # Console handler

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)

    console_handler.setFormatter(

        logging.Formatter(

            '%(asctime)s - '

            '%(levelname)s - '

            '%(message)s'
        )
    )

    logger.addHandler(
        console_handler
    )

def log_prediction(

    customer_id,

    prediction,

    probability
):

    """Log prediction details"""

    logger.info(

        f"Prediction | "

        f"customer_id={customer_id} | "

        f"prediction={prediction} | "

        f"probability={probability:.4f}"
    )