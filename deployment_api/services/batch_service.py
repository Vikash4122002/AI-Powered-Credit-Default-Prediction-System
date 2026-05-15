from services.prediction_service import (
    predict_default
)

from utils.logger import (
    logger
)

def batch_predict(

    customers,

    threshold=0.3
):

    """
    Batch prediction
    for multiple customers
    """

    results = []

    success_count = 0

    failed_count = 0

    default_count = 0

    for idx, customer in enumerate(customers):

        try:

            # Predict customer

            result = predict_default(

                customer,

                threshold
            )

            # Add customer ID

            result['customer_id'] = idx

            # Count defaults

            if result['prediction'] == 1:

                default_count += 1

            success_count += 1

            results.append(result)

        except Exception as e:

            failed_count += 1

            logger.error(

                f"Prediction failed "

                f"for customer {idx}: "

                f"{str(e)}"
            )

            # Error response

            results.append({

                'customer_id':
                idx,

                'prediction':
                None,

                'risk':
                'Error',

                'probability':
                0.0,

                'risk_level':
                'ERROR',

                'threshold_used':
                threshold
            })

    # Final batch log

    logger.info(

        f"Batch prediction completed | "

        f"Total={len(customers)} | "

        f"Success={success_count} | "

        f"Failed={failed_count} | "

        f"Defaults={default_count}"
    )

    return {

        "total_customers":
        len(customers),

        "successful_predictions":
        success_count,

        "failed_predictions":
        failed_count,

        "predicted_defaults":
        default_count,

        "threshold_used":
        threshold,

        "results":
        results
    }