from src.Utils import make_predictions, get_results
from loguru import logger

def predict_model():
    predictions = make_predictions()
    accuracy, classification_report = get_results(predictions)
    logger.info("Finished making predictions, accuracy = {}".format(accuracy))

    return accuracy, classification_report