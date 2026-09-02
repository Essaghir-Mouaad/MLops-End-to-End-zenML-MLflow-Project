import logging
import pandas as pd
from typing import Tuple

from sklearn.pipeline import Pipeline
from src.model_evaluator import ModelEvaluator, RegressionEvaluatorStrategy

# pyrefly: ignore [missing-import]
from zenml import step


@step
def model_evaluator_step(
    trained_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[dict, float]:
    """
    Evaluates the model and returns the evaluation metrics.

    Args:
        trained_model: The trained model.
        X_test: The test features.
        y_test: The test labels.

    Returns:
        A tuple containing the trained model and the evaluation metrics.
    """

    # ensure the inputs are valids

    if not isinstance(trained_model, Pipeline):
        raise TypeError("trained_model must be a Pipeline")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a DataFrame")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a Series")

    logging.info("Applying the same preprocessing on the test set")

    X_test_precessed = trained_model.named_steps["preprocess"].transform(X_test)

    y_true = y_test

    evaluator = ModelEvaluator(RegressionEvaluatorStrategy())

    evaluation_metrics = evaluator.evaluate(
        X_test_precessed, y_true, trained_model.named_steps["regressor"]
    )

    if not isinstance(evaluation_metrics, dict):
        raise TypeError("evaluation_metrics must be a dictionary")

    mse = evaluation_metrics["Mean Square Error"]
    # r2 = evaluation_metrics['R2 Score']

    return evaluation_metrics, mse
  