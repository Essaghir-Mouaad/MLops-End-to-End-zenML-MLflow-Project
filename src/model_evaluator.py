import logging
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

# setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# abstract Base Class for Model Evaluation Starategy


class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self, X_test: pd.DataFrame, y_true: pd.Series, model: RegressorMixin
    ) -> dict:
        pass


class RegressionEvaluatorStrategy(ModelEvaluationStrategy):
    def evaluate_model(
        self, X_test: pd.DataFrame, y_true: pd.DataFrame, model: RegressorMixin
    ) -> dict:
        try:
            logging.info("Starting model predciton ..")
            y_pred = model.predict(X_test)

            logging.info("Calculating the evaluation metrics...")
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            metrics = {"Mean Square Error": mse, "R2 Score": r2}

            logging.info("Evaluation metrics calculated successfully.")
            return metrics
        except Exception as e:
            logging.error(f"Error during model evaluation: {str(e)}")
            raise


# Context class for model evaluation


class ModelEvaluator:
    def __init__(self, strategy: ModelEvaluationStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: ModelEvaluationStrategy):
        self.strategy = strategy

    def evaluate(
        self, X_test: pd.DataFrame, y_true: pd.Series, model: RegressorMixin
    ) -> dict:
        logging.info(f"Evaluating model using selected strategy")
        return self.strategy.evaluate_model(X_test, y_true, model)
