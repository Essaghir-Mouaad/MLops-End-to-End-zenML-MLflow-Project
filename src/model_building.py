from logging import info
import logging
import pandas as pd
from abc import ABC, abstractmethod

from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# setup logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Abstract base class


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(
        self, X_train: pd.DataFrame, y_traib: pd.Series
    ) -> RegressorMixin:
        pass


class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:

        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a DataFrame")
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a Series")
        if X_train.isnull().values.any() or y_train.isnull().values.any():
            raise ValueError("X_train and y_train must not contain missing values")

        logging.info("Initailizing The LinearRegression Model .....")
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )

        logging, info("Training Linear Regression Model ....")
        pipeline.fit(X_train, y_train)

        logging.info("Linear Regression Model training completed")

        return pipeline


class ModelBuildingContext:
    def __init__(self, strategy: ModelBuildingStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: ModelBuildingStrategy):
        logging.info("Setting new strategy .... ")
        self.strategy = strategy
        logging.info("Strategy changed successfully .....")

    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        logging.info("Building and training model .... ")
        return self.strategy.build_and_train_model(X_train, y_train)


# # Example usage
# if __name__ == "__main__":
#     # Example DataFrame (replace with actual data loading)
#     df = pd.read_csv("/home/mouaad/Desktop/Projects/Learn/MLops/house_price_prediction/extract_path/AmesHousing.csv")
#     X_train = df.drop(columns=['SalePrice'])
#     y_train = df['SalePrice']

#     # Example usage of Linear Regression Strategy
#     model_builder = ModelBuildingContext(LinearRegressionStrategy())
#     trained_model = model_builder.build_model(X_train, y_train)
#     print(trained_model.named_steps['model'].coef_)  # Print model coefficients

#     pass
