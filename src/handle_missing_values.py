import pandas as pd
import logging
from abc import ABC, abstractmethod


# setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        self.axis = axis
        self.thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(
            f"Dropping missing values with axis {self.axis} and thresh {self.thresh}"
        )
        df_cleaned = df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped")
        return df_cleaned


class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method: str, fill_value: float):
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Filling Missing values using the method {self.method}")
        df_cleaned = df.copy()

        if self.method == "mean":
            numerical_cols = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numerical_cols] = df_cleaned[numerical_cols].fillna(
                df_cleaned[numerical_cols].mean()
            )
        elif self.method == "median":
            numerical_cols = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numerical_cols] = df_cleaned[numerical_cols].fillna(
                df_cleaned[numerical_cols].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column] = df_cleaned[column].fillna(
                    df_cleaned[column].mode().iloc[0], inplace=True
                )
        else:
            logging.warning(f"Unknown method {self.method}, using fill_value")
            df_cleaned = df_cleaned.fillna(self.fill_value)

        logging.info("Missing values handled")
        return df_cleaned


class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        self.strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("handling Missing values ....")
        return self.strategy.handle(df)


# Example usage
if __name__ == "__main__":
    # # Example dataframe
    # df = pd.read_csv('/home/mouaad/Desktop/Projects/Learn/MLops/house_price_prediction/extract_path/AmesHousing.csv')

    # # Initialize missing value handler with a specific strategy
    # missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0, thresh=3))
    # df_cleaned = missing_value_handler.handle_missing_values(df)

    # # Switch to filling missing values with mean
    # missing_value_handler.set_strategy(FillMissingValuesStrategy(method='mean', fill_value=None))
    # df_filled = missing_value_handler.handle_missing_values(df)
    # print(df_cleaned.head())
    # print(df_filled.describe())
    pass
