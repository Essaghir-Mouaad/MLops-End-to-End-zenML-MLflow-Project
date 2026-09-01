import logging
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformation
        """
        pass


# Concrete Strategy for Log Transformation
# ------------------------------------
# This strategy applies log transformation to features, which is useful for skewed data.


class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(
            f"Applying the log transformation on the features: {self.features}"
        )
        df_transformed = df.copy()

        for col in self.features:
            df_transformed[col] = np.log1p(df_transformed[col])

        logging.info("Log transformation applied successfully")
        return df_transformed


# Concrete Strategy for Standard Scaling
# -------------------------------------
# This strategy applies standard scaling to features, scaling them to have a mean of 0 and a variance of 1.


class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying the standard scaling on the features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(
            df_transformed[self.features]
        )
        logging.info("Standard scaling applied successfully")
        return df_transformed


# Concrete Strategy for Min-Max Scaling
# -------------------------------------
# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].


class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = MinMaxScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying the min-max scaling on the features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(
            df_transformed[self.features]
        )
        logging.info("Min-max scaling applied successfully")
        return df_transformed


# Concrete Strategy for One-Hot Encoding
# -------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them into a binary vector representation.


class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.encoder = OneHotEncoder(sparse_output=False)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying the one-hot encoding on the features: {self.features}")
        df_transformed = df.copy()
        encode_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )

        df_transformed = df_transformed.drop(columns=self.features)
        df_transformed = pd.concat([df_transformed, encode_df], axis=1)
        logging.info("One-hot encoding applied successfully")
        return df_transformed


class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        self.strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Feature engineering started....")
        return self.strategy.apply_transformation(df)


# Example usage
if __name__ == "__main__":
    # # Example dataframe
    # df = pd.read_csv('/home/mouaad/Desktop/Projects/Learn/MLops/house_price_prediction/extract_path/AmesHousing.csv')

    # # Log Transformation Example
    # log_transformer = FeatureEngineer(LogTransformation(features=['SalePrice', 'Gr Liv Area']))
    # df_log_transformed = log_transformer.apply_feature_engineering(df)
    # print(df_log_transformed.head())
    # # Standard Scaling Example
    # standard_scaler = FeatureEngineer(StandardScaling(features=['SalePrice', 'Gr Liv Area']))
    # df_standard_scaled = standard_scaler.apply_feature_engineering(df)
    # print(df_standard_scaled.head())
    # # Min-Max Scaling Example
    # minmax_scaler = FeatureEngineer(MinMaxScaling(features=['SalePrice', 'Gr Liv Area']))
    # df_minmax_scaled = minmax_scaler.apply_feature_engineering(df)
    # print(df_minmax_scaled.head())
    # # One-Hot Encoding Example
    # onehot_encoder = FeatureEngineer(OneHotEncoding(features=['Neighborhood']))
    # df_onehot_encoded = onehot_encoder.apply_feature_engineering(df)
    # print(df_onehot_encoded.head())
    pass
