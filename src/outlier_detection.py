import pandas as pd
import numpy as np
import seaborn as sns
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import logging

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OutlierDetectionStrategy(ABC):
    """
    Abstract base class for outlier detection strategies
    """

    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers in the DataFrame

        Args:
            df (pd.DataFrame): The DataFrame to detect outliers in

        Returns:
            pd.DataFrame: The DataFrame with outliers detected
        """
        pass


class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(
            f"Detecting outliers using z-score with threshold {self.threshold}"
        )
        z_score = np.abs((df - df.mean()) / df.std())
        outliers = z_score > self.threshold
        logging.info(f"Found {outliers.sum()} outliers")
        return outliers


class IQROutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=1.5):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Detecting outliers using IQR with threshold {self.threshold}")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR
        outliers = df[(df < lower_bound) | (df > upper_bound)]
        logging.info("Outliers detected")
        return outliers


class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers...")
        return self.strategy.detect_outliers(df)

    def handle_outliers(
        self, df: pd.DataFrame, method: str = "remove", **kwargs
    ) -> pd.DataFrame:
        outliers = self.detect_outliers(df)

        if method == "remove":
            logging.info("Removing Outliers ...")
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method == "cap":
            logging.info("Capping Outliers ...")
            df_cleaned = df.clip(
                lower=outliers.quantile(0.01), upper=outliers.quantile(0.99)
            )
        else:
            logging.warning(f"Unkown method {method}")
            return df

        logging.info("Outliers handled successfully")
        return df_cleaned

    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info("Visualizing outliers...")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outliers visualized successfully")

# Example usage
if __name__ == "__main__":
    # # Example dataframe
    # df = pd.read_csv("/home/mouaad/Desktop/Projects/Learn/MLops/house_price_prediction/extract_path/AmesHousing.csv")
    # df_numeric = df.select_dtypes(include=[np.number]).dropna()

    # # Initialize the OutlierDetector with the Z-Score based Outlier Detection Strategy
    # outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))

    # # Detect and handle outliers
    # outliers = outlier_detector.detect_outliers(df_numeric)
    # df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")

    # print(df_cleaned.shape)
    # # Visualize outliers in specific features
    # # outlier_detector.visualize_outliers(df_cleaned, features=["SalePrice", "Gr Liv Area"])
    pass