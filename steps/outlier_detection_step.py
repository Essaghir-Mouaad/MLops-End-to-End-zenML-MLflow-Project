import logging
import pandas as pd
from src.outlier_detection import (
    OutlierDetector,
    ZScoreOutlierDetection,
    IQROutlierDetection,
)
# pyrefly: ignore [missing-import]
from zenml import step


@step
def outlier_detection_step(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    logging.info("Outlier detection started")

    if df is None:
        logging.error("Recieved a NoneType DataFrame")
        raise ValueError("Recieved a NoneType DataFrame")

    if not isinstance(df, pd.DataFrame):
        logging.error("Recieved a non-DataFrame object")
        raise TypeError("Recieved a non-DataFrame object")

    if column_name not in df.columns:
        logging.error(f"Column {column_name} not found in DataFrame")
        raise ValueError(f"Column {column_name} not found in DataFrame")

    df_numeric = df.select_dtypes(include=[int, float])

    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))
    outlier = outlier_detector.detect_outliers(df_numeric)
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method="remove")
    logging.info("Outlier detection completed")
    return df_cleaned
