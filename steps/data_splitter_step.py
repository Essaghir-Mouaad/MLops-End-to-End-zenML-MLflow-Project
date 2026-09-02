from typing import Tuple
import pandas as pd

from src.data_splitter import DataSplittingContext, SimpletrainTestSplitStrategy

# pyrefly: ignore [missing-import]
from zenml import step


@step
def data_splitting_step(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into training and testing sets"""

    splitter = DataSplittingContext(SimpletrainTestSplitStrategy())

    X_train, X_test, y_train, y_test = splitter.split(df, target_column)

    return X_train, X_test, y_train, y_test
