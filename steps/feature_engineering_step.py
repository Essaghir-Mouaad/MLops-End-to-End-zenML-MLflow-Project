import pandas as pd
from src.featur_engineering import (
    LogTransformation,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    FeatureEngineer,
)

#
# pyrefly: ignore [missing-import]
from zenml import step


@step
def feature_engineering_step(
    df: pd.DataFrame, strategy: str = "log", features: list = None
) -> pd.DataFrame:

    if features is None:
        features = []

    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineer(StandardScaler(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxScaler(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncoder(features))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    df_transformed = engineer.apply_feature_engineering(df)
    return df_transformed
