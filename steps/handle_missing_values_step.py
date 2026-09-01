import pandas  as pd 
from src.handle_missing_values import (
    DropMissingValuesStrategy,
    FillMissingValuesStrategy,
    MissingValueHandler
)

# pyrefly: ignore [missing-import]
from zenml import step

@step

def handle_missing_values_step(df: pd.DataFrame, strategy: str ="mean") -> pd.DataFrame:
    """
    handle the missing values in the dataframe 
    """
    if strategy == "drop":
        handler = MissingValueHandler(
        strategy=DropMissingValuesStrategy(
            axis=0,
            thresh=None
        )
    )
    elif strategy in ["mean", "median", "mode", "constant"]:
        handler = MissingValueHandler(
            strategy=FillMissingValuesStrategy(
                method=strategy,
                fill_value=None
            )
        )
    else:
        raise ValueError("Invalid strategy")
    
    return handler.handle_missing_values(df) 
    

