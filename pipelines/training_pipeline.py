import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step

# pyrefly: ignore [missing-import]
from zenml import pipeline, Model


@pipeline(
    model=Model(
        name="house_price_prediction",
    )
)
def ml_piepline(file_path: str):
    """
    This pipeline ingest the data from the file path and preprocess it

    Args:
        file_path (str): The path to the data file

    Returns:
        pd.DataFrame: The preprocessed data
    """
    # Data Ingestion
    raw_data = data_ingestion_step(file_path=file_path)

    # Handling Missing Values
    clean_data = handle_missing_values_step(raw_data)
