import pandas as pd
from src.ingest_data import DataIngestorFactory

# pyrefly: ignore [missing-import]
from zenml import step
import os

# pyrefly: ignore [missing-import]


@step
def data_ingestion_step(file_path: str) -> pd.DataFrame:
    """ingest the data from the file path using the data ingestor factory"""

    file_extension = os.path.splitext(file_path)[1]
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    data = data_ingestor.ingest(file_path)
    return data
