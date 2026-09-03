import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineering_step
from steps.outlier_detection_step import outlier_detection_step
from steps.model_building_step import model_building_step
from steps.data_splitter_step import data_splitting_step
from steps.model_evaluator_step import model_evaluator_step

# pyrefly: ignore [missing-import]
from zenml import pipeline, Model


@pipeline(
    model=Model(
        name="house_price_prediction",
    )
)
def ml_pipeline(file_path: str):
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

    # Feature Engineering
    engineered_data = feature_engineering_step(
        clean_data, strategy="log", features=["Gr Liv Area", "SalePrice"]
    )

    # Outlier Detection
    clean_data = outlier_detection_step(engineered_data, column_name="SalePrice")

    # the data splitting

    X_train, X_test, y_train, y_test = data_splitting_step(
        df=clean_data, target_column="SalePrice"
    )

    # model building

    model = model_building_step(X_train=X_train, y_train=y_train)

    # model evaluation

    evaluation_metrics, mse = model_evaluator_step(
        X_test=X_test, y_test=y_test, trained_model=model
    )


    return model