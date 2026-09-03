from pipelines.training_pipeline import ml_pipeline
# pyrefly: ignore [missing-import]
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


if __name__ == "__main__":
    ml_pipeline(
        file_path="/home/mouaad/Desktop/Projects/Learn/MLops/house_price_prediction/extract_path/AmesHousing.csv"
    )

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )

