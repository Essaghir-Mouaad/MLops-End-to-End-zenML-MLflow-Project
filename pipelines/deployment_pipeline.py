import os

from pipelines.training_pipeline import ml_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.prediction_server_loader import prediction_server_loader
from steps.predictor import predictor

# pyrefly: ignore [missing-import]
from zenml import pipeline

# pyrefly: ignore [missing-import]
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@pipeline(enable_cache=False)
def continuous_deployment_pipeline(workers: int = 3):
    trained_model = ml_pipeline(file_path="/home/mouaad/Desktop/Projects/Learn/MLops/house_price_prediction/extract_path/AmesHousing.csv")

    # deployed the trained model

    mlflow_model_deployer_step(workers=workers, deploy_decision=True, model=trained_model)


@pipeline(enable_cache=False)
def inference_pipeline():

    # load the batch data for inference
    batch_data = dynamic_importer()

    # Load the deployed model from the model registwr y

    model_deployment_service = prediction_server_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )

    # Run Prediction
    predictor(service=model_deployment_service, data=batch_data)
