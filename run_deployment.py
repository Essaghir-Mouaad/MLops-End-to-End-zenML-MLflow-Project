import os
import sys
# Prepend conda environment bin directory to PATH so MLflow sub-daemons (uvicorn) find the correct python environment
sys_executable_dir = os.path.dirname(sys.executable)
os.environ["PATH"] = f"{sys_executable_dir}:{os.environ.get('PATH', '')}"

from sqlalchemy import false
import click

# pyrefly: ignore [missing-import]
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from rich import print
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)

# pyrefly: ignore [missing-import]
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


@click.command()
@click.option(
    "--stop-service", is_flag=True, default=False, help="Stop the deployment server."
)
def run_main(stop_service: bool):
    """ """

    model_name = "prices_predictor"

    if stop_service:
        # get the mlflow model deployer stack
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        # fetch existing models
        existing_services = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            step_name="mlflow_model_deployer_step",
            model_name=model_name,
            running=True,
        )

        if existing_services:
            existing_services[0].stop(timeout=10)

        return

    continuous_deployment_pipeline(workers=3)

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # Run the inference pipeline
    inference_pipeline()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs."
    )

    # Fetch existing services with the same pipeline name, step name, and model name
    service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
    )

    if service[0]:
        print(
            f"The MLflow prediction server is running locally as a daemon "
            f"process and accepts inference requests at:\n"
            f"    {service[0].prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )


if __name__ == "__main__":
    run_main()
