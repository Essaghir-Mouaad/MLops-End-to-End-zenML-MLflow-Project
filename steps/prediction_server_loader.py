# pyrefly: ignore [missing-import]
from zenml import step

# pyrefly: ignore [missing-import]
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer

# pyrefly: ignore [missing-import]
from zenml.integrations.mlflow.services import MLFlowDeploymentService

@step(enable_cache=False)
def prediction_server_loader(
    pipeline_name: str, step_name: str
) -> MLFlowDeploymentService:

    # get the model deployer from the mlflow integration
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch the services with the correct pipeline nad step name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name, pipeline_step_name=step_name
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{step_name} step in the {pipeline_name} "
            f"pipeline is currently "
            f"running."
        )

    return existing_services[0]
