# pyrefly: ignore [missing-import]
from zenml import step
import json
import pandas as pd 
# pyrefly: ignore [missing-import]
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import numpy as np 

@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: str
) -> np.ndarray:
    """
    Make predictions using the deployed MLflow service.
    """
    
    # Start the server if not running
    service.start(timeout=10)

    # Load the data JSON
    json_data = json.loads(data)

    # Convert to DataFrame using columns and data from orient='split' format
    if "columns" in json_data and "data" in json_data:
        df = pd.DataFrame(json_data["data"], columns=json_data["columns"])
    else:
        df = pd.DataFrame(json_data)

    # Make prediction using DataFrame
    prediction = service.predict(df)

    return prediction