from typing import Annotated
import logging
import pandas as pd
import mlflow

from src.model_building import ModelBuildingContext, LinearRegressionStrategy
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# pyrefly: ignore [missing-import]
from zenml import ArtifactConfig, step, Model

# pyrefly: ignore [missing-import]
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker

model = Model(
    name="price_predictor",
    version=None,
    license="Apache 2.0",
    description="Predicting house prices",
)


@step(experiment_tracker=experiment_tracker.name, model=model, enable_cache=False)
def model_building_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Annotated[
    Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)
]:
    """
    Train a model using the given data.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        The trained pipeline.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a DataFrame")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a Series")
    if X_train.isnull().values.any() or y_train.isnull().values.any():
        raise ValueError("X_train and y_train must not contain missing values")

    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_features = X_train.select_dtypes(include=[int, float]).columns

    logging.info(f"Categorical Features: {categorical_features.tolist()}")
    logging.info(f"Numerical Features: {numerical_features.tolist()}")

    # define preprocessing for both numerical and categorical features

    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # 0
            ("onehot", OneHotEncoder(handle_unknown="ignore")),  # 1
        ]
    )

    # Bundle preprocessing for numerical and categorical data

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),  # 0
            ("cat", categorical_transformer, categorical_features),  # 1
        ]
    )

    # define the model training pipeine

    pipeline = Pipeline(
        steps=[("preprocess", preprocessor), ("regressor", LinearRegression())]
    )

    # start the Mlflow to run the model

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()
        # mlflow.log_params("fit_tercept": True})

        # mlflow.log_metric("r2_score", .85)

        logging.info("Training Linear Regression Model ......")

        pipeline.fit(X_train, y_train)

        logging.info("Linear Regression Model training completed")

        # Explicitly log the model artifact so ZenML model deployer can locate and deploy it
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_train.iloc[:5],
            serialization_format="cloudpickle",
        )

        # log the columns that the model expect

        onehot_encoder = (
            pipeline.named_steps["preprocess"].transformers_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[categorical_features])
        expected_columns = (
            numerical_features.tolist()
            + onehot_encoder.get_feature_names_out(categorical_features).tolist()
        )

        logging.info(f"Model Expected Columns: {expected_columns}")
        logging.info(f"Training completed successfully")

    except Exception as e:
        logging.error(f"Error training Linear Regression Model: {str(e)}")
        raise e

    finally:
        if mlflow.active_run():
            mlflow.end_run()

    return pipeline
