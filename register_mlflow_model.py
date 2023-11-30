import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


import warnings
import logging
import sys

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

def train_model(train_x, train_y, alpha, l1_ratio):
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    return lr

def register_mlflow_model(model, train_x, model_name, remote_server_uri, parameters=None, metrics=None):
    with mlflow.start_run():
        # Log the model
        predictions = model.predict(train_x)
        signature = infer_signature(train_x, predictions)

        mlflow.set_tracking_uri(remote_server_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            # Register the model in the Model Registry
            mlflow.sklearn.log_model(
                model, "model", registered_model_name=model_name, signature=signature
            )
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)

        # Log parameters if provided
        if parameters is not None:
            for param_name, param_value in parameters.items():
                mlflow.log_param(param_name, param_value)

        # Log metrics if provided
        if metrics is not None:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    model_name = "ElasticnetWineModel"
    remote_server_uri = "https://dagshub.com/s172444s/mlflow_experiments.mlflow"

    # Training the model
    trained_model = train_model(train_x, train_y, alpha, l1_ratio)

    # Evaluating metrics
    test_x = test.drop(["quality"], axis=1)
    test_y = test[["quality"]]
    predicted_qualities = trained_model.predict(test_x)
    metrics = eval_metrics(test_y, predicted_qualities)

    # Defining parameters
    parameters = {"alpha": alpha, "l1_ratio": l1_ratio}

    # Registering the model in MLflow
    register_mlflow_model(trained_model, train_x, model_name, remote_server_uri, parameters, metrics)
