## MLflow experiments


Credentials for using MLflow:

MLFLOW_TRACKING_URI=https://dagshub.com/s172444s/mlflow_experiments.mlflow \
MLFLOW_TRACKING_USERNAME=s172444s \
MLFLOW_TRACKING_PASSWORD=fda46f678c941f6e0f5d526399c71dcd2b1c6cfa \


## MLflow serving

Running two terminals at the same time (we can change the run id and port as we wish):
1. We can either run(based on the run id):
mlflow models serve --model-uri runs:/e7e1ea35d4334e5e838354cbd4201a91/model --no-conda --port 5000
or running (based on production, or staging):
mlflow models serve --model-uri models:/ElasticnetWineModel/Production --no-conda --port 5000

2. python serving.py (We can change the serving data as we wish)