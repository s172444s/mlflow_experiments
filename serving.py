import requests

inference_request = {
    'dataframe_records': [{
    'fixed acidity': 7.4,
    'volatile acidity': 0.7,
    'citric acid': 0,
    'residual sugar': 1.9,
    'chlorides': 0.076,
    'free sulfur dioxide': 11,
    'total sulfur dioxide': 34,
    'density': 0.9978,
    'pH': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4}, 
    {
    'fixed acidity': 26,
    'volatile acidity': 0.5,
    'citric acid': 0,
    'residual sugar': 0,
    'chlorides': 0,
    'free sulfur dioxide': 0,
    'total sulfur dioxide': 0,
    'density': 56,
    'pH': 3.51,
    'sulphates': 0.23,
    'alcohol': 0}, 
    {
    "fixed acidity": 7.3,
    "volatile acidity": 0.65,
    "citric acid": 0,
    "residual sugar": 1.2,
    "chlorides": 0.065,
    "free sulfur dioxide": 15,
    "total sulfur dioxide": 21,
    "density": 0.9946,
    "pH": 3.39,
    "sulphates": 0.47,
    "alcohol": 10
}]
}


endpoint = "http://localhost:5000/invocations"

response = requests.post(endpoint, json=inference_request)

print(response.text)