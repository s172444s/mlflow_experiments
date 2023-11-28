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
    'alcohol': 9.4}]#[[6.7,3.3,5.7,2.1]]
}


endpoint = "http://localhost:5000/invocations"

response = requests.post(endpoint, json=inference_request)

print(response.text)