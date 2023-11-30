from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import os
import mlflow.pyfunc

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# Load the model
model = mlflow.pyfunc.load_model(model_uri="runs:/e7e1ea35d4334e5e838354cbd4201a91/model")

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Get the JSON data from the request
        data = request.json

        # Perform the prediction using the loaded model
        prediction = model.predict(data)

        # Return the prediction as JSON
        return jsonify(round(prediction.tolist()[0], 2))
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
