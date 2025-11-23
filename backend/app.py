from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

with open("model", "rb") as model_file:
    model = pickle.load(model_file)

@app.route("/")
def home():
    return "Welcome to the Chronic Kidney Disease Prediction API. Use POST /predict to make predictions."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = [
        data['white_blood_cell_count'],
        data['blood_glucose_random'],
        data['blood_urea'],
        data['serum_creatinine'],
        data['packed_cell_volume'],
        data['albumin'],
        data['haemoglobin'],
        data['age'],
        data['sugar'],
        data['hypertension'],
    ]
    featuresarray = np.array([features])
    prediction = model.predict(featuresarray)
    predictvalue = prediction[0].item()
    print(prediction)
    return jsonify({"prediction": predictvalue})

if __name__ == "__main__":
    app.run(debug=True)
