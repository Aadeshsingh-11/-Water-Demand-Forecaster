from flask import Flask,request,jsonify 
import joblib
import numpy as np

app = Flask(__name__)

# Loading model and scaler

model = joblib.load("water_usage_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/",methods=["GET"])
def home():
    return {"message": "Water Demand Forecasting API running"}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        data["season"],
        data["temperature"],
        data["rainfall"],
        data["population"],
        data["year"],
        data["month"],
        data["day"],
        data["dayofweek"]
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    return jsonify({
        "predicted_water_usage": float(prediction[0])
    })

if __name__ == "__main__":
    app.run()