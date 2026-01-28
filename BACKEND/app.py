from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Load model and scaler
model_path = os.path.join(os.path.dirname(__file__), "..", "water_usage_model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "..", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Inputs from frontend
    temperature = data.get("temperature", 25.0)
    rainfall = data.get("rainfall", 5.0)
    festival = data.get("festival", 0)  # 0 or 1

    # Date features
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    dayofweek = now.weekday()

    # Season calculation
    if month in [12, 1, 2]:
        season = 2  # Winter
    elif month in [3, 4, 5]:
        season = 1  # Summer
    else:
        season = 0  # Monsoon

    # Humidity estimation
    if season == 0:  # Monsoon
        humidity_pct = min(85, 60 + rainfall * 2)
    elif season == 1:  # Summer
        humidity_pct = max(30, 45 - (temperature - 25) + rainfall)
    else:  # Winter
        humidity_pct = 50 + rainfall * 1.5

    humidity_pct = max(20, min(95, humidity_pct))

    # Fixed population
    population = 1200000

    # Feature vector
    features = np.array([[
        temperature,
        rainfall,
        humidity_pct,
        population,
        festival,
        season,
        year,
        month,
        day,
        dayofweek
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    predicted_usage = float(prediction[0])  # Billion Liters

    # -------------------------------
    # SYSTEM RISK FACTOR LOGIC
    # -------------------------------
    risk_score = 0

    if temperature > 40:
        risk_score += 1
    if rainfall < 10:
        risk_score += 1
    if festival == 1:
        risk_score += 1
    if predicted_usage > 0.18:
        risk_score += 1

    if risk_score <= 1:
        system_risk = "Low"
    elif risk_score == 2:
        system_risk = "Medium"
    else:
        system_risk = "High"

    # -------------------------------
    # AI CONFIDENCE INDEX (dynamic)
    # -------------------------------
    base_confidence = 95
    confidence_penalty = risk_score * 1.8
    ai_confidence = max(88.0, min(99.5, base_confidence - confidence_penalty))

    return jsonify({
        "predicted_water_usage": round(predicted_usage, 3),
        "system_risk_factor": system_risk,
        "ai_confidence_index": round(ai_confidence, 1)
    })


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Water Demand Forecaster API",
        "endpoints": {
            "/predict": "POST - Predict water usage",
            "/": "GET - API info"
        },
        "required_inputs": {
            "temperature": "Temperature in Celsius",
            "rainfall": "Rainfall in mm",
            "festival": "0 or 1 (festival day indicator)"
        }
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
