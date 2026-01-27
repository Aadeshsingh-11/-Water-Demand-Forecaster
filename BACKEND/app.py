from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Load model and scaler from parent directory
model_path = os.path.join(os.path.dirname(__file__), "..", "water_usage_model.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "..", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Only 3 inputs from frontend
    temperature = data.get("temperature", 25.0)
    rainfall = data.get("rainfall", 5.0)
    festival = data.get("festival", 0)  # 0 or 1

    # Automatically calculate date-related features
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    dayofweek = now.weekday()

    # Calculate season from month (Monsoon=0, Summer=1, Winter=2)
    if month in [12, 1, 2]:
        season = 2  # Winter
    elif month in [3, 4, 5]:
        season = 1  # Summer
    else:  # June to November
        season = 0  # Monsoon

    # Estimate humidity based on season and rainfall
    # Higher rainfall typically means higher humidity
    # Monsoon season has higher humidity, summer lower
    if season == 0:  # Monsoon
        humidity_pct = min(85, 60 + rainfall * 2)
    elif season == 1:  # Summer
        humidity_pct = max(30, 45 - (temperature - 25) + rainfall)
    else:  # Winter
        humidity_pct = 50 + rainfall * 1.5
    
    # Clamp humidity between realistic bounds
    humidity_pct = max(20, min(95, humidity_pct))

    # Fixed population (average from dataset)
    population = 1200000

    # Feature order: temperature, rainfall_mm, humidity_pct, population, is_festival, season, year, month, day, dayofweek
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

    # Return only the predicted value for frontend integration
    return jsonify({
        "predicted_water_usage": float(prediction[0])
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
