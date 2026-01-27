from datetime import datetime

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # frontend inputs
    temperature = data["temperature"]
    rainfall = data["rainfall"]
    festival = data["festival"]  # 0 or 1

    # auto date features
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    dayofweek = now.weekday()

    # simple season logic
    if month in [12, 1, 2]:
        season = 0  # Winter
    elif month in [3, 4, 5]:
        season = 1  # Summer
    elif month in [6, 7, 8, 9]:
        season = 2  # Monsoon
    else:
        season = 3  # Post-monsoon

    population = 500000  # fixed / city-wise constant

    features = np.array([[
        season,
        temperature,
        rainfall,
        population,
        year,
        month,
        day,
        dayofweek
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    return jsonify({
        "predicted_water_usage": float(prediction[0])
    })
