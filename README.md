# Water Demand Forecaster

A machine learning project to predict urban water usage based on various factors including temperature, rainfall, population, and seasonal patterns.

## Project Structure

```
.
├── main.ipynb                      # Jupyter notebook for model training
├── urban_water_usage_dataset.csv   # Dataset
├── water_usage_model.pkl           # Trained model
├── scaler.pkl                      # Feature scaler
├── BACKEND/
│   ├── app.py                      # Flask API server
│   └── requirements.txt            # Python dependencies
└── README.md
```

## Features

- Linear Regression model with 99.99% R² score
- REST API for water usage predictions
- Automatic feature engineering (date-based features)
- Seasonal classification

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r BACKEND/requirements.txt
   ```

2. **Train the model (optional):**
   - Open and run `main.ipynb`
   - This will generate `water_usage_model.pkl` and `scaler.pkl`

3. **Run the Flask API:**
   ```bash
   cd BACKEND
   python app.py
   ```

   The API will be available at `http://localhost:5000`

## API Usage

**Endpoint:** `POST /predict`

**Request body:**
```json
{
  "temperature": 25.5,
  "rainfall": 10.2,
  "festival": 1
}
```

**Response:**
```json
{
  "predicted_water_usage": 1234567.89
}
```

The API automatically extracts date-based features (year, month, day, day of week) and determines the season from the current date.

## Model Performance

- **R² Score:** 0.9999
- **RMSE:** 70,543.87

## License

MIT
