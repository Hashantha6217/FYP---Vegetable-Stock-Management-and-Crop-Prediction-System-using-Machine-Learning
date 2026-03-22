from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date
from typing import Literal, List
import joblib
import numpy as np
import os
import requests
import pandas as pd

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check Endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# OpenWeatherMap API Key
OPENWEATHER_API_KEY = "a2ad6ab003658896c45c572ec007bbc2"

# Request and Response Models
class PredictionRequest(BaseModel):
    input_region: str               # e.g., "Ampara"
    harvest_date: date              # date to fetch weather and base prediction
    crop_final_date: date           # date to predict market price
    is_fruit: Literal[True, False]  # True for fruit, False for vegetable

class PredictedCrop(BaseModel):
    crop: str
    price: str
    impact_score: float
    confidence: str

class PredictionResponse(BaseModel):
    date: str
    weather_used: dict
    predictions: List[PredictedCrop]

# Fetch weather for target day (limited to 5-day forecast range)
def fetch_weather_data(city: str, target_date: date):
    url = (
        f"https://api.openweathermap.org/data/2.5/forecast?q={city},LK"
        f"&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    response = requests.get(url)
    data = response.json()

    if "list" not in data:
        raise Exception(f"Weather data unavailable for {city}: {data.get('message')}")

    for entry in data["list"]:
        entry_date = entry["dt_txt"].split(" ")[0]
        if entry_date == str(target_date):
            temp = entry["main"]["temp"]
            humidity = entry["main"]["humidity"]
            rainfall = entry.get("rain", {}).get("3h", 0.0)
            return temp, rainfall, humidity

    raise Exception(f"Weather data for {target_date} not found in forecast.")

# Predict top crops
def predict_top_crops(region, temperature, rainfall, humidity, impact_score, model_dir, crop_final_date, is_fruit):
    model_cls = joblib.load(os.path.join(model_dir, "model_cls.pkl"))
    scaler_cls = joblib.load(os.path.join(model_dir, "scaler_cls.pkl"))
    model_reg = joblib.load(os.path.join(model_dir, "model_reg.pkl"))
    scaler_reg = joblib.load(os.path.join(model_dir, "scaler_reg.pkl"))
    region_encoder = joblib.load(os.path.join(model_dir, "region_encoder.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    region_encoded = region_encoder.transform([region])[0]

    # === Use correct classification features ===
    price_column = "fruit_Price per Unit (LKR/kg)" if is_fruit else "vegitable_Price per Unit (LKR/kg)"
    features_cls = [
        'Region_encoded', 'Temperature', 'Rainfall (mm)', 'Humidity (%)',
        'Crop Yield Impact Score', price_column, 'Day', 'Month', 'Year'
    ]

    cls_input_df = pd.DataFrame([[
        region_encoded, temperature, rainfall, humidity,
        impact_score, 0.0,
        crop_final_date.day, crop_final_date.month, crop_final_date.year
    ]], columns=features_cls)

    cls_input_scaled = scaler_cls.transform(cls_input_df)
    probs = model_cls.predict_proba(cls_input_scaled)[0]
    top5_indices = np.argsort(probs)[::-1][:5]

    results = []

    features_reg = [
        'Temperature', 'Rainfall (mm)', 'Humidity (%)',
        'Crop Yield Impact Score', 'Day', 'Month', 'Year',
        'Label_encoded', 'Region_encoded'
    ]

    for idx in top5_indices:
        label = label_encoder.inverse_transform([idx])[0]
        confidence = probs[idx] * 100

        reg_input_df = pd.DataFrame([[
            temperature, rainfall, humidity, impact_score,
            crop_final_date.day, crop_final_date.month, crop_final_date.year,
            idx, region_encoded
        ]], columns=features_reg)

        reg_input_scaled = scaler_reg.transform(reg_input_df)
        price = model_reg.predict(reg_input_scaled)[0]

        results.append({
            "crop": label,
            "price": f"{price:.2f} LKR/kg",
            "impact_score": round(impact_score, 2),
            "confidence": f"{confidence:.1f}%"
        })

    return results

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        subfolder = "Fruits" if req.is_fruit else "Vegetables"
        region_name = f"{req.input_region}_{subfolder.lower()}"
        model_dir = os.path.join("models", subfolder, region_name)

        temperature, rainfall, humidity = fetch_weather_data(req.input_region, req.harvest_date)
        impact_score = 8.5  

        predictions = predict_top_crops(
            region=req.input_region,
            temperature=temperature,
            rainfall=rainfall,
            humidity=humidity,
            impact_score=impact_score,
            model_dir=model_dir,
            crop_final_date=req.crop_final_date,
            is_fruit=req.is_fruit
        )

        return PredictionResponse(
            date=str(req.harvest_date),
            weather_used={
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall
            },
            predictions=predictions
        )

    except Exception as e:
        return PredictionResponse(
            date=str(req.harvest_date),
            weather_used={},
            predictions=[]
        )
