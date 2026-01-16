from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from typing import Dict, Any
from math import radians, cos, sin, asin, sqrt
from datetime import datetime

# =====================================================
# APP SETUP
# =====================================================
app = FastAPI(title="Disaster Early Warning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# API KEYS (SET IN RAILWAY / ENV)
# =====================================================
AMBEE_KEY = os.getenv("AMBEE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# =====================================================
# STATE COORDINATES (ADD MORE IF NEEDED)
# =====================================================
STATE_COORDS = {
    "Assam": (26.2006, 92.9376),
    "Delhi": (28.6139, 77.2090),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Kerala": (10.8505, 76.2711),
    "Odisha": (20.9517, 85.0985),
    "Tamil Nadu": (11.1271, 78.6569),
}

# =====================================================
# AMBEE EVENT TYPE MAP (CRITICAL FIX)
# =====================================================
AMBEE_EVENT_MAP = {
    "flood": "FL",
    "earthquake": "EQ",
    "cyclone": "TC",
    "landslide": "LS",
    "wildfire": "WF",
    "heatwave": None  # Heatwave not in Ambee disasters
}

# =====================================================
# DISTANCE CALCULATION (HAVERSINE)
# =====================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# =====================================================
# AMBEE DISASTER RISK (REAL & FILTERED)
# =====================================================
def get_ambee_disaster_risk(lat: float, lng: float, disaster_type: str) -> float:
    event_code = AMBEE_EVENT_MAP.get(disaster_type)

    if not event_code:
        return 0.0  # Heatwave etc.

    url = "https://api.ambeedata.com/disasters/latest/by-lat-lng"
    headers = {"x-api-key": AMBEE_KEY}
    params = {
        "lat": lat,
        "lng": lng,
        "eventType": event_code,
        "limit": 10
    }

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code != 200:
            return 0.0

        data = r.json().get("data", [])
        risk = 0.0

        for event in data:
            ev_lat = event.get("latitude")
            ev_lng = event.get("longitude")
            severity = event.get("severity", 1)

            if ev_lat and ev_lng:
                distance = haversine(lat, lng, ev_lat, ev_lng)
                if distance <= 200:  # ONLY nearby events
                    risk += severity * 20

        return min(90.0, risk)

    except:
        return 0.0

# =====================================================
# OPEN-METEO WEATHER (TRUSTED)
# =====================================================
def get_weather(lat: float, lng: float) -> Dict[str, float]:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lng,
            "current": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m",
            "timezone": "Asia/Kolkata"
        }
        r = requests.get(url, params=params, timeout=10)
        c = r.json()["current"]
        return {
            "temperature": c["temperature_2m"],
            "humidity": c["relative_humidity_2m"],
            "precipitation": c["precipitation"],
            "wind_speed": c["wind_speed_10m"]
        }
    except:
        return {"temperature": 25, "humidity": 60, "precipitation": 0, "wind_speed": 5}

# =====================================================
# WEATHER-BASED RISK (REAL LOGIC)
# =====================================================
def weather_risk(weather: dict, disaster: str) -> float:
    t = weather["temperature"]
    r = weather["precipitation"]
    w = weather["wind_speed"]

    if disaster == "flood":
        return min(80, r * 25)
    if disaster == "heatwave":
        return max(0, (t - 38) * 8)
    if disaster == "cyclone":
        return min(80, w * 6)
    if disaster == "landslide":
        return min(70, r * 20)
    if disaster == "earthquake":
        return 40  # baseline seismic risk

    return 0

# =====================================================
# HUGGINGFACE SEMANTIC CONFIDENCE (OPTIONAL)
# =====================================================
def hf_confidence(state: str, disaster: str, weather: dict) -> float:
    try:
        url = "https://api-inference.huggingface.co/models/AventIQ-AI/Bert-Disaster-SOS-Message-Classifier"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        prompt = (
            f"{disaster} risk in {state}, India. "
            f"Temp {weather['temperature']}C, "
            f"Rain {weather['precipitation']}mm, "
            f"Wind {weather['wind_speed']}kmh."
        )
        r = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=10)
        if r.status_code == 200:
            return r.json()[0]["score"] * 100
    except:
        pass
    return 50.0

# =====================================================
# MAIN PREDICTION ENDPOINT
# =====================================================
@app.get("/predict/{state}")
async def predict(state: str, disaster_type: str):
    if state not in STATE_COORDS:
        raise HTTPException(404, "State not found")

    lat, lng = STATE_COORDS[state]

    weather = get_weather(lat, lng)
    ambee = get_ambee_disaster_risk(lat, lng, disaster_type)
    weather_score = weather_risk(weather, disaster_type)
    hf_score = hf_confidence(state, disaster_type, weather)

    final_risk = round(
        ambee * 0.5 +
        weather_score * 0.3 +
        hf_score * 0.2,
        1
    )

    return {
        "state": state,
        "disaster_type": disaster_type,
        "risk_percentage": final_risk,
        "weather": weather,
        "confidence": round(hf_score / 100, 2),
        "details": {
            "ambee": round(ambee, 1),
            "weather_factor": round(weather_score, 1),
            "hf": round(hf_score, 1),
            "calculation_time": datetime.now().isoformat()
        }
    }

# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
async def health():
    return {"status": "healthy"}
