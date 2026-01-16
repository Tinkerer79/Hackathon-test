from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import requests, os
from datetime import datetime

app = FastAPI(title="India Disaster Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

AMBEE_KEY = os.getenv("AMBEE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# =============================
# STATE COORDINATES
# =============================
STATE_COORDS = {
    "Assam": (26.2006, 92.9376),
    "Delhi": (28.6139, 77.2090),
    "Maharashtra": (19.7515, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Tamil Nadu": (11.1271, 78.6569),
    "Odisha": (20.9517, 85.0985),
    "West Bengal": (22.9868, 87.8550),
    "Gujarat": (22.2587, 71.1924),
    "Rajasthan": (27.0238, 74.2179),
    "Manipur": (24.6633, 93.9063),
    "Goa": (15.2993, 74.1240),
    "Chhattisgarh": (21.2787, 81.8661),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Uttarakhand": (30.0668, 79.0193),
    "Bihar": (25.0961, 85.3131),
    "Karnataka": (15.3173, 75.7139),
    "Telangana": (17.1232, 79.2088),
    "Andhra Pradesh": (15.9129, 79.7400),
    "Punjab": (31.1471, 75.3412),
    "Haryana": (29.0588, 76.0856),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Jharkhand": (23.6102, 85.2799)
}

COASTAL_STATES = {"Kerala", "Tamil Nadu", "Odisha", "Andhra Pradesh",
                  "West Bengal", "Gujarat", "Maharashtra", "Goa", "Karnataka"}

# =============================
# WEATHER FROM OPEN-METEO
# =============================
def get_weather(lat: float, lng: float) -> dict:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lng,
            "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m",
            "timezone": "Asia/Kolkata"
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json().get("hourly", {})

        # latest hour
        temperature = data.get("temperature_2m", [None])[-1]
        precipitation = data.get("precipitation", [None])[-1]
        humidity = data.get("relative_humidity_2m", [None])[-1]
        wind_speed = data.get("wind_speed_10m", [None])[-1]

        return {
            "temperature": temperature,
            "humidity": humidity,
            "precipitation": precipitation,
            "wind_speed": wind_speed
        }
    except Exception as e:
        print("Open-Meteo Error:", e)
        return {
            "temperature": None,
            "humidity": None,
            "precipitation": None,
            "wind_speed": None
        }

# =============================
# AMBEE DISASTER RISK
# =============================
def get_ambee_disaster_risk(lat, lng, disaster):
    try:
        url = "https://api.ambeedata.com/disasters/latest/by-lat-lng"
        headers = {"x-api-key": AMBEE_KEY}
        params = {"lat": lat, "lng": lng, "eventType": disaster.upper(), "limit": 5}
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", [])
            score = 30
            for e in data[:3]:
                score += e.get("severity", 1) * 15
            return min(score, 95)
        return 30
    except:
        return 30

# =============================
# HUGGING FACE SEMANTIC SCORE
# =============================
def hf_confidence(state: str, disaster_type: str, weather: dict) -> float:
    try:
        url = "https://router.huggingface.co/api-inference/models/AventIQ-AI/Bert-Disaster-SOS-Message-Classifier"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        prompt = f"{disaster_type} emergency in {state}, weather: {weather}"
        response = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result[0]["score"] * 100 if result else 50.0
    except Exception as e:
        print("HF Error:", e)
        return 50.0

# =============================
# WEATHER-BASED RISK
# =============================
def weather_risk(w, d):
    if d == "flood":
        return (w["precipitation"] or 0) * 20 + (w["humidity"] or 0)
    if d == "heatwave":
        return max(0, ((w["temperature"] or 25) - 38) * 8)
    if d == "earthquake":
        return 40
    return 30

# =============================
# RECOMMENDATIONS
# =============================
def generate_recommendations(disaster, risk):
    if risk > 70:
        return [f"Immediate alert for {disaster}", "Emergency services standby"]
    if risk > 40:
        return [f"Monitor {disaster} situation", "Public advisory recommended"]
    return ["Situation normal"]

# =============================
# PREDICTION ENDPOINT
# =============================
@app.get("/predict/{state}")
async def predict(state: str, disaster_type: str = Query(...)):
    if state not in STATE_COORDS:
        raise HTTPException(404, "State not found")

    if disaster_type == "cyclone" and state not in COASTAL_STATES:
        return {
            "region": state,
            "disaster_type": disaster_type,
            "risk_percentage": 0.0,
            "risk_level": "LOW",
            "temperature": None,
            "humidity": None,
            "rainfall": None,
            "wind_speed": None,
            "confidence": 0.95,
            "recommendations": ["Cyclone risk not applicable for this state"],
            "timestamp": datetime.now().isoformat()
        }

    lat, lng = STATE_COORDS[state]
    weather = get_weather(lat, lng)
    ambee_score = get_ambee_disaster_risk(lat, lng, disaster_type)
    weather_score = weather_risk(weather, disaster_type)
    hf_score = hf_confidence(state, disaster_type, weather)

    final_risk = round(ambee_score * 0.5 + weather_score * 0.3 + hf_score * 0.2, 1)

    # Determine risk level
    if final_risk < 40:
        risk_level = "LOW"
    elif final_risk < 60:
        risk_level = "MODERATE"
    elif final_risk < 80:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return {
        "region": state,
        "disaster_type": disaster_type,
        "risk_percentage": final_risk,
        "risk_level": risk_level,
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "rainfall": weather["precipitation"],
        "wind_speed": weather["wind_speed"],
        "confidence": round(0.6 + final_risk / 200, 2),
        "recommendations": generate_recommendations(disaster_type, final_risk),
        "timestamp": datetime.now().isoformat()
    }

# =============================
# ALL STATES ENDPOINT (for regional overview)
# =============================
@app.get("/all")
async def all_predictions(disaster_type: str = Query(...)):
    results = []
    for state in STATE_COORDS.keys():
        pred = await predict(state, disaster_type)
        results.append(pred)
    # Sort descending by risk
    results.sort(key=lambda x: x["risk_percentage"], reverse=True)
    return results