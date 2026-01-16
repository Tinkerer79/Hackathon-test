from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import requests, os
from datetime import datetime

app = FastAPI(title="Disaster Early Warning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

AMBEE_KEY = os.getenv("AMBEE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# =====================================================
# STATE COORDINATES (IMPORTANT)
# =====================================================
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
    "Manipur": (24.6633, 93.9063)
}

COASTAL_STATES = {
    "Kerala", "Tamil Nadu", "Odisha",
    "West Bengal", "Gujarat",
    "Maharashtra", "Goa", "Karnataka"
}

# =====================================================
# WEATHER (OPEN-METEO)
# =====================================================
def get_weather(lat, lng):
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
# AMBEE DISASTER RISK
# =====================================================
def get_ambee_disaster_risk(lat, lng, disaster):
    try:
        url = "https://api.ambeedata.com/disasters/latest/by-lat-lng"
        headers = {"x-api-key": AMBEE_KEY}
        params = {
            "lat": lat,
            "lng": lng,
            "eventType": disaster.upper(),
            "limit": 5
        }
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

# =====================================================
# HUGGING FACE SEMANTIC RISK
# =====================================================
def hf_confidence(state, disaster, weather):
    try:
        url = "https://api-inference.huggingface.co/models/AventIQ-AI/Bert-Disaster-SOS-Message-Classifier"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        prompt = f"{disaster} emergency in {state}, weather {weather}"
        r = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=10)
        return r.json()[0]["score"] * 100
    except:
        return 50

# =====================================================
# WEATHER BASED RISK
# =====================================================
def weather_risk(w, d):
    if d == "flood":
        return min(80, w["precipitation"] * 20 + w["humidity"])
    if d == "heatwave":
        return max(0, (w["temperature"] - 38) * 8)
    if d == "earthquake":
        return 40
    return 30

# =====================================================
# RECOMMENDATIONS
# =====================================================
def generate_recommendations(disaster, risk):
    if risk > 70:
        return [f"Immediate alert for {disaster}", "Emergency services standby"]
    if risk > 40:
        return [f"Monitor {disaster} situation", "Public advisory recommended"]
    return ["Situation normal"]

# =====================================================
# MAIN ENDPOINT
# =====================================================
@app.get("/predict/{state}")
async def predict(
    state: str,
    disaster_type: str = Query(...)
):
    if state not in STATE_COORDS:
        raise HTTPException(404, "State not found")

    lat, lng = STATE_COORDS[state]
    weather = get_weather(lat, lng)

    if disaster_type == "cyclone" and state not in COASTAL_STATES:
        return {
            "risk_percentage": 0.0,
            "weather": weather,
            "hf_semantic_score": 0.0,
            "confidence": 0.95,
            "recommendations": ["Cyclone not applicable for this state"],
            "details": {
                "sources": "Geographic Filter",
                "calculation_time": datetime.now().isoformat()
            }
        }

    ambee = get_ambee_disaster_risk(lat, lng, disaster_type)
    weather_score = weather_risk(weather, disaster_type)
    hf = hf_confidence(state, disaster_type, weather)

    final_risk = round(
        ambee * 0.5 + weather_score * 0.3 + hf * 0.2,
        1
    )

    return {
        "risk_percentage": final_risk,
        "weather": weather,
        "hf_semantic_score": round(hf / 100, 2),
        "confidence": round(0.6 + final_risk / 200, 2),
        "recommendations": generate_recommendations(disaster_type, final_risk),
        "details": {
            "sources": "Ambee + Open-Meteo + Hugging Face",
            "calculation_time": datetime.now().isoformat()
        }
    }
