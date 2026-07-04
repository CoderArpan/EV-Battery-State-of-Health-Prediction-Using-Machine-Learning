"""
DriveSense EV Battery Analytics API - FastAPI Backend
====================================================
Hackathon-winning grade backend featuring Pydantic validation, 
advanced ML insights, modular architecture, caching, and ORJSON for speed.
"""

import logging
from contextlib import asynccontextmanager
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import our modular ML engine
from ml_engine import MLEngine

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize ML Engine
ml = MLEngine()

# ---------------------------------------------------------------------------
# Pydantic Schemas for Request Validation
# ---------------------------------------------------------------------------
class BatteryFeatures(BaseModel):
    Car_Model: str = Field(..., description="E.g., 'Nissan Leaf', 'Tesla Model 3'")
    Battery_Type: str = Field(..., description="E.g., 'NMC', 'LFP'")
    Battery_Capacity_kWh: float = Field(..., description="Battery capacity in kWh")
    Vehicle_Age_Months: float = Field(..., description="Age of the vehicle in months")
    Total_Charging_Cycles: float = Field(..., description="Total number of charging cycles")
    Avg_Temperature_C: float = Field(..., description="Average operating temperature in Celsius")
    Fast_Charge_Ratio: float = Field(..., description="Ratio of fast charges to total charges (0.0 to 1.0)")
    Avg_Discharge_Rate_C: float = Field(..., description="Average discharge rate (C-rate)")
    Driving_Style: str = Field(..., description="E.g., 'Aggressive', 'Moderate', 'Eco'")
    Internal_Resistance_Ohm: float = Field(..., description="Internal resistance in Ohms")

    class Config:
        json_schema_extra = {
            "example": {
                "Car_Model": "Nissan Leaf",
                "Battery_Type": "NMC",
                "Battery_Capacity_kWh": 40.0,
                "Vehicle_Age_Months": 44,
                "Total_Charging_Cycles": 259,
                "Avg_Temperature_C": 12.8,
                "Fast_Charge_Ratio": 0.18,
                "Avg_Discharge_Rate_C": 2.19,
                "Driving_Style": "Moderate",
                "Internal_Resistance_Ohm": 0.0562
            }
        }

# ---------------------------------------------------------------------------
# Startup Event
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load from disk or train if missing
    ml.load_or_train()
    yield

# ---------------------------------------------------------------------------
# FastAPI App using ORJSONResponse for speed
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DriveSense EV Battery API",
    description="High-performance API for EV Battery State of Health analysis with ML caching.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Core Endpoints (With caching for static data)
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to DriveSense API. Go to /docs for Swagger UI."}

@app.get("/api/metrics", tags=["Analytics"])
@lru_cache(maxsize=1)
def get_metrics():
    return ml.model_data["metrics"]

@app.get("/api/feature-importance", tags=["Analytics"])
@lru_cache(maxsize=1)
def get_feature_importance():
    importances = ml.model_data["model"].feature_importances_
    return {
        "features": ml.model_data["feature_names"],
        "importances": [round(float(i), 4) for i in importances]
    }

@app.get("/api/predictions", tags=["Analytics"])
@lru_cache(maxsize=1)
def get_test_predictions():
    return {
        "actual": [round(float(v), 4) for v in ml.model_data["test_data"]["y_test"]],
        "predicted": [round(float(v), 4) for v in ml.model_data["test_data"]["predictions"]]
    }

@app.get("/api/dataset", tags=["Data Exploration"])
@lru_cache(maxsize=1)
def get_dataset_preview():
    raw = ml.model_data["raw_df"]
    preview = raw.head(100)
    return {
        "columns": preview.columns.tolist(),
        "rows": preview.values.tolist(),
        "total_rows": len(raw),
        "stats": {
            "avg_soh": round(raw["SoH_Percent"].mean(), 4),
            "min_soh": round(raw["SoH_Percent"].min(), 4),
            "max_soh": round(raw["SoH_Percent"].max(), 4),
            "total_vehicles": len(raw),
            "car_models_count": raw["Car_Model"].nunique(),
            "battery_types_count": raw["Battery_Type"].nunique(),
            "avg_age_months": round(raw["Vehicle_Age_Months"].mean(), 4),
            "avg_cycles": round(raw["Total_Charging_Cycles"].mean(), 4)
        }
    }

@app.get("/api/correlation", tags=["Data Exploration"])
@lru_cache(maxsize=1)
def get_correlation_matrix():
    df = ml.model_data["clean_df"]
    corr = df.corr()
    return {
        "labels": corr.columns.tolist(),
        "matrix": np.round(corr.values, 4).tolist()
    }

@app.get("/api/distribution", tags=["Data Exploration"])
@lru_cache(maxsize=1)
def get_soh_distribution():
    raw = ml.model_data["raw_df"]
    counts, bins = np.histogram(raw["SoH_Percent"], bins=20)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    return {
        "labels": labels,
        "counts": counts.tolist()
    }

@app.get("/api/status-distribution", tags=["Data Exploration"])
@lru_cache(maxsize=1)
def get_status_distribution():
    raw = ml.model_data["raw_df"]
    counts = raw["Battery_Status"].value_counts()
    return {
        "labels": counts.index.tolist(),
        "counts": counts.values.tolist()
    }

@app.get("/api/form-options", tags=["Data Exploration"])
@lru_cache(maxsize=1)
def get_form_options():
    raw = ml.model_data["raw_df"]
    return {
        "car_models": sorted(raw["Car_Model"].unique().tolist()),
        "battery_types": sorted(raw["Battery_Type"].unique().tolist()),
        "driving_styles": sorted(raw["Driving_Style"].unique().tolist()),
    }

# ---------------------------------------------------------------------------
# Predictive Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/predict", tags=["Predictive Engine"])
def predict_battery_health(features: BatteryFeatures):
    try:
        soh_prediction = ml.predict(features.dict())
        return {
            "soh_prediction": soh_prediction,
            "status": ml.determine_status(soh_prediction)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/prescriptive-analytics", tags=["Predictive Engine"])
def prescriptive_analytics(features: BatteryFeatures):
    base_data = features.dict()
    base_soh = ml.predict(base_data)
    
    scenarios = []
    
    # Scenario 1: Reduce Fast Charging by 50%
    if base_data["Fast_Charge_Ratio"] > 0.05:
        scenario_data = base_data.copy()
        scenario_data["Fast_Charge_Ratio"] = scenario_data["Fast_Charge_Ratio"] * 0.5
        new_soh = ml.predict(scenario_data)
        scenarios.append({
            "action": "Reduce fast charging dependency by 50%",
            "new_soh": new_soh,
            "improvement": round(new_soh - base_soh, 4)
        })
        
    # Scenario 2: Improve Driving Style to 'Eco'
    if base_data["Driving_Style"] != "Eco":
        scenario_data = base_data.copy()
        scenario_data["Driving_Style"] = "Eco"
        new_soh = ml.predict(scenario_data)
        scenarios.append({
            "action": "Adopt 'Eco' driving style",
            "new_soh": new_soh,
            "improvement": round(new_soh - base_soh, 4)
        })

    # Scenario 3: Better Temperature Management
    if abs(base_data["Avg_Temperature_C"] - 22.0) > 5.0:
        scenario_data = base_data.copy()
        scenario_data["Avg_Temperature_C"] = 22.0
        new_soh = ml.predict(scenario_data)
        scenarios.append({
            "action": "Optimize battery thermal management to ~22°C",
            "new_soh": new_soh,
            "improvement": round(new_soh - base_soh, 4)
        })
        
    scenarios = sorted(scenarios, key=lambda x: x["improvement"], reverse=True)

    return {
        "current_soh": base_soh,
        "current_status": ml.determine_status(base_soh),
        "scenarios": scenarios
    }

@app.post("/api/mechanic-report", tags=["Predictive Engine"])
def generate_mechanic_report(features: BatteryFeatures):
    data = features.dict()
    soh = ml.predict(data)
    status = ml.determine_status(soh)
    
    insights = []
    if data["Fast_Charge_Ratio"] > 0.3:
        insights.append(f"The battery is being fast-charged very frequently ({data['Fast_Charge_Ratio']*100:.1f}% of the time). This accelerates lithium plating and structural degradation.")
    
    if data["Avg_Temperature_C"] > 35:
        insights.append(f"Operating at a high average temperature ({data['Avg_Temperature_C']}°C) is severely impacting the internal chemistry, accelerating capacity loss.")
    elif data["Avg_Temperature_C"] < 5:
        insights.append(f"Operating at a very low average temperature ({data['Avg_Temperature_C']}°C) causes high internal resistance and potential lithium plating during charging.")
        
    if data["Driving_Style"] == "Aggressive":
        insights.append("Aggressive driving style leads to high discharge rates, generating excess internal heat and stressing the cells.")
        
    if data["Internal_Resistance_Ohm"] > 0.08:
        insights.append(f"Internal resistance is critically high ({data['Internal_Resistance_Ohm']}Ω), indicating significant aging and reduced power delivery capability.")
        
    if data["Total_Charging_Cycles"] > 1000:
        insights.append(f"The battery has endured a high number of charge cycles ({int(data['Total_Charging_Cycles'])}), natural chemical exhaustion is expected.")

    report_text = f"Diagnostic Report for {data['Car_Model']} ({data['Battery_Type']} Battery)\n\n"
    report_text += f"Current State of Health (SoH) is predicted at {soh:.2f}%, which is classified as {status}.\n\n"
    
    if not insights:
        report_text += "The battery is operating under optimal conditions. No immediate stress factors detected. Continue regular maintenance."
    else:
        report_text += "Key Stress Factors Identified:\n"
        for idx, insight in enumerate(insights, 1):
            report_text += f"{idx}. {insight}\n"
            
        report_text += "\nRecommendation: Review the 'What-If' prescriptive analytics to see how changing these habits can extend battery life."

    return {
        "vehicle": data["Car_Model"],
        "predicted_soh": soh,
        "status": status,
        "report_text": report_text
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
