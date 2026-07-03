"""
DriveSense EV Battery Analytics API - FastAPI Backend
====================================================
Hackathon-winning grade backend featuring Pydantic validation, 
advanced ML insights, and auto-generated Swagger UI docs.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global state to hold ML models and dataset info
ml_models: Dict[str, Any] = {}

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
# ML Initialization
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    dataset_path = Path(__file__).parent / "archive (2) (1).csv"
    if not dataset_path.exists():
        logger.error("Dataset not found at %s", dataset_path)
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info("Loading dataset from %s …", dataset_path)
    df = pd.read_csv(dataset_path)
    raw_df = df.copy()

    df = df.drop_duplicates()
    if "Vehicle_ID" in df.columns:
        df = df.drop("Vehicle_ID", axis=1)

    encoders = {}
    cat_cols = ["Car_Model", "Battery_Type", "Driving_Style", "Battery_Status"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(["SoH_Percent", "Battery_Status"], axis=1)
    y = df["SoH_Percent"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    logger.info("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("Model training complete.")

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    # Store in global state
    ml_models.update({
        "raw_df": raw_df,
        "clean_df": df,
        "model": model,
        "encoders": encoders,
        "feature_names": feature_names,
        "metrics": {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "accuracy": round(r2 * 100, 2),
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        },
        "test_data": {
            "y_test": y_test.tolist(),
            "predictions": predictions.tolist()
        }
    })
    
    yield
    # --- Shutdown ---
    ml_models.clear()

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DriveSense EV Battery API",
    description="Hackathon-grade API for EV Battery State of Health analysis, predictive maintenance, and mechanic reports.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def predict_soh(data: dict) -> float:
    """Predict SoH using the loaded ML model and encoders."""
    encoders = ml_models["encoders"]
    feature_names = ml_models["feature_names"]
    
    row = {}
    for feat in feature_names:
        if feat in ("Car_Model", "Battery_Type", "Driving_Style"):
            try:
                le = encoders[feat]
                row[feat] = int(le.transform([data[feat]])[0])
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid value for '{feat}': {data[feat]}")
        else:
            row[feat] = float(data[feat])

    input_df = pd.DataFrame([row], columns=feature_names)
    prediction = ml_models["model"].predict(input_df)[0]
    return round(float(prediction), 4)

def determine_status(soh: float) -> str:
    if soh >= 90: return "Healthy"
    if soh >= 80: return "Moderate"
    return "Critical"

# ---------------------------------------------------------------------------
# Core Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to DriveSense API. Go to /docs for Swagger UI."}

@app.get("/api/metrics", tags=["Analytics"])
def get_metrics():
    return ml_models["metrics"]

@app.get("/api/feature-importance", tags=["Analytics"])
def get_feature_importance():
    importances = ml_models["model"].feature_importances_
    return {
        "features": ml_models["feature_names"],
        "importances": [round(float(i), 4) for i in importances]
    }

@app.get("/api/predictions", tags=["Analytics"])
def get_test_predictions():
    return {
        "actual": [round(float(v), 4) for v in ml_models["test_data"]["y_test"]],
        "predicted": [round(float(v), 4) for v in ml_models["test_data"]["predictions"]]
    }

@app.get("/api/dataset", tags=["Data Exploration"])
def get_dataset_preview():
    raw = ml_models["raw_df"]
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
def get_correlation_matrix():
    df = ml_models["clean_df"]
    corr = df.corr()
    return {
        "labels": corr.columns.tolist(),
        "matrix": np.round(corr.values, 4).tolist()
    }

@app.get("/api/distribution", tags=["Data Exploration"])
def get_soh_distribution():
    raw = ml_models["raw_df"]
    counts, bins = np.histogram(raw["SoH_Percent"], bins=20)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    return {
        "labels": labels,
        "counts": counts.tolist()
    }

@app.get("/api/status-distribution", tags=["Data Exploration"])
def get_status_distribution():
    raw = ml_models["raw_df"]
    counts = raw["Battery_Status"].value_counts()
    return {
        "labels": counts.index.tolist(),
        "counts": counts.values.tolist()
    }

@app.get("/api/form-options", tags=["Data Exploration"])
def get_form_options():
    raw = ml_models["raw_df"]
    return {
        "car_models": sorted(raw["Car_Model"].unique().tolist()),
        "battery_types": sorted(raw["Battery_Type"].unique().tolist()),
        "driving_styles": sorted(raw["Driving_Style"].unique().tolist()),
    }

# ---------------------------------------------------------------------------
# Hackathon-Winning Endpoints (AI & Predictive Analytics)
# ---------------------------------------------------------------------------
@app.post("/api/predict", tags=["Predictive Engine"])
def predict_battery_health(features: BatteryFeatures):
    """Predict Battery State of Health (SoH) based on input parameters."""
    soh_prediction = predict_soh(features.dict())
    return {
        "soh_prediction": soh_prediction,
        "status": determine_status(soh_prediction)
    }

@app.post("/api/prescriptive-analytics", tags=["Predictive Engine"])
def prescriptive_analytics(features: BatteryFeatures):
    """
    Perform a 'What-If' analysis to see how changing driving behavior 
    or charging habits impacts battery health.
    """
    base_data = features.dict()
    base_soh = predict_soh(base_data)
    
    scenarios = []
    
    # Scenario 1: Reduce Fast Charging by 50%
    if base_data["Fast_Charge_Ratio"] > 0.05:
        scenario_data = base_data.copy()
        scenario_data["Fast_Charge_Ratio"] = scenario_data["Fast_Charge_Ratio"] * 0.5
        new_soh = predict_soh(scenario_data)
        scenarios.append({
            "action": "Reduce fast charging dependency by 50%",
            "new_soh": new_soh,
            "improvement": round(new_soh - base_soh, 4)
        })
        
    # Scenario 2: Improve Driving Style to 'Eco'
    if base_data["Driving_Style"] != "Eco":
        scenario_data = base_data.copy()
        scenario_data["Driving_Style"] = "Eco"
        new_soh = predict_soh(scenario_data)
        scenarios.append({
            "action": "Adopt 'Eco' driving style",
            "new_soh": new_soh,
            "improvement": round(new_soh - base_soh, 4)
        })

    # Scenario 3: Better Temperature Management (keep close to 22C)
    if abs(base_data["Avg_Temperature_C"] - 22.0) > 5.0:
        scenario_data = base_data.copy()
        # Bring temp closer to ideal 22C
        scenario_data["Avg_Temperature_C"] = 22.0
        new_soh = predict_soh(scenario_data)
        scenarios.append({
            "action": "Optimize battery thermal management to ~22°C",
            "new_soh": new_soh,
            "improvement": round(new_soh - base_soh, 4)
        })
        
    # Sort scenarios by highest improvement
    scenarios = sorted(scenarios, key=lambda x: x["improvement"], reverse=True)

    return {
        "current_soh": base_soh,
        "current_status": determine_status(base_soh),
        "scenarios": scenarios
    }

@app.post("/api/mechanic-report", tags=["Predictive Engine"])
def generate_mechanic_report(features: BatteryFeatures):
    """
    Generate an AI-style human-readable mechanic's diagnostic report 
    based on the raw data parameters.
    """
    data = features.dict()
    soh = predict_soh(data)
    status = determine_status(soh)
    
    # Generate insights based on rules
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

    # Construct the report text
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
