# ⚡ DriveSense: EV Battery Intelligence Platform

**Hackathon-Winning Backend Architecture**

DriveSense is an enterprise-grade, API-first platform designed to predict, analyze, and extend the lifespan of Electric Vehicle (EV) batteries. Powered by **FastAPI** and **RandomForestRegressor**, it goes beyond simple predictions by offering **Prescriptive Analytics (What-If)** and **AI-Generated Mechanic Reports**.

---

## 🌟 Key Features

1. **Auto-Generated Swagger Docs**: Built on FastAPI, the entire API is documented and testable interactively out-of-the-box.
2. **Predictive Engine**: High-accuracy `State of Health (SoH)` predictions using Random Forest.
3. **Prescriptive Analytics (What-If)**: The API simulates different scenarios (e.g., "What if I reduce fast charging by 50%?") and returns the quantifiable improvement in battery health.
4. **Mechanic AI Report**: Translates raw data into a human-readable diagnostic report, identifying stress factors like high temperatures or aggressive driving styles.
5. **Pydantic Validation**: Strict input validation ensures only clean data enters the ML pipeline.

---

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **Machine Learning**: Scikit-Learn (RandomForestRegressor, LabelEncoder)
- **Data Manipulation**: Pandas, NumPy
- **Server**: Uvicorn

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install fastapi uvicorn pandas scikit-learn numpy pydantic
```

### 2. Run the Server
```bash
uvicorn main:app --reload --port 8000
```

### 3. Explore the API
Open your browser and navigate to:
- **Swagger UI (Interactive API Docs):** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 🔌 Core API Endpoints

### `POST /api/predict`
Predicts the exact State of Health (%) and classifies the battery status (Healthy, Moderate, Critical).

### `POST /api/prescriptive-analytics`
The "What-If" engine. Submit a vehicle's stats, and the API will return a prioritized list of actionable scenarios to improve battery health.

### `POST /api/mechanic-report`
Generates a detailed, natural-language diagnostic report explaining *why* a battery is degrading and highlighting specific stress factors (e.g., excessive fast charging).

### `GET /api/metrics`
Returns the R², MAE, MSE, and RMSE scores of the ML model based on the train/test split.

---

## 📦 Testing with Postman
A Postman collection (`DriveSense_API_Collection.postman_collection.json`) is included in the repository. Simply import it into Postman to instantly test all routes with pre-filled JSON payloads.

---
*Built for the future of sustainable mobility.* 🌍🔋
