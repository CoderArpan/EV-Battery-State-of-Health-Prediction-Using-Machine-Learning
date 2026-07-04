# ⚡ DriveSense: EV Battery Intelligence Platform

Welcome to the **DriveSense Backend API**! This project is an enterprise-grade, blazing-fast API for analyzing Electric Vehicle (EV) battery health. We built this using **FastAPI** and **Machine Learning** to help you build the best hackathon project possible.

Don't worry if you're a beginner! This guide will walk you through exactly how to run the API and connect it to your own frontend dashboard (like React, HTML/JS, or Next.js).

---

## 🌟 What Does This API Do?

Instead of just returning boring numbers, this API is like an AI Mechanic for EV batteries:
1. **Battery Health Prediction**: Tell it the car model, temperature, and age, and it predicts the exact `State of Health (SoH)` percentage.
2. **"What-If" Analytics**: You can ask it, *"What if I stop fast charging so much?"* and the API will tell you exactly how many months of battery life you will save.
3. **Mechanic Report**: The API writes a full English paragraph explaining *why* the battery is dying (e.g., "The battery is getting too hot").

---

## 🚀 How to Run the Server

It takes less than 2 minutes to get this running on your local machine!

### Step 1: Install Dependencies
Open your terminal/command prompt in this folder and run:
```bash
pip install -r requirements.txt
```
*(This installs FastAPI, the ML libraries, and tools to make the server fast).*

### Step 2: Start the Server
Run this command to start the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```
You should see a message saying `Application startup complete`. 

**🔥 Zero-Delay Startup Feature**: The first time you run this, it will train the AI model and save it as a `model.joblib` file. The next time you start the server, it will load instantly!

### Step 3: Test It Out!
Open your web browser and go to:
**👉 http://localhost:8000/docs**

This opens the **Swagger UI**. It's an interactive dashboard where you can see every single API endpoint and test them by clicking the **"Try it out"** button. Judges *love* seeing this!

---

## 💻 How to Connect Your Frontend (Beginner Guide)

If you are building a React/HTML frontend, here is how you talk to this backend using JavaScript's `fetch()`:

### Example 1: Getting Basic Dataset Stats
Want to show how many cars are in the database on your dashboard?
```javascript
// Run this in your frontend JavaScript
fetch('http://localhost:8000/api/dataset')
  .then(response => response.json())
  .then(data => {
    console.log("Total Vehicles:", data.stats.total_vehicles);
    console.log("Average Health:", data.stats.avg_soh);
  });
```

### Example 2: Making a Prediction & Getting a Mechanic Report
When a user clicks "Predict" on your form, send this JSON to the backend:
```javascript
const vehicleData = {
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
};

// Ask the AI Mechanic for a report
fetch('http://localhost:8000/api/mechanic-report', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(vehicleData)
})
.then(response => response.json())
.then(data => {
  console.log("Health Status:", data.status); // e.g. "Healthy"
  console.log("AI Report:", data.report_text); // Prints the English paragraph!
});
```

---

## 🛠️ Postman Collection

If you prefer testing with Postman, we have included a file named `DriveSense_API_Collection.postman_collection.json`.
1. Open Postman.
2. Click **Import** (top left).
3. Drag and drop that file.
4. All the requests are pre-filled and ready to go!

---
*Built with ❤️ for the next generation of EVs.*
