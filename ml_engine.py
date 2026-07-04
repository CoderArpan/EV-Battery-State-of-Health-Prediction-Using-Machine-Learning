import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class MLEngine:
    def __init__(self, data_path: str = "archive (2) (1).csv", model_path: str = "model.joblib"):
        self.data_path = Path(__file__).parent / data_path
        self.model_path = Path(__file__).parent / model_path
        self.model_data: Dict[str, Any] = {}

    def load_or_train(self) -> Dict[str, Any]:
        """
        Loads the pre-trained model from disk. 
        If it doesn't exist, trains a new model and saves it.
        """
        if self.model_path.exists():
            logger.info("Pre-trained model found. Loading from %s (Zero-delay startup)", self.model_path)
            try:
                self.model_data = joblib.load(self.model_path)
                return self.model_data
            except Exception as e:
                logger.error("Failed to load model: %s. Retraining...", e)

        logger.info("No pre-trained model found. Training new model...")
        return self.train_and_save()

    def train_and_save(self) -> Dict[str, Any]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        logger.info("Loading dataset...")
        df = pd.read_csv(self.data_path)
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

        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        self.model_data = {
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
        }

        logger.info("Saving trained model to disk...")
        joblib.dump(self.model_data, self.model_path, compress=3)
        logger.info("Model saved successfully.")
        
        return self.model_data

    def predict(self, data: dict) -> float:
        """Predict SoH using the loaded ML model and encoders."""
        encoders = self.model_data["encoders"]
        feature_names = self.model_data["feature_names"]
        
        row = {}
        for feat in feature_names:
            if feat in ("Car_Model", "Battery_Type", "Driving_Style"):
                try:
                    le = encoders[feat]
                    row[feat] = int(le.transform([data[feat]])[0])
                except ValueError:
                    raise ValueError(f"Invalid value for '{feat}': {data[feat]}")
            else:
                row[feat] = float(data[feat])

        input_df = pd.DataFrame([row], columns=feature_names)
        prediction = self.model_data["model"].predict(input_df)[0]
        return round(float(prediction), 4)

    @staticmethod
    def determine_status(soh: float) -> str:
        if soh >= 90: return "Healthy"
        if soh >= 80: return "Moderate"
        return "Critical"
