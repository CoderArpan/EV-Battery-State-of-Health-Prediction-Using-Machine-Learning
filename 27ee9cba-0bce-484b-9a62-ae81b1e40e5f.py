# ============================================================
# Electric Vehicle Battery State of Health Prediction
# Author : Refactored Version
# ============================================================

import warnings
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------- Logging ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s : %(message)s"
)

# ---------------- Constants ---------------- #

DATASET_PATH = "archive (2) (1).csv"

MODEL_PATH = "battery_soh_model.pkl"

RANDOM_STATE = 42

TARGET_COLUMN = "SoH_Percent"

DROP_COLUMNS = [
    "Vehicle_ID"
]

# ============================================================
# Load Dataset
# ============================================================

def load_dataset(path: str) -> pd.DataFrame:

    logging.info("Loading dataset...")

    file = Path(path)

    if not file.exists():
        raise FileNotFoundError(
            f"Dataset not found : {path}"
        )

    df = pd.read_csv(file)

    logging.info(
        f"Dataset Loaded Successfully : {df.shape}"
    )

    return df


# ============================================================
# Dataset Information
# ============================================================

def inspect_dataset(df: pd.DataFrame):

    print("\nFirst Five Rows\n")
    print(df.head())

    print("\nDataset Shape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nStatistical Summary")
    print(df.describe())


# ============================================================
# Clean Dataset
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    logging.info("Cleaning dataset...")

    duplicates = df.duplicated().sum()

    if duplicates > 0:
        logging.info(f"Removing {duplicates} duplicate rows")
        df = df.drop_duplicates()

    if "Vehicle_ID" in df.columns:
        df.drop("Vehicle_ID", axis=1, inplace=True)

    logging.info("Cleaning completed")

    return df


# ============================================================
# Encode Categorical Columns
# ============================================================

def encode_features(df: pd.DataFrame):

    encoders = {}

    categorical_columns = [

        "Car_Model",

        "Battery_Type",

        "Driving_Style",

        "Battery_Status"

    ]

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(df[column])

        encoders[column] = encoder

    logging.info("Categorical columns encoded")

    return df, encoders


# ============================================================
# Prepare Training Data
# ============================================================

def prepare_data(df: pd.DataFrame):

    X = df.drop(
        [TARGET_COLUMN, "Battery_Status"],
        axis=1
    )

    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=RANDOM_STATE

    )

    logging.info(
        f"Training Samples : {len(X_train)}"
    )

    logging.info(
        f"Testing Samples : {len(X_test)}"
    )

    return X_train, X_test, y_train, y_test, X


# ============================================================
# Visualization Theme
# ============================================================

plt.style.use("ggplot")

sns.set_context("talk")

# ============================================================
# Exploratory Data Analysis (EDA)
# ============================================================

def plot_correlation_heatmap(df: pd.DataFrame):

    plt.figure(figsize=(16, 12))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cmap="coolwarm",
        linewidths=0.5,
        square=True
    )

    plt.title("Correlation Heatmap", fontsize=16)

    plt.xticks(rotation=45, ha="right", fontsize=9)

    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------

def plot_soh_distribution(df: pd.DataFrame):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[TARGET_COLUMN],
        bins=30,
        kde=True,
        color="steelblue"
    )

    plt.title("Distribution of Battery State of Health")

    plt.xlabel("SoH (%)")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------

def scatter_plot(
    df: pd.DataFrame,
    feature: str,
    title: str
):

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x=feature,
        y=TARGET_COLUMN,
        alpha=0.75
    )

    plt.title(title)

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------

def battery_status_distribution(df: pd.DataFrame):

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x="Battery_Status"
    )

    plt.title("Battery Status Distribution")

    plt.xlabel("Battery Status")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------

def feature_distributions(df: pd.DataFrame):

    numerical_columns = [

        "Battery_Capacity_kWh",

        "Vehicle_Age_Months",

        "Total_Charging_Cycles",

        "Avg_Temperature_C",

        "Fast_Charge_Ratio",

        "Avg_Discharge_Rate_C",

        "Internal_Resistance_Ohm"

    ]

    for column in numerical_columns:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            df[column],
            kde=True,
            bins=25
        )

        plt.title(f"{column} Distribution")

        plt.tight_layout()

        plt.show()


# ------------------------------------------------------------

def feature_boxplots(df: pd.DataFrame):

    numerical_columns = [

        "Battery_Capacity_kWh",

        "Vehicle_Age_Months",

        "Total_Charging_Cycles",

        "Avg_Temperature_C",

        "Fast_Charge_Ratio",

        "Avg_Discharge_Rate_C",

        "Internal_Resistance_Ohm"

    ]

    for column in numerical_columns:

        plt.figure(figsize=(8, 4))

        sns.boxplot(
            x=df[column]
        )

        plt.title(f"{column} Boxplot")

        plt.tight_layout()

        plt.show()


# ------------------------------------------------------------

def perform_eda(df: pd.DataFrame):

    logging.info("Performing Exploratory Data Analysis...")

    plot_correlation_heatmap(df)

    plot_soh_distribution(df)

    scatter_plot(
        df,
        "Vehicle_Age_Months",
        "Vehicle Age vs Battery Health"
    )

    scatter_plot(
        df,
        "Total_Charging_Cycles",
        "Charging Cycles vs Battery Health"
    )

    scatter_plot(
        df,
        "Avg_Temperature_C",
        "Temperature vs Battery Health"
    )

    scatter_plot(
        df,
        "Fast_Charge_Ratio",
        "Fast Charge Ratio vs Battery Health"
    )

    battery_status_distribution(df)

    feature_distributions(df)

    feature_boxplots(df)

    logging.info("EDA Completed Successfully.")


# ============================================================
# Model Training
# ============================================================

def train_model(X_train, y_train):

    logging.info("Training Random Forest Model...")

    model = RandomForestRegressor(
        random_state=RANDOM_STATE
    )

    parameters = {

        "n_estimators": [100, 150, 200],

        "max_depth": [None, 10, 20],

        "min_samples_split": [2, 5],

        "min_samples_leaf": [1, 2]

    }

    grid_search = GridSearchCV(

        estimator=model,

        param_grid=parameters,

        cv=5,

        scoring="r2",

        n_jobs=-1

    )

    grid_search.fit(X_train, y_train)

    logging.info("Model Training Completed")

    logging.info(f"Best Parameters : {grid_search.best_params_}")

    logging.info(
        f"Best Cross Validation Score : {grid_search.best_score_:.4f}"
    )

    return grid_search.best_estimator_


# ============================================================
# Cross Validation
# ============================================================

def perform_cross_validation(model, X, y):

    logging.info("Performing Cross Validation...")

    scores = cross_val_score(

        model,

        X,

        y,

        cv=5,

        scoring="r2",

        n_jobs=-1

    )

    print("\nCross Validation Scores")

    for index, score in enumerate(scores, start=1):

        print(f"Fold {index} : {score:.4f}")

    print(f"\nAverage R² Score : {scores.mean():.4f}")

    print(f"Standard Deviation : {scores.std():.4f}")


# ============================================================
# Model Evaluation
# ============================================================

def evaluate_model(model, X_test, y_test):

    logging.info("Evaluating Model...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    print("\n========== MODEL PERFORMANCE ==========")

    print(f"Mean Absolute Error : {mae:.4f}")

    print(f"Mean Squared Error : {mse:.4f}")

    print(f"Root Mean Squared Error : {rmse:.4f}")

    print(f"R² Score : {r2:.4f}")

    return predictions


# ============================================================
# Actual vs Predicted Plot
# ============================================================

def plot_predictions(y_test, predictions):

    plt.figure(figsize=(8,6))

    plt.scatter(

        y_test,

        predictions,

        alpha=0.75

    )

    plt.plot(

        [y_test.min(), y_test.max()],

        [y_test.min(), y_test.max()],

        color="red",

        linestyle="--",

        linewidth=2

    )

    plt.xlabel("Actual SoH")

    plt.ylabel("Predicted SoH")

    plt.title("Actual vs Predicted Battery Health")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# Prediction Results Table
# ============================================================

def prediction_results(y_test, predictions):

    results = pd.DataFrame({

        "Actual SoH": y_test,

        "Predicted SoH": predictions

    })

    print("\nPrediction Results")

    print(results.head(10))

    return results


# ============================================================
# Feature Importance
# ============================================================

def feature_importance(model, X):

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    plt.figure(figsize=(10,6))

    sns.barplot(

        data=importance,

        x="Importance",

        y="Feature"

    )

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.show()

    print("\nMost Important Features\n")

    print(importance)

    return importance


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    df = load_dataset(DATASET_PATH)

    inspect_dataset(df)

    df = clean_dataset(df)

    df, encoders = encode_features(df)

    perform_eda(df)

    X_train, X_test, y_train, y_test, X = prepare_data(df)

    model = train_model(X_train, y_train)

    perform_cross_validation(model, X, df[TARGET_COLUMN])

    predictions = evaluate_model(model, X_test, y_test)

    plot_predictions(y_test, predictions)

    prediction_results(y_test, predictions)

    feature_importance(model, X)

    joblib.dump(model, MODEL_PATH)

    logging.info(f"Model saved to {MODEL_PATH}")