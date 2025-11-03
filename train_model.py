# train_model.py
"""
Train pipeline for EV energy consumption model.

Outputs:
 - model_rf.pkl         (trained RandomForest)
 - model_features.json  (feature order used)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_NAME = "EV.csv"  # make sure this file is present
MODEL_OUT = "model_rf.pkl"
FEATURES_OUT = "model_features.json"

# Columns expected in your CSV (as found earlier)
NUMERIC_COLS = [
    "Speed_kmh",
    "Acceleration_ms2",
    "Battery_State_%",
    "Battery_Voltage_V",
    "Battery_Temperature_C",
    "Slope_%",
    "Temperature_C",
    "Humidity_%",
    "Wind_Speed_ms",
    "Tire_Pressure_psi",
    "Vehicle_Weight_kg",
]
CATEGORICAL_COLS = [
    "Driving_Mode",
    "Road_Type",
    "Traffic_Condition",
    "Weather_Condition",
]


def load_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}. Place dataset in project folder.")
    df = pd.read_csv(path)
    return df


def preprocess(df):
    df = df.copy()
    # drop rows with zero or negative distance to avoid division by zero
    df = df[df["Distance_Travelled_km"] > 0].reset_index(drop=True)
    # create consumption target kWh per 100 km
    df["consumption_kwh_per_100km"] = (df["Energy_Consumption_kWh"] / df["Distance_Travelled_km"]) * 100.0
    df = df[df["consumption_kwh_per_100km"] > 0].reset_index(drop=True)

    # ensure categorical columns exist and fillna
    for c in CATEGORICAL_COLS:
        if c not in df.columns:
            df[c] = "Unknown"
        df[c] = df[c].fillna("Unknown").astype(str)

    # ensure numeric columns exist, coerce, fill with median
    for n in NUMERIC_COLS:
        if n not in df.columns:
            df[n] = 0.0
        else:
            df[n] = pd.to_numeric(df[n], errors="coerce")
            df[n] = df[n].fillna(df[n].median())

    # basic clipping
    df["Speed_kmh"] = df["Speed_kmh"].clip(lower=0, upper=250)
    df["consumption_kwh_per_100km"] = df["consumption_kwh_per_100km"].clip(lower=0.1)

    # one-hot encode categorical columns
    df_cat = pd.get_dummies(df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)
    X = pd.concat([df[NUMERIC_COLS], df_cat], axis=1)
    y = df["consumption_kwh_per_100km"].values

    return X, y, df


def train_and_save(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=24,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )

    print("Training RandomForestRegressor ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE (kWh/100km): {mae:.4f}")
    print(f"RMSE (kWh/100km): {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    # save model and feature order
    joblib.dump(model, MODEL_OUT)
    with open(FEATURES_OUT, "w") as f:
        json.dump(X.columns.tolist(), f)

    print(f"Saved {MODEL_OUT} and {FEATURES_OUT}")
    return model


if __name__ == "__main__":
    print("Loading CSV...")
    df = load_df(CSV_NAME)
    print("Preprocessing ...")
    X, y, _ = preprocess(df)
    print("Feature matrix shape:", X.shape)
    model = train_and_save(X, y)
    print("Done.")
