# app.py
"""
Run with:
    streamlit run app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from fpdf import FPDF

# Files saved by train_model.py
MODEL_FILE = "model_rf.pkl"
FEATURES_FILE = "model_features.json"

# Multipliers and presets
DRIVING_STYLE_MULT = {"Eco": 0.85, "Normal": 1.0, "Aggressive": 1.18}
ROAD_TYPE_MULT = {"Highway": 0.90, "City": 1.15, "Mixed": 1.0}

VEHICLES = {
    "Tata Nexon EV (30.2 kWh)": 30.2,
    "MG ZS EV (44.5 kWh)": 44.5,
    "Hyundai Kona EV (39.2 kWh)": 39.2,
    "Custom (enter kWh)": None,
}

# Validate model files
if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_FILE):
    st.error("Model files not found. Run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_FILE)
with open(FEATURES_FILE, "r") as f:
    MODEL_FEATURES = json.load(f)

st.set_page_config(page_title="EV Smart Dashboard", layout="wide", page_icon="🔋")
st.markdown("<h1 style='text-align:center;'>⚡ EV Smart Range Prediction — Final</h1>", unsafe_allow_html=True)
st.write("## Vehicle & Scenario")

# Sidebar: vehicle + scenario
with st.sidebar:
    st.header("Vehicle & Scenario")
    vehicle_choice = st.selectbox("Select vehicle (or custom)", list(VEHICLES.keys()))
    if VEHICLES[vehicle_choice] is None:
        battery_capacity = st.number_input("Enter battery capacity (kWh)", min_value=1.0, value=30.2, step=0.1)
    else:
        battery_capacity = VEHICLES[vehicle_choice]

    st.subheader("Behavior")
    driving_style = st.selectbox("Driving style", ["Normal", "Eco", "Aggressive"])
    road_type = st.selectbox("Road type", ["Mixed", "City", "Highway"])
    st.markdown("---")
    st.write("Model files:")
    st.write(f"- `{MODEL_FILE}`")
    st.write(f"- `{FEATURES_FILE}`")

st.markdown("### Inputs")
col1, col2, col3, col4 = st.columns(4)
with col1:
    speed = st.slider("Vehicle Speed (km/h)", 0, 140, 60)
with col2:
    acceleration = st.slider("Acceleration (m/s²)", -3.0, 5.0, 0.5, step=0.1)
with col3:
    soc = st.slider("Battery State of Charge (%)", 1, 100, 80)
with col4:
    temp = st.slider("Ambient Temperature (°C)", -10, 45, 25)

col5, col6 = st.columns(2)
with col5:
    slope = st.number_input("Road slope (%)", value=0.0, step=0.1)
with col6:
    weight = st.number_input("Vehicle weight (kg)", value=1500, step=10)

st.markdown("---")

# Build feature vector consistent with model features
def build_input_row():
    # base numeric map (use reasonable defaults for missing fields)
    numeric = {
        "Speed_kmh": speed,
        "Acceleration_ms2": acceleration,
        "Battery_State_%": soc,
        "Battery_Voltage_V": 0.0,
        "Battery_Temperature_C": 0.0,
        "Slope_%": slope,
        "Temperature_C": temp,
        "Humidity_%": 0.0,
        "Wind_Speed_ms": 0.0,
        "Tire_Pressure_psi": 0.0,
        "Vehicle_Weight_kg": weight,
    }
    # categorical dummies: set zeros then set driving/road if present
    cat_cols = {}
    for feat in MODEL_FEATURES:
        if feat.startswith("Driving_Mode_") or feat.startswith("Road_Type_") or feat.startswith("Traffic_Condition_") or feat.startswith("Weather_Condition_"):
            cat_cols[feat] = 0

    # set driving style column name if present
    dm = f"Driving_Mode_{driving_style}"
    if dm in cat_cols:
        cat_cols[dm] = 1
    rt = f"Road_Type_{road_type}"
    if rt in cat_cols:
        cat_cols[rt] = 1

    # combine numeric + categorical into a row aligned to MODEL_FEATURES
    row = {}
    row.update(numeric)
    row.update(cat_cols)
    # ensure all MODEL_FEATURES exist; fill zeros for missing
    for f in MODEL_FEATURES:
        row.setdefault(f, 0.0)
    # create DataFrame with ordered columns
    df_row = pd.DataFrame([row], columns=MODEL_FEATURES)
    df_row = df_row.fillna(0.0)
    return df_row

input_df = build_input_row()

# Predict consumption kWh per 100 km
pred_kwh_per_100km = float(model.predict(input_df.values)[0])

# Apply multipliers (driving style & road type) and environmental adjustments
style_mult = DRIVING_STYLE_MULT.get(driving_style, 1.0)
road_mult = ROAD_TYPE_MULT.get(road_type, 1.0)

# temperature effect
temp_mult = 1.0
if temp < 5:
    temp_mult += 0.25  # cold penalty
elif temp > 35:
    temp_mult += 0.10  # hot penalty

# slope penalty
slope_mult = 1.0
if slope > 4:
    slope_mult += 0.12

# speed penalty for very high speeds
speed_mult = 1.0
if speed > 120:
    speed_mult += 0.20
elif speed > 90:
    speed_mult += 0.08

adjusted_consumption = pred_kwh_per_100km * style_mult * road_mult * temp_mult * slope_mult * speed_mult

# convert to kWh per km
consumption_kwh_per_km = adjusted_consumption / 100.0
eff_km_per_kwh = (1.0 / consumption_kwh_per_km) if consumption_kwh_per_km > 0 else 0.0

full_range_km = battery_capacity / consumption_kwh_per_km if consumption_kwh_per_km > 0 else 0.0
usable_range_km = full_range_km * (soc / 100.0)

# Show metrics
st.markdown("## 🔋 Range & Efficiency")
c1, c2, c3 = st.columns(3)
c1.metric("Predicted consumption (kWh/100km)", f"{pred_kwh_per_100km:.2f}")
c2.metric("Adjusted consumption (kWh/100km)", f"{adjusted_consumption:.2f}")
c3.metric("Efficiency (km/kWh)", f"{eff_km_per_kwh:.2f}")

# Gauge
g = go.Figure(go.Indicator(
    mode="gauge+number",
    value=usable_range_km,
    title={'text': "Estimated Usable Range (km)"},
    gauge={'axis': {'range': [0, max(200, battery_capacity*8)]},
           'bar': {'color': "green" if usable_range_km > 100 else "orange" if usable_range_km>30 else "red"}}
))
st.plotly_chart(g, use_container_width=True)

st.markdown("### Details")
st.write(f"- Vehicle chosen: **{vehicle_choice}** (battery = {battery_capacity:.1f} kWh)")
st.write(f"- Driving style multiplier: **{style_mult:.2f}** ; Road type multiplier: **{road_mult:.2f}**")
st.write(f"- Environmental multipliers: temp {temp_mult:.2f}, slope {slope_mult:.2f}, speed {speed_mult:.2f}")
st.write(f"- Full battery range (100%): **{full_range_km:.1f} km**")
st.write(f"- Estimated usable range ({soc}%): **{usable_range_km:.1f} km**")

# Recommendations
st.markdown("### 🧭 Recommendations")
recs = []
if driving_style == "Aggressive":
    recs.append("Aggressive driving increases consumption. Smooth acceleration and maintain steady speed to save energy.")
if speed > 100:
    recs.append("High speed increases aerodynamic losses. Reduce speed to increase range.")
if temp < 5:
    recs.append("Cold conditions reduce battery efficiency — precondition cabin when plugged in to improve range.")
if soc < 20:
    recs.append("Battery low (<20%). Plan a charge stop soon.")
if slope > 4:
    recs.append("Significant uphill slope detected — expect higher consumption.")
if not recs:
    recs.append("Conditions look normal. Use eco driving to maximize range.")

for r in recs:
    st.info(r)

# PDF report creation
def create_pdf(filename="EV_range_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "EV Range Prediction Report", ln=True, align="C")
    pdf.ln(6)
    pdf.cell(0, 8, f"Vehicle: {vehicle_choice} (battery {battery_capacity:.1f} kWh)", ln=True)
    pdf.cell(0, 8, f"Speed: {speed} km/h, Accel: {acceleration} m/s^2, Temp: {temp} °C, SOC: {soc}%", ln=True)
    pdf.cell(0, 8, f"Predicted consumption (kWh/100km): {pred_kwh_per_100km:.3f}", ln=True)
    pdf.cell(0, 8, f"Adjusted consumption (kWh/100km): {adjusted_consumption:.3f}", ln=True)
    pdf.cell(0, 8, f"Full range (100%): {full_range_km:.2f} km", ln=True)
    pdf.cell(0, 8, f"Estimated usable range ({soc}%): {usable_range_km:.2f} km", ln=True)
    pdf.ln(6)
    pdf.cell(0, 8, "Recommendations:", ln=True)
    for r in recs:
        pdf.multi_cell(0, 8, f"- {r}")
    pdf.output(filename)
    return filename

if st.button("Download PDF Report"):
    path = create_pdf()
    with open(path, "rb") as f:
        st.download_button("Click to download", f, file_name="EV_range_report.pdf", mime="application/pdf")

st.markdown("---")
st.caption("Model: RandomForestRegressor trained on dataset. Results are estimates — perform field validation.")
