import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# Load Model
with open("ev_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="EV Smart Dashboard", page_icon="🔋", layout="wide")

st.markdown("<h1 style='text-align:center;'>⚡ EV Smart Range Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)

speed = col1.slider("Vehicle Speed (km/h)", 0, 140, 60)
temperature = col2.slider("Temperature (°C)", -10, 60, 25)
battery = col3.slider("Battery Charge (%)", 5, 100, 80)

input_data = np.array([[speed, temperature, battery]])
predicted_range = model.predict(input_data)[0]
usable_range = (battery / 100) * predicted_range

st.markdown("### 🔋 Range Estimation")

gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=usable_range,
    title={'text': "Estimated Usable Range (km)"},
    gauge={'axis': {'range': [0, 350]}}
))

st.plotly_chart(gauge_fig, use_container_width=True)

st.markdown("---")

colA, colB = st.columns(2)

colA.metric("Full Range (if 100% battery)", f"{predicted_range:.1f} km")
colB.metric("Usable Range (current %)", f"{usable_range:.1f} km")

st.markdown("---")
st.markdown("### 🚗 Driving Efficiency Notes")

if speed > 90:
    st.warning("⚠️ Driving at high speed reduces range due to aerodynamic drag.")

if temperature < 10 or temperature > 35:
    st.warning("🌡️ Extreme temperature affects battery performance.")

if battery < 20:
    st.error("🔴 Low Battery! Consider planning charging soon.")

st.success("✅ Prediction Completed Successfully.")
st.caption("Developed for Internship Project – EV Energy Consumption Modeling")
