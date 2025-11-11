import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import utils 
import sys # For clean error handling

# --- Configuration ---
MODEL_FILENAME = "final_ev_consumption_model_xgb.pkl" 
MAE_FILENAME = "model_mae_consumption.pkl" 
SHAP_IMPORTANCE_FILE = "model_shap_importance.json" 

@st.cache_resource
def load_resources():
    """Load the model, config, MAE, and feature importance files."""
    try:
        model = joblib.load(MODEL_FILENAME)
        config = utils.load_config() 
        mae = joblib.load(MAE_FILENAME)
        
        with open(SHAP_IMPORTANCE_FILE, 'r') as f:
            importance_data = json.load(f) 
            
        return model, config, mae, pd.DataFrame(importance_data)
    except FileNotFoundError as e:
        st.error(f"Error loading required files. Please run train_model.py first. Missing file: {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during resource loading: {e}")
        return None, None, None, None

def plot_gauge(value, max_value, title):
    """Creates a Plotly Gauge Chart for Range."""
    display_value = min(value, max_value)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_value * 0.15], 'color': 'red'},
                {'range': [max_value * 0.15, max_value * 0.35], 'color': 'orange'},
                {'range': [max_value * 0.35, max_value], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.15
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def display_range_results(total_capacity, predicted_consumption, distance, battery_health, reserve_pct, mae):
    """Calculates and displays all primary results."""
    
    remaining_range_km = utils.calculate_realistic_range(
        total_capacity, predicted_consumption, distance, battery_health, reserve_pct
    )
    consumption_rate_kwh_per_km = predicted_consumption / distance
    
    total_theoretical_range = total_capacity / consumption_rate_kwh_per_km
    
    st.subheader("Final Range Prediction")
    
    col_gauge, col_metrics = st.columns([1.5, 2])

    with col_gauge:
        gauge_fig = plot_gauge(
            remaining_range_km, 
            total_theoretical_range, 
            "Remaining Range (km)"
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_metrics:
        st.markdown("### Performance Metrics")
        col_m1, col_m2 = st.columns(2)
        
        col_m1.metric(
            label="Total Possible Range (km)", 
            value=f"{total_theoretical_range:,.0f} km"
        )
        col_m2.metric(
            label="Effective Battery Capacity", 
            value=f"{total_capacity * (battery_health / 100.0):.1f} kWh"
        )
        
        st.markdown("### Consumption & Error")
        col_m3, col_m4 = st.columns(2)
        
        col_m3.metric(
            label=f"Predicted Consumption / {distance} km", 
            value=f"{predicted_consumption:.3f} KWh"
        )
        col_m4.metric(
            label="Model MAE (Consumption Error)", 
            value=f"±{mae:.4f} KWh",
            delta_color="off"
        )
        
    st.info(
        f"The prediction incorporates **{battery_health}% Battery Health** and a **{reserve_pct}% Reserve** factor, leading to an estimated remaining range of **{remaining_range_km:,.0f} km**."
    )


def display_shap_importance(importance_df):
    """Displays the SHAP feature importance visualization."""
    st.subheader("Model Explainability: Global SHAP Importance")
    st.markdown("SHAP values show the average magnitude of impact each feature has on the prediction.")
    
    # Plotly visualization logic
    fig = px.bar(
        importance_df.head(10), 
        x='SHAP_Importance', 
        y='Feature', 
        orientation='h',
        title='Global Feature Importance (Mean Absolute SHAP)',
        labels={'Feature': 'Feature Name', 'SHAP_Importance': 'Average Impact Magnitude'},
        color='SHAP_Importance',
        color_continuous_scale=px.colors.sequential.Teal
    )
    fig.update_layout(yaxis={'autorange': "reversed"}, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Streamlit application main function."""
    st.set_page_config(
        page_title="⚡️ Professional EV Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    model, features_config, mae, importance_df = load_resources()
    if model is None:
        st.stop()

    st.title("⚡️ EV Consumption & Range Predictor: Professional Dashboard")
    st.markdown("XGBoost Model with SHAP Interpretability.")
    
    # --- Sidebar for Configuration and Vehicle Info ---
    with st.sidebar:
        st.header("⚙️ Vehicle & Reserve Configuration")
        total_battery_capacity = st.number_input("Total Battery Capacity (kWh)", value=60.0, min_value=10.0, max_value=200.0, step=0.5, help="Total nominal energy storage capacity.")
        battery_health_pct = st.slider("Battery Health (%)", value=90, min_value=50, max_value=100, step=1, help="Current state of battery degradation.")
        reserve_pct = st.slider("Range Reserve (%)", value=5, min_value=0, max_value=20, step=1, help="Percentage of total range reserved for safety.")
        st.markdown("---")
        st.caption("Model Version: XGBoost 1.5.0")


    # --- Main Tabs ---
    tab_input, tab_analysis = st.tabs(["1️⃣ Input & Predict Range", "2️⃣ Model Analysis & SHAP"])

    with tab_input:
        st.subheader("Input Driving & Environmental Factors")
        
        # Define Input Columns
        col_driving, col_environmental, col_trip = st.columns(3)

        with col_driving:
            st.markdown("##### 🚗 Driving & Weight")
            speed = st.number_input("Average Speed (km/h)", value=70.0, min_value=0.0, max_value=150.0, step=1.0)
            vehicle_weight = st.number_input("Vehicle Weight (kg)", value=1800.0, min_value=500.0, max_value=4000.0, step=10.0)
            road_type = st.selectbox("Road Type", options=["1", "2", "3", "4"])
        
        with col_environmental:
            st.markdown("##### 🌡️ Environment & Power")
            ambient_temp = st.slider("Ambient Temperature (°C)", value=20, min_value=-20, max_value=45, step=1)
            battery_vol = st.number_input("Battery Voltage (Vol)", value=390.0, min_value=200.0, max_value=800.0, step=10.0)
        
        with col_trip:
            st.markdown("##### ⛰️ Trip Factors")
            slope = st.number_input("Slope (%)", value=2.0, min_value=-10.0, max_value=10.0, step=0.1)
            distance = st.number_input("Distance Sample (km)", value=10.0, min_value=1.0, max_value=100.0, step=1.0)
            
        st.markdown("---")
        
        # --- Prediction Button ---
        if st.button("Calculate Estimated Range 🚀", type="primary"):
            
            # --- PROFESSIONAL VALIDATION CHECK ---
            if distance <= 0:
                st.error("Error: Distance must be a positive value to calculate consumption rate.")
                st.stop()
            if vehicle_weight <= 0:
                st.error("Error: Vehicle weight must be a positive value.")
                st.stop()
            # -------------------------------------

            # 1. Input Data Collection 
            input_data = {
                "speed_kmh": speed, "weather_c": ambient_temp, "battery_vol": battery_vol,
                "vehicle_we": vehicle_weight, "slope_%": slope, "road_type": road_type
            }
            
            input_df = pd.DataFrame([input_data])
            
            # 2. Apply Feature Engineering
            input_df_engineered = utils.feature_engineer(input_df)

            # 3. Select all necessary columns (Base + Engineered)
            all_input_cols = (
                features_config['numerical_features'] + 
                features_config['categorical_features'] +
                features_config['new_engineered_features'] 
            )
            X_input = input_df_engineered[all_input_cols]

            # 4. Prediction
            try:
                predicted_consumption = model.predict(X_input)[0]
                predicted_consumption = max(0.001, predicted_consumption) 
                
                # 5. Store results in session state and switch to analysis tab
                st.session_state['show_results'] = True
                st.session_state['predicted_consumption'] = predicted_consumption
                st.session_state['input_distance'] = distance
                st.session_state['total_capacity'] = total_battery_capacity
                st.session_state['battery_health'] = battery_health_pct
                st.session_state['reserve_pct'] = reserve_pct
                # Corrected command for Streamlit
                st.rerun() 

            except Exception as e:
                st.error(f"Prediction failed. Error: {e}")
                sys.stderr.write(f"Prediction error occurred: {e}\n") # Log the error

    # --- Analysis Tab ---
    with tab_analysis:
        if 'show_results' in st.session_state and st.session_state['show_results']:
            # Display primary metrics
            display_range_results(
                st.session_state['total_capacity'],
                st.session_state['predicted_consumption'],
                st.session_state['input_distance'],
                st.session_state['battery_health'],
                st.session_state['reserve_pct'],
                mae
            )
            st.markdown("---")
            # Display SHAP importance
            display_shap_importance(importance_df)
        else:
            st.info("👈 Please enter your parameters in the 'Input & Predict Range' tab and click 'Calculate' to view the analysis.")

    st.markdown("---")
    st.caption("© 2025 EV Range Prediction Project. Model trained using Scikit-learn and XGBoost.")

if __name__ == "__main__":
    main()