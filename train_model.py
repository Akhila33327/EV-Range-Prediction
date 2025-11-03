import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Battery capacity you assume (Change if needed)
BATTERY_CAPACITY_KWH = 30.2   # Tata Nexon EV Standard Battery

# Load dataset
df = pd.read_csv("EV.csv")  # <-- Ensure your dataset is named EV.csv

# Remove missing values
df = df.dropna()

# Feature Engineering
df['km_per_kwh'] = df['Distance_Travelled_km'] / df['Energy_Consumption_kWh']
df['range_km'] = (df['Battery_State_%'] / 100) * df['km_per_kwh'] * BATTERY_CAPACITY_KWH

# Select features and label
X = df[['Speed_kmh', 'Temperature_C', 'Battery_State_%']]
y = df['range_km']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Save Model
with open("ev_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model Training Complete.")
print("✅ Model Saved as ev_model.pkl")
print("✅ You can now run: streamlit run app.py")
