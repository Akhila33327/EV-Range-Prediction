# inspect_columns.py
import pandas as pd
import os

DATA_PATH = 'EV_Energy_Consumption_Dataset.csv'  # <-- change filename here if different

if not os.path.exists(DATA_PATH):
    print("ERROR: dataset file not found at:", DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH, nrows=5)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:\n")
    print(df.head().to_string())
