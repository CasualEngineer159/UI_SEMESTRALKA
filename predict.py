import pandas as pd
import joblib
from data_process import process_data

# load artefacts
print("Načítám model...")
mlp = joblib.load('model_mlp.pkl')
scaler = joblib.load('model_scaler.pkl')
model_columns = joblib.load('model_columns.pkl')
le = joblib.load('model_le.pkl')

# input data
df_input = pd.read_csv("MHMP_dopravni_prestupky_2024.csv")

# process data
df_clean = process_data(df_input)

# One-Hot Encoding
df_encoded = pd.get_dummies(df_clean)

# rearrange columns (Reindexing), fills 0 where missing
df_ready = df_encoded.reindex(columns=model_columns, fill_value=0)

# scaling
df_scaled = scaler.transform(df_ready)

# Prediction
prediction_idx = mlp.predict(df_scaled)[0]
prediction_name = le.inverse_transform([prediction_idx])[0]
probability = mlp.predict_proba(df_scaled).max()

print(f"-----------------------------")
print(f"Výsledek analýzy: {prediction_name}")
print(f"Jistota modelu:   {probability:.2%}")
print(f"-----------------------------")