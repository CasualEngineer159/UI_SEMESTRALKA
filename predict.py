import pandas as pd
import joblib
from data_process import process_data

# load model and artifacts
print("Loading model and tools...")
mlp = joblib.load('model_mlp.pkl')
scaler = joblib.load('model_scaler.pkl')
model_columns = joblib.load('model_columns.pkl')
le = joblib.load('model_le.pkl')  # LabelEncoder (maps 0/1 back to class names)

# load original dataset
print("Loading dataset...")
df_orig = pd.read_csv("MHMP_dopravni_prestupky_2024.csv")

# process data
print("Processing data...")
df_clean = process_data(df_orig)

# Store the actual values for later comparison
reality_text = df_clean["OZNAM"]

# Prepare model inputs (drop the target column)
df_input = df_clean.drop(columns=["OZNAM"])
# Ensure columns match the training structure exactly
df_encoded = pd.get_dummies(df_input)
df_ready = df_encoded.reindex(columns=model_columns, fill_value=0)
df_scaled = scaler.transform(df_ready)

print("Running predictions...")
prediction_index = mlp.predict(df_scaled)

# Convert numerical predictions (0/1) back to text labels (e.g., MPP/PČR)
prediction_text = le.inverse_transform(prediction_index)

# Get prediction confidence scores (maximum probability)
probability = mlp.predict_proba(df_scaled).max(axis=1)

# --- COMPARISON ---
# Create a results DataFrame for evaluation
results = pd.DataFrame({
    "ACTUAL": reality_text,
    "PREDICTED": prediction_text,
    "CONFIDENCE": probability,
    # Context columns to understand the nature of the violation
    "LAW": df_clean["LAW_CLEAN"],
    "CAR": df_clean["CAR_TYPE"],
    "LOCATION": df_clean["PRAGUE"],
})

# Add a boolean column indicating if the prediction was correct
results["MATCH"] = results["ACTUAL"] == results["PREDICTED"]

# --- STATISTICS ---
total_count = len(results)
correct_count = results["MATCH"].sum()
accuracy = (correct_count / total_count) * 100

print(f"\n==========================================")
print(f"FULL DATASET EVALUATION RESULTS")
print(f"==========================================")
print(f"Total rows:      {total_count}")
print(f"Correctly pred.: {correct_count}")
print(f"Errors:          {total_count - correct_count}")
print(f"Accuracy:        {accuracy:.2f} %")
print(f"==========================================\n")

# --- ERROR ANALYSIS (Misclassified instances) ---
# Filter only rows where the model made a mistake
errors = results[results["MATCH"] == False]

if not errors.empty:
    print("Sample of 5 misclassified instances:")
    print(errors[["ACTUAL", "PREDICTED", "CONFIDENCE", "LAW", "CAR"]].head(5))

    # Save errors to CSV for manual inspection in Excel
    errors.to_csv("model_errors.csv", index=False)
    print("\nList of all errors saved to 'model_errors.csv'.")
else:
    print("Congratulations! The model made zero errors.")