import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from data_process import process_data # Importujeme naši knihovnu!

# 1. load data and process them
print("Načítám data...")
raw_df = pd.read_csv("MHMP_dopravni_prestupky_2023.csv")

print("Čistím data...")
df_clean = process_data(raw_df)

# 2. prepare X and y
X_raw = df_clean.drop(columns=["OZNAM"])
y_raw = df_clean["OZNAM"]

# Encoder for target (PČR/MPP -> 1/0)
le = LabelEncoder()
y = le.fit_transform(y_raw)

# One-Hot Encoding for inputs
X_encoded = pd.get_dummies(X_raw, drop_first=True)

# 3. save names od columns
model_columns = list(X_encoded.columns)

# 4. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Training
print("Trénuji MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)
print(f"Trénink hotov. Skóre na testu: {mlp.score(scaler.transform(X_test), y_test):.4f}")

# 6. Export
print("Ukládám model...")
joblib.dump(mlp, 'model_mlp.pkl')
joblib.dump(scaler, 'model_scaler.pkl')
joblib.dump(model_columns, 'model_columns.pkl')
joblib.dump(le, 'model_le.pkl')
print("HOTOVO.")