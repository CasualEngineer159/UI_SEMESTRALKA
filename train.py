import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from data_process import process_data

# 1. load data and process them
print("loading data...")
raw_df = pd.read_csv("MHMP_dopravni_prestupky_2023.csv")

print("clearing data...")
df_clean = process_data(raw_df)

"""df_clean = pd.read_csv("2023_clean.csv")"""

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
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Training
print("training MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=50, random_state=42)
mlp.fit(X_train_scaled, y_train)
print(f"training completed. test score: {mlp.score(scaler.transform(X_test), y_test):.4f}")


# 6. Export
print("saving model...")
joblib.dump(mlp, 'model_mlp.pkl')
joblib.dump(scaler, 'model_scaler.pkl')
joblib.dump(model_columns, 'model_columns.pkl')
joblib.dump(le, 'model_le.pkl')
print("DONE.")

# Predikce na testovacích datech
y_pred = mlp.predict(scaler.transform(X_test))

# Výpis reportu
print("detail report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Vizuální matice záměn
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('model prediction')
plt.ylabel('reality')
plt.title('confusion matrix')
plt.show()