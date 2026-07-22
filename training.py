import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("loan_cleaned.csv")

# ==========================
# Clean Column Names
# ==========================
df.columns = df.columns.str.strip().str.lower()

print("Dataset Loaded Successfully!")
print("Columns:", df.columns.tolist())

# ==========================
# Find Target Column
# ==========================
possible_targets = [
    "loan_status",
    "loan status",
    "loanstatus",
    "status",
    "approved",
    "loan_approved"
]

target = None

for col in possible_targets:
    if col in df.columns:
        target = col
        break

if target is None:
    print("\nError: Target column not found!")
    print("Available Columns:", df.columns.tolist())
    exit()

print(f"\nTarget Column: {target}")

# ==========================
# Features & Target
# ==========================
X = df.drop(columns=[target])
y = df[target]

# ==========================
# Feature Scaling
# ==========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================
# Train/Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# ==========================
# Train Model
# ==========================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================
# Prediction
# ==========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("==============================")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================
# Save Model (Compressed)
# ==========================
joblib.dump(model, "loan_model.pkl", compress=9)
joblib.dump(scaler, "scaler.pkl", compress=9)

print("\nModel Saved Successfully as loan_model.pkl")
print("Scaler Saved Successfully as scaler.pkl")