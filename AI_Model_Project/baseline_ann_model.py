# baseline_ann_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# === Step 1: Load Data ===
print("🔹 Loading dataset...")
df = pd.read_csv("ai_surrogate_dataset.csv")

X = df.drop(columns=["Max_Stress_MPa", "Max_Deformation_mm"])
y = df[["Max_Stress_MPa", "Max_Deformation_mm"]]

# === Step 2: Normalize Inputs ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 3: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Step 4: Train ANN Model ===
print("🔹 Training baseline ANN model...")
model = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=2000, random_state=42)
model.fit(X_train, y_train)

# === Step 5: Predict ===
y_pred = model.predict(X_test)

# === Step 6: Evaluate ===
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ MSE: {mse:.2f}")
print(f"✅ R² Score: {r2:.2f}")

# === Step 7: Save Model ===
joblib.dump(model, "ann_surrogate_model.pkl")
print("✅ Model saved as ann_surrogate_model.pkl")

# === Step 8: Plot Results ===
# Stress
plt.figure()
plt.scatter(y_test["Max_Stress_MPa"], y_pred[:, 0], c='blue', label="Stress")
plt.plot([y_test["Max_Stress_MPa"].min(), y_test["Max_Stress_MPa"].max()],
         [y_test["Max_Stress_MPa"].min(), y_test["Max_Stress_MPa"].max()], 'k--')
plt.xlabel("Actual Max Stress (MPa)")
plt.ylabel("Predicted Max Stress (MPa)")
plt.title("Stress: Actual vs Predicted")
plt.legend()
plt.grid()
plt.savefig("stress_actual_vs_predicted.png")

# Deformation
plt.figure()
plt.scatter(y_test["Max_Deformation_mm"], y_pred[:, 1], c='green', label="Deformation")
plt.plot([y_test["Max_Deformation_mm"].min(), y_test["Max_Deformation_mm"].max()],
         [y_test["Max_Deformation_mm"].min(), y_test["Max_Deformation_mm"].max()], 'k--')
plt.xlabel("Actual Max Deformation (mm)")
plt.ylabel("Predicted Max Deformation (mm)")
plt.title("Deformation: Actual vs Predicted")
plt.legend()
plt.grid()
plt.savefig("deformation_actual_vs_predicted.png")

print("📊 Plots saved:")
print("- stress_actual_vs_predicted.png")
print("- deformation_actual_vs_predicted.png")
