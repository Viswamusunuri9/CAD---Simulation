# randomforest model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# === Step 1: Load and Prepare Dataset ===
print("📂 Loading dataset...")
df = pd.read_csv("ai_surrogate_dataset.csv")

X = df.drop(columns=["Max_Stress_MPa", "Max_Deformation_mm"])
y = df[["Max_Stress_MPa", "Max_Deformation_mm"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === Step 2: Train RandomForest Surrogate ===
print("🌲 Training RandomForest surrogate model...")
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# === Step 3: Save Model ===
joblib.dump(model, "rf_surrogate_model.pkl")
print("✅ Model saved as rf_surrogate_model.pkl")

# === Step 4: Predict and Evaluate ===
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"📈 R² Score: {r2:.4f}")
print(f"📉 MSE: {mse:.2f}")

# === Step 5: Save Evaluation Plots ===
# Stress
plt.figure()
plt.scatter(y_test["Max_Stress_MPa"], y_pred[:, 0], c='blue')
plt.plot([y_test["Max_Stress_MPa"].min(), y_test["Max_Stress_MPa"].max()],
         [y_test["Max_Stress_MPa"].min(), y_test["Max_Stress_MPa"].max()], 'k--')
plt.xlabel("Actual Max Stress (MPa)")
plt.ylabel("Predicted Max Stress (MPa)")
plt.title("RF: Stress - Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig("rf_stress_actual_vs_predicted.png")

# Deformation
plt.figure()
plt.scatter(y_test["Max_Deformation_mm"], y_pred[:, 1], c='green')
plt.plot([y_test["Max_Deformation_mm"].min(), y_test["Max_Deformation_mm"].max()],
         [y_test["Max_Deformation_mm"].min(), y_test["Max_Deformation_mm"].max()], 'k--')
plt.xlabel("Actual Max Deformation (mm)")
plt.ylabel("Predicted Max Deformation (mm)")
plt.title("RF: Deformation - Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig("rf_deformation_actual_vs_predicted.png")

# === Step 6: Feature Importance (Optional, for stress only)
importances = model.estimators_[0].feature_importances_
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance for Max_Stress_MPa (RF)")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
print("📊 Plots saved.")
