# tuned_ann_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import joblib

# === STEP 1: LOAD DATA ===
df = pd.read_csv("ai_surrogate_dataset.csv")
X = df.drop(columns=["Max_Stress_MPa", "Max_Deformation_mm"])
y = df[["Max_Stress_MPa", "Max_Deformation_mm"]]

# === STEP 2: NORMALIZE FEATURES ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === STEP 3: TRAIN/TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === STEP 4: GRID SEARCH SETUP ===
param_grid = {
    'estimator__hidden_layer_sizes': [(32,), (64,), (32, 32), (64, 64)],
    'estimator__activation': ['relu', 'tanh'],
    'estimator__solver': ['adam', 'lbfgs'],
    'estimator__learning_rate_init': [0.001, 0.01]
}

base_model = MLPRegressor(max_iter=3000, random_state=42)

grid_model = GridSearchCV(
    estimator=MultiOutputRegressor(base_model),
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    verbose=2,
    n_jobs=-1
)

print("🔍 Running hyperparameter tuning...")
grid_model.fit(X_train, y_train)

# === STEP 5: BEST MODEL & EVALUATION ===
best_model = grid_model.best_estimator_
print("✅ Best Parameters Found:")
print(grid_model.best_params_)

# Predict
y_pred = best_model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"📈 R² Score: {r2:.4f}")
print(f"📉 MSE: {mse:.2f}")

# === STEP 6: SAVE MODEL ===
joblib.dump(best_model, "tuned_ann_model.pkl")
print("✅ Tuned model saved as tuned_ann_model.pkl")

# === STEP 7: PLOTS ===
# Plot 1: Stress
plt.figure()
plt.scatter(y_test["Max_Stress_MPa"], y_pred[:, 0], c='blue', label="Stress")
plt.plot([y_test["Max_Stress_MPa"].min(), y_test["Max_Stress_MPa"].max()],
         [y_test["Max_Stress_MPa"].min(), y_test["Max_Stress_MPa"].max()], 'k--')
plt.xlabel("Actual Max Stress (MPa)")
plt.ylabel("Predicted Max Stress (MPa)")
plt.title("Tuned: Stress - Actual vs Predicted")
plt.legend()
plt.grid()
plt.savefig("tuned_stress_actual_vs_predicted.png")

# Plot 2: Deformation
plt.figure()
plt.scatter(y_test["Max_Deformation_mm"], y_pred[:, 1], c='green', label="Deformation")
plt.plot([y_test["Max_Deformation_mm"].min(), y_test["Max_Deformation_mm"].max()],
         [y_test["Max_Deformation_mm"].min(), y_test["Max_Deformation_mm"].max()], 'k--')
plt.xlabel("Actual Max Deformation (mm)")
plt.ylabel("Predicted Max Deformation (mm)")
plt.title("Tuned: Deformation - Actual vs Predicted")
plt.legend()
plt.grid()
plt.savefig("tuned_deformation_actual_vs_predicted.png")

print("📊 Saved Plots:")
print("- tuned_stress_actual_vs_predicted.png")
print("- tuned_deformation_actual_vs_predicted.png")
