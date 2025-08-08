
# STEP 3: Train Linear Regression Model for Surrogate Simulation
# Author: ChatGPT for Viswa (M.Tech Thesis)

# 📦 Required Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 📂 STEP 3.1: Load Preprocessed Data
X_train = pd.read_csv("X_train.csv")
Y_train = pd.read_csv("Y_train.csv")
X_test = pd.read_csv("X_test.csv")
Y_test = pd.read_csv("Y_test.csv")

# ✅ STEP 3.2: Train the Model
model = LinearRegression()
model.fit(X_train, Y_train)

# ✅ STEP 3.3: Predict on Test Set
Y_pred = model.predict(X_test)

# ✅ STEP 3.4: Evaluate Model
mse_disp = mean_squared_error(Y_test["Max Displacement (mm)"], Y_pred[:, 0])
mse_stress = mean_squared_error(Y_test["Max Stress (MPa)"], Y_pred[:, 1])
r2_disp = r2_score(Y_test["Max Displacement (mm)"], Y_pred[:, 0])
r2_stress = r2_score(Y_test["Max Stress (MPa)"], Y_pred[:, 1])

print("📊 Model Evaluation:")
print(f"Max Displacement → MSE: {mse_disp:.4f}, R²: {r2_disp:.4f}")
print(f"Max Stress       → MSE: {mse_stress:.4f}, R²: {r2_stress:.4f}")

# ✅ STEP 3.5: Plot Actual vs Predicted
plt.figure(figsize=(10, 4))

# Plot 1: Displacement
plt.subplot(1, 2, 1)
plt.scatter(Y_test["Max Displacement (mm)"], Y_pred[:, 0], color='blue')
plt.plot([Y_test["Max Displacement (mm)"].min(), Y_test["Max Displacement (mm)"].max()],
         [Y_test["Max Displacement (mm)"].min(), Y_test["Max Displacement (mm)"].max()],
         color='red', linestyle='--')
plt.xlabel("Actual Displacement (mm)")
plt.ylabel("Predicted Displacement (mm)")
plt.title("Displacement: Actual vs Predicted")

# Plot 2: Stress
plt.subplot(1, 2, 2)
plt.scatter(Y_test["Max Stress (MPa)"], Y_pred[:, 1], color='green')
plt.plot([Y_test["Max Stress (MPa)"].min(), Y_test["Max Stress (MPa)"].max()],
         [Y_test["Max Stress (MPa)"].min(), Y_test["Max Stress (MPa)"].max()],
         color='red', linestyle='--')
plt.xlabel("Actual Stress (MPa)")
plt.ylabel("Predicted Stress (MPa)")
plt.title("Stress: Actual vs Predicted")

plt.tight_layout()
plt.savefig("linear_regression_results.png")
plt.show()
