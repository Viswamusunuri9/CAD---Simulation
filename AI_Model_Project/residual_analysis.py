# residual analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# === STEP 1: Load dataset and prepare features ===
print("📂 Loading dataset...")
df = pd.read_csv("ai_surrogate_dataset.csv")
X = df.drop(columns=["Max_Stress_MPa", "Max_Deformation_mm"])
y = df[["Max_Stress_MPa", "Max_Deformation_mm"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === STEP 2: Load all models ===
print("📥 Loading trained models...")
models = {
    "ANN_Baseline": joblib.load("ann_surrogate_model.pkl"),
    "ANN_Tuned": joblib.load("tuned_ann_model.pkl"),
    "RandomForest": joblib.load("rf_surrogate_model.pkl")
}

residual_data = []

# === STEP 3: Predict and calculate residuals ===
for name, model in models.items():
    print(f"🔍 Calculating residuals for {name}...")
    y_pred = model.predict(X_test)

    residuals_stress = y_test["Max_Stress_MPa"].values - y_pred[:, 0]
    residuals_deformation = y_test["Max_Deformation_mm"].values - y_pred[:, 1]

    for i in range(len(residuals_stress)):
        residual_data.append({
            "Model": name,
            "Residual_Stress": residuals_stress[i],
            "Residual_Deformation": residuals_deformation[i],
            "Actual_Stress": y_test["Max_Stress_MPa"].values[i],
            "Actual_Deformation": y_test["Max_Deformation_mm"].values[i]
        })

# Save as CSV
df_residuals = pd.DataFrame(residual_data)
df_residuals.to_csv("residuals_summary.csv", index=False)
print("✅ Saved residuals to residuals_summary.csv")

# === STEP 4: Plot residual histograms ===
print("📊 Generating residual plots...")

for model in df_residuals["Model"].unique():
    subset = df_residuals[df_residuals["Model"] == model]

    # Histogram: Residuals - Stress
    plt.figure()
    sns.histplot(subset["Residual_Stress"], kde=True, bins=10, color='skyblue')
    plt.title(f"{model}: Residual Histogram (Stress)")
    plt.xlabel("Residual = Actual - Predicted (MPa)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"residual_hist_{model}_stress.png")

    # Histogram: Residuals - Deformation
    plt.figure()
    sns.histplot(subset["Residual_Deformation"], kde=True, bins=10, color='lightgreen')
    plt.title(f"{model}: Residual Histogram (Deformation)")
    plt.xlabel("Residual = Actual - Predicted (mm)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"residual_hist_{model}_deformation.png")

    # Scatter: Actual vs Residual (Stress)
    plt.figure()
    plt.scatter(subset["Actual_Stress"], subset["Residual_Stress"], color='blue')
    plt.axhline(0, linestyle='--', color='black')
    plt.title(f"{model}: Residuals vs Actual (Stress)")
    plt.xlabel("Actual Stress (MPa)")
    plt.ylabel("Residual Stress (MPa)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"residual_scatter_{model}_stress.png")

    # Scatter: Actual vs Residual (Deformation)
    plt.figure()
    plt.scatter(subset["Actual_Deformation"], subset["Residual_Deformation"], color='green')
    plt.axhline(0, linestyle='--', color='black')
    plt.title(f"{model}: Residuals vs Actual (Deformation)")
    plt.xlabel("Actual Deformation (mm)")
    plt.ylabel("Residual Deformation (mm)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"residual_scatter_{model}_deformation.png")

# Boxplot comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_residuals, x="Model", y="Residual_Stress", palette="Set2")
plt.title("Residual Stress Distribution per Model")
plt.grid(True)
plt.tight_layout()
plt.savefig("residual_boxplot_stress.png")

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_residuals, x="Model", y="Residual_Deformation", palette="Set3")
plt.title("Residual Deformation Distribution per Model")
plt.grid(True)
plt.tight_layout()
plt.savefig("residual_boxplot_deformation.png")

print("✅ Residual analysis complete. All plots and CSV are saved.")
