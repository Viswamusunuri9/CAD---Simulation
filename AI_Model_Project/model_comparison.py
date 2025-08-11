# model comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === STEP 1: LOAD AND PREPARE DATA ===
print("📂 Loading dataset...")
df = pd.read_csv("ai_surrogate_dataset.csv")
X = df.drop(columns=["Max_Stress_MPa", "Max_Deformation_mm"])
y = df[["Max_Stress_MPa", "Max_Deformation_mm"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === STEP 2: LOAD TRAINED MODELS ===
print("📥 Loading models...")
models = {
    "ANN_Baseline": joblib.load("ann_surrogate_model.pkl"),
    "ANN_Tuned": joblib.load("tuned_ann_model.pkl"),
    "RandomForest": joblib.load("rf_surrogate_model.pkl")
}

results = []

# === STEP 3: PREDICT AND EVALUATE EACH MODEL ===
for name, model in models.items():
    print(f"🔍 Evaluating {name}...")
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append({
        "Model": name,
        "R2_Score": r2,
        "MSE": mse,
        "MAE": mae
    })

# === STEP 4: SAVE RESULTS TO CSV ===
df_results = pd.DataFrame(results)
df_results.to_csv("model_comparison_results.csv", index=False)
print("✅ Saved comparison table to model_comparison_results.csv")

# === STEP 5: PLOT METRIC COMPARISONS ===
# R² Score
plt.figure()
sns.barplot(data=df_results, x="Model", y="R2_Score", palette="Set2")
plt.title("R² Score Comparison")
plt.ylabel("R² Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("r2_score_comparison.png")

# MSE
plt.figure()
sns.barplot(data=df_results, x="Model", y="MSE", palette="Set3")
plt.title("Mean Squared Error (MSE) Comparison")
plt.ylabel("MSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("mse_comparison.png")

# MAE
plt.figure()
sns.barplot(data=df_results, x="Model", y="MAE", palette="Set1")
plt.title("Mean Absolute Error (MAE) Comparison")
plt.ylabel("MAE")
plt.grid(True)
plt.tight_layout()
plt.savefig("mae_comparison.png")

print("📊 Saved plots:")
print("- r2_score_comparison.png")
print("- mse_comparison.png")
print("- mae_comparison.png")
