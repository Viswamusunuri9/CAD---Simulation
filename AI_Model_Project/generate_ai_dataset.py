import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Step 1: Load and Inspect Raw Simulation Data
print("📥 Loading simulation log...")
df = pd.read_csv("simulation_log.csv")

print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns")
print("🔍 Columns:", df.columns.tolist())

print("\nChecking for nulls:")
print(df.isnull().sum())

print("\nChecking for duplicates:")
print(df.duplicated().sum())

print("\nPreviewing first few rows:")
print(df.head())

# Step 2: Feature Engineering – Inputs
print("\n🎯 One-hot encoding 'Material'...")
df_encoded = pd.get_dummies(df, columns=["Material"])

input_cols = [
    "Length_mm", "Height_mm", "Width_mm", "Load_N",
    "Material_ABS", "Material_Carbon Fiber", "Material_Glass Fiber"
]
output_cols = ["Max_Stress_MPa", "Max_Deformation_mm"]

X = df_encoded[input_cols]
y = df_encoded[output_cols]

# Step 3: Feature Engineering – Outputs
print("\n📈 Output Description:")
print(y.describe())

if (y < 0).any().any():
    print("⚠️ WARNING: Negative values found in stress or deformation!")

# Step 4: Combine and Export Final AI Dataset
ai_dataset = pd.concat([X, y], axis=1)
ai_dataset.to_csv("ai_surrogate_dataset.csv", index=False)
print(f"\n✅ AI-ready dataset saved as 'ai_surrogate_dataset.csv'")
print(f"🧾 Final shape: {ai_dataset.shape}")

# Step 5: Integrity Checks
print("\n🔬 Step 5: Running integrity checks and saving visualizations...")

# 5.1 Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(ai_dataset.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

# 5.2 Output Distributions
sns.histplot(y["Max_Stress_MPa"], bins=10, kde=True)
plt.title("Distribution of Max Stress (MPa)")
plt.xlabel("Max_Stress_MPa")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("stress_distribution.png")
plt.close()

sns.histplot(y["Max_Deformation_mm"], bins=10, kde=True)
plt.title("Distribution of Max Deformation (mm)")
plt.xlabel("Max_Deformation_mm")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("deformation_distribution.png")
plt.close()

# 5.3 Boxplot by Material
df_box = df.copy()
plt.figure(figsize=(8, 6))
sns.boxplot(x="Material", y="Max_Deformation_mm", data=df_box)
plt.title("Material vs Max Deformation")
plt.tight_layout()
plt.savefig("material_deformation_boxplot.png")
plt.close()

# 5.4 Scatterplot: Load vs Deformation
palette = {"ABS": "blue", "Carbon Fiber": "red", "Glass Fiber": "green"}
sns.scatterplot(data=df_box, x="Load_N", y="Max_Deformation_mm", hue="Material", palette=palette)
plt.title("Load vs Max Deformation by Material")
plt.tight_layout()
plt.savefig("deformation_vs_load.png")
plt.close()

print("✅ Visualizations saved:")
print("- correlation_matrix.png")
print("- stress_distribution.png")
print("- deformation_distribution.png")
print("- material_deformation_boxplot.png")
print("- deformation_vs_load.png")
