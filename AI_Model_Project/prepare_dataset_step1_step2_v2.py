
# STEP 1 + STEP 2 (FIXED): AI Surrogate Model Dataset Preparation
# Author: ChatGPT for Viswa (M.Tech Thesis Support)

# 📦 Required Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 📂 STEP 1.1: Load Your Dataset
csv_path = "Log Results.csv"
df = pd.read_csv(csv_path)

# ✅ STEP 1.2: Select relevant columns
df_selected = df[[
    "Load (N)", "Length (mm)", "Height (mm)", "Width (mm)", "Material",
    "Max Displacement (mm)", "Max Stress (MPa)"
]].copy()

# ✅ STEP 1.3: Drop any rows with missing values (important fix)
df_selected.dropna(inplace=True)

# ✅ STEP 1.4: Encode Material as numeric values
material_map = {
    "Aluminium": 0,
    "Steel": 1,
    "Carbon Fiber (approx.)": 2
}
df_selected["Material"] = df_selected["Material"].map(material_map)

# ✅ STEP 1.5: Split into features (X) and targets (Y)
X = df_selected[["Load (N)", "Length (mm)", "Height (mm)", "Width (mm)", "Material"]]
Y = df_selected[["Max Displacement (mm)", "Max Stress (MPa)"]]

# ✅ STEP 2.1: Normalize feature inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ STEP 2.2: Train/Test Split (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# 📝 OPTIONAL: Save processed data to CSV
pd.DataFrame(X_train, columns=X.columns).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv("X_test.csv", index=False)
Y_train.reset_index(drop=True).to_csv("Y_train.csv", index=False)
Y_test.reset_index(drop=True).to_csv("Y_test.csv", index=False)

# ✅ Print sample to verify
print("✅ Cleaned and Normalized X_train Sample:")
print(pd.DataFrame(X_train, columns=X.columns).head())

print("✅ Cleaned Y_train Sample:")
print(Y_train.head())
