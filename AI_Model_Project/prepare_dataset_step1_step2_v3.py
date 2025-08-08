
# STEP 1 + STEP 2 (v3): AI Dataset Preparation with Clean Column Handling
# Author: ChatGPT for Viswa (M.Tech Thesis Support)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# STEP 1.1: Load dataset
csv_path = "Log Results.csv"
df = pd.read_csv(csv_path)

# STEP 1.2: Drop duplicate columns (e.g., 'Load (N).1') and any NaNs
df = df.drop(columns=["Load (N).1"], errors='ignore')
df = df.dropna().reset_index(drop=True)

# STEP 1.3: Encode 'Material' column
material_map = {
    "Aluminium": 0,
    "Steel": 1,
    "Carbon Fiber (approx.)": 2
}
df["Material"] = df["Material"].map(material_map)

# STEP 1.4: Define feature (X) and target (Y) columns
X_cols = ["Load (N)", "Length (mm)", "Height (mm)", "Width (mm)", "Material"]
Y_cols = ["Max Displacement (mm)", "Max Stress (MPa)"]

X = df[X_cols]
Y = df[Y_cols]

# STEP 2.1: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 2.2: Train/test split (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# STEP 2.3: Save all outputs (optional)
pd.DataFrame(X_train, columns=X_cols).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test, columns=X_cols).to_csv("X_test.csv", index=False)
Y_train.reset_index(drop=True).to_csv("Y_train.csv", index=False)
Y_test.reset_index(drop=True).to_csv("Y_test.csv", index=False)

# Print verification samples
print("✅ Fixed and Normalized X_train:")
print(pd.DataFrame(X_train, columns=X_cols).head())

print("\n✅ Corresponding Y_train:")
print(Y_train.head())
