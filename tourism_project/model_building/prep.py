"""Clean the source data, export a cleaned copy, and create stratified splits."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"
CATEGORICAL_COLUMNS = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]
df = pd.read_csv(RAW_PATH)
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])
for col in CATEGORICAL_COLUMNS:
    df[col] = df[col].str.strip()
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate profiles removed after dropping IDs: {before - len(df)}")
print(f"Missing values remaining: {int(df.isna().sum().sum())}")

# Save a cleaned copy separately; never overwrite the source CSV.
cleaned_path = Path("tourism_project/artifacts/tourism_cleaned.csv")
cleaned_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(cleaned_path, index=False)
print(f"Cleaned dataset saved to {cleaned_path}")

X = df.drop(columns=[TARGET])
y = df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print(f"Train: {Xtrain.shape}; test: {Xtest.shape}")
print("Train target counts:")
print(ytrain.value_counts().sort_index())
print("Test target counts:")
print(ytest.value_counts().sort_index())
