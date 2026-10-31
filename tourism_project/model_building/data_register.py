"""Step 1 - Data registration.

Registers tourism.csv (kept in the repo under tourism_project/data/) as the single
source of truth for the pipeline. The script FAILS the workflow if the file is
missing, a required column is absent, or the target column is malformed.

Run from the repository root:
    python tourism_project/model_building/data_register.py
"""
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"  # raw dataset inside the data folder
TARGET = "ProdTaken"                           # 1 = purchased the package, 0 = did not

# Columns every downstream step relies on (see the data dictionary)
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# --- Validation: fail fast so a bad dataset never reaches the next job ---
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

if df.empty:
    raise ValueError("Dataset is empty.")

if not set(df[TARGET].dropna().unique()).issubset({0, 1}):
    raise ValueError(f"Target column '{TARGET}' must contain only 0/1 values.")

extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]

# --- Summary ---
print("Dataset registered successfully.")
print(f"Source file : {RAW_PATH}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
if extra:
    print(f"Extra (non-modelling) columns that data prep will drop: {extra}")
print(f"Missing values (total): {int(df.isna().sum().sum())}")
print(f"Duplicate rows (ignoring ID columns): "
      f"{int(df.drop(columns=['CustomerID'] + extra).duplicated().sum())}")
print(f"{TARGET} distribution:")
print(df[TARGET].value_counts())
print(f"Purchase rate: {df[TARGET].mean():.1%}")
