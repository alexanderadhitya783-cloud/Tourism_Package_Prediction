"""Step 2 - Data preparation.

Loads the registered dataset straight from the repo's data folder, cleans it,
removes columns that carry no predictive signal, splits it into train/test sets
and writes four CSVs. The workflow uploads those CSVs as an artifact so the
next job (model training) can download them.

Run from the repository root:
    python tourism_project/model_building/prep.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"  # 1 if the customer purchased the package, else 0

# Text columns that are one-hot-encoded later, inside the model pipeline
CATEGORICAL_COLUMNS = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]

# ----------------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {RAW_PATH}")

# ----------------------------------------------------------------------------
# 2. Clean
# ----------------------------------------------------------------------------
# 2a. Remove columns that are not predictive: the row index ("Unnamed: 0") and
#     the customer identifier ("CustomerID") would only let the model memorise rows.
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

# 2b. Standardise text values: trim stray spaces and fix the known typo
#     "Fe Male" -> "Female" (otherwise it becomes a third, bogus gender category).
for col in CATEGORICAL_COLUMNS:
    df[col] = df[col].str.strip()
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# 2c. Remove exact duplicate rows (none expected, kept as a safeguard)
n_before = len(df)
df = df.drop_duplicates()
print(f"Duplicate rows removed: {n_before - len(df)}")

# 2d. Missing values: this dataset has none; XGBoost also handles NaN natively
print(f"Missing values remaining: {int(df.isna().sum().sum())}")

# NOTE: categorical columns are intentionally left as raw strings. The training
# pipeline one-hot-encodes them and the Streamlit app sends raw category values,
# so training and serving always use the same representation.

# ----------------------------------------------------------------------------
# 3. Split (stratified, so the ~19% purchase rate is preserved in both sets)
# ----------------------------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------------------------
# 4. Save locally (the workflow uploads these four files as an artifact)
# ----------------------------------------------------------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print(f"Train: {Xtrain.shape}  |  Test: {Xtest.shape}")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
print("ProdTaken distribution in test:")
print(ytest.value_counts())
