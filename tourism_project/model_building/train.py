"""Step 3 - Model building with experiment tracking.

Loads the train/test splits produced by the previous workflow job, tunes an
XGBoost classifier with GridSearchCV, logs EVERY parameter combination (and its
cross-validated score) to MLflow, evaluates the best model on the train and test
sets, and saves it into tourism_project/deployment/ so the workflow can commit
it back to the repository, where Streamlit Community Cloud picks it up.

Run from the repository root (the workflow starts an MLflow server first):
    python tourism_project/model_building/train.py
"""
import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ----------------------------------------------------------------------------
# MLflow experiment tracking (the workflow starts a local MLflow server on :5000)
# ----------------------------------------------------------------------------
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("tourism-package-prediction")

# ----------------------------------------------------------------------------
# Load the splits downloaded from the previous job's artifact
# ----------------------------------------------------------------------------
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]

# Only ~19% of customers buy, so weight the positive class to handle the imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing: scale numeric features, one-hot-encode categorical features.
# It lives INSIDE the saved pipeline, so the Streamlit app can pass raw values.
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Base model (n_jobs=1: GridSearchCV already parallelises across the CV fits)
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, n_jobs=1)

# Hyperparameter grid (96 combinations x 5 folds)
param_grid = {
    "xgbclassifier__n_estimators": [150, 300],        # number of boosting trees
    "xgbclassifier__max_depth": [6, 8, 10],           # tree depth: higher = more complex, risk of overfitting
    "xgbclassifier__colsample_bytree": [0.6, 0.8],    # fraction of features sampled per tree
    "xgbclassifier__colsample_bylevel": [0.6, 1.0],   # fraction of features sampled per tree level
    "xgbclassifier__learning_rate": [0.1, 0.2],       # boosting step size
    "xgbclassifier__reg_lambda": [1, 5],              # L2 regularisation strength
}

# Model pipeline: preprocessing + XGBoost
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Decision threshold on the predicted purchase probability. F1 is flat between
# 0.30 and 0.50 on out-of-fold training predictions; 0.45 gives a good
# precision/recall balance. The Streamlit app uses the same value.
classification_threshold = 0.45

with mlflow.start_run(run_name="xgboost-grid-search"):
    # Hyperparameter tuning. F1 (purchase class) is used instead of accuracy
    # because accuracy is misleading on imbalanced data.
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination tried, as a nested run, so all trials can
    # be compared side by side in the MLflow UI
    results = grid_search.cv_results_
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"  # one link per trial would flood the log
    for i in range(len(results["params"])):
        with mlflow.start_run(run_name=f"trial_{i:03d}", nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])
    os.environ.pop("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT")
    print(f"Logged {len(results['params'])} nested MLflow runs (one per parameter combination)")

    # Keep a CSV of all trials (the workflow uploads it as an artifact)
    trials = pd.DataFrame(results).sort_values("rank_test_score")
    trials.to_csv("tuning_results.csv", index=False)
    mlflow.log_artifact("tuning_results.csv")

    # Log the best hyperparameters in the parent run
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("cv_scoring", "f1")
    mlflow.log_param("classification_threshold", classification_threshold)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    print(f"Combinations tried: {len(results['params'])}")
    print(f"Best parameters   : {grid_search.best_params_}")
    print(f"Best CV F1        : {grid_search.best_score_:.4f}")

    # Store the best model
    best_model = grid_search.best_estimator_

    # Predictions on train and test data using the custom threshold
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    print("\nTrain classification report:")
    print(classification_report(ytrain, y_pred_train))
    print("Test classification report:")
    print(classification_report(ytest, y_pred_test))

    # Log evaluation metrics (precision/recall/F1 are for the purchase class "1")
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "train_roc_auc": roc_auc_score(ytrain, y_pred_train_proba),
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"],
        "test_roc_auc": roc_auc_score(ytest, y_pred_test_proba),
    })

    # Save the model next to app.py so the Streamlit app can load it directly,
    # and log it as an MLflow artifact for traceability
    model_path = "tourism_project/deployment/best_tourism_package_model_v1.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
