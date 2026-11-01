"""Train, evaluate, track, and save the pre-contact tourism purchase model."""
import os
import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report, roc_auc_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("tourism-package-prediction-precontact")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")

# Fields expected to be known from customer/profile data before sales contact.
NUMERIC_FEATURES = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = ["Occupation", "Gender", "MaritalStatus", "Designation"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
missing = [c for c in FEATURES if c not in Xtrain.columns]
if missing:
    raise ValueError(f"Expected model features are missing from the training data: {missing}")
Xtrain = Xtrain[FEATURES]
Xtest = Xtest[FEATURES]

class_weight = float((ytrain == 0).sum() / max((ytrain == 1).sum(), 1))
preprocessor = make_column_transformer(
    (StandardScaler(), NUMERIC_FEATURES),
    (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
)
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    n_jobs=1,
    eval_metric="logloss",
)
param_grid = {
    "xgbclassifier__n_estimators": [150, 300],
    "xgbclassifier__max_depth": [6, 8, 10],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__colsample_bylevel": [0.6, 1.0],
    "xgbclassifier__learning_rate": [0.1, 0.2],
    "xgbclassifier__reg_lambda": [1, 5],
}
model_pipeline = make_pipeline(preprocessor, xgb_model)
classification_threshold = 0.50

with mlflow.start_run(run_name="xgboost-precontact-grid-search"):
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=2,
        refit=True,
        error_score="raise",
    )
    grid_search.fit(Xtrain, ytrain)
    results = grid_search.cv_results_
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
    try:
        for i, params in enumerate(results["params"]):
            with mlflow.start_run(run_name=f"trial_{i:03d}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_cv_f1", float(results["mean_test_score"][i]))
                mlflow.log_metric("std_cv_f1", float(results["std_test_score"][i]))
    finally:
        os.environ.pop("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", None)

    trials = pd.DataFrame(results).sort_values("rank_test_score")
    trials.to_csv("tuning_results.csv", index=False)
    mlflow.log_artifact("tuning_results.csv")
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("cv_scoring", "f1")
    mlflow.log_param("classification_threshold", classification_threshold)
    mlflow.log_param("feature_scope", "pre-contact customer profile")
    mlflow.log_metric("best_cv_f1", float(grid_search.best_score_))

    best_model = grid_search.best_estimator_
    y_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_train_pred = (y_train_proba >= classification_threshold).astype(int)
    y_test_pred = (y_test_proba >= classification_threshold).astype(int)
    train_report = classification_report(ytrain, y_train_pred, output_dict=True, zero_division=0)
    test_report = classification_report(ytest, y_test_pred, output_dict=True, zero_division=0)

    print("Best parameters:", grid_search.best_params_)
    print(f"Best 5-fold CV F1: {grid_search.best_score_:.4f}")
    print("Train report:")
    print(classification_report(ytrain, y_train_pred, zero_division=0))
    print("Test report:")
    print(classification_report(ytest, y_test_pred, zero_division=0))
    print(
        f"Held-out test: accuracy={test_report['accuracy']:.4f}, "
        f"precision={test_report['1']['precision']:.4f}, "
        f"recall={test_report['1']['recall']:.4f}, "
        f"F1={test_report['1']['f1-score']:.4f}, "
        f"ROC-AUC={roc_auc_score(ytest, y_test_proba):.4f}, "
        f"log-loss={log_loss(ytest, y_test_proba, labels=[0, 1]):.4f}"
    )
    mlflow.log_metrics({
        "train_accuracy": float(train_report["accuracy"]),
        "train_precision": float(train_report["1"]["precision"]),
        "train_recall": float(train_report["1"]["recall"]),
        "train_f1": float(train_report["1"]["f1-score"]),
        "train_roc_auc": float(roc_auc_score(ytrain, y_train_proba)),
        "train_log_loss": float(log_loss(ytrain, y_train_proba, labels=[0, 1])),
        "test_accuracy": float(test_report["accuracy"]),
        "test_precision": float(test_report["1"]["precision"]),
        "test_recall": float(test_report["1"]["recall"]),
        "test_f1": float(test_report["1"]["f1-score"]),
        "test_roc_auc": float(roc_auc_score(ytest, y_test_proba)),
        "test_log_loss": float(log_loss(ytest, y_test_proba, labels=[0, 1])),
    })
    model_path = "tourism_project/deployment/best_tourism_package_model_v1.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
