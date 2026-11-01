# Tourism Package Purchase Prediction — MLOps

This course project trains an XGBoost model to estimate `ProdTaken` from customer profile information available before a sales pitch. It includes dataset validation, cleaning, stratified train/test splitting, MLflow experiment tracking, a GitHub Actions workflow, and a Streamlit application.

## Final model scope

The final app uses 13 pre-contact profile features: 9 numeric and 4 categorical. It excludes `TypeofContact`, `ProductPitched`, `DurationOfPitch`, `NumberOfFollowups`, and `PitchSatisfactionScore`, because those describe contact or pitch outcomes rather than information available before outreach. The decision threshold is 0.50.

The validated local holdout results are accuracy 0.9066, positive-class precision 0.8077, recall 0.6774, F1 0.7368, ROC-AUC 0.9180, and log-loss 0.2586. Best 5-fold CV F1 is 0.7226. Training F1 is 1.00, so overfitting is a material limitation. These are academic results, not evidence of production readiness.

## Repository contents

- `AML_MLOps_Tourism_Package_Prediction_FINAL.ipynb` — final notebook with the profile-only experiment and current evidence notes.
- `tourism_project/data/tourism.csv` — the course-provided dataset used by the pipeline.
- `tourism_project/model_building/` — validation, preparation, and training scripts.
- `tourism_project/deployment/` — Streamlit app, dependencies, and model artifact produced by Actions.
- `.github/workflows/pipeline.yml` — three-job workflow triggered by pushes to `main` or manual dispatch.

## Run locally

Use Python 3.11 and install `tourism_project/requirements.txt`. From the repository root, run:

```bash
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
streamlit run tourism_project/deployment/app.py
```

For local MLflow UI tracking, start MLflow separately and set `MLFLOW_TRACKING_URI` before running the training script. The workflow starts its own MLflow server, uploads tracking results as a workflow artifact, and commits the newly trained model to `main` after successful training.

## Workflow and deployment status

The final profile-only pipeline completed successfully in GitHub Actions: [run 41268744961](https://github.com/alexanderadhitya783-cloud/Tourism_Package_Prediction/actions/runs/41268744961). All three jobs passed, including training and committing the refreshed model artifact. The GitHub-run metrics match the verified local results: 5-fold CV F1 0.7226; holdout accuracy 0.9066, precision 0.8077, recall 0.6774, F1 0.7368, ROC-AUC 0.9180, and log-loss 0.2586.

The Streamlit application is repository code only; it is **not** deployed to Streamlit Community Cloud and there is no live app URL yet.

## Data and safety notice

This repository includes the course-provided customer-level dataset, including a `CustomerID` column and profile, income, travel, pitch, and outcome fields. Its public visibility was enabled at the repository owner's request for assignment access. Do not add real customer data, personal identifiers, access tokens, or other secrets. Confirm the course dataset's sharing terms before reusing it elsewhere. The app is for academic demonstration only; review fairness and privacy before any operational use.
