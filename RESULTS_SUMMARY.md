# Tourism Package Prediction — verified final status

## Dataset

The supplied source CSV contains 4,128 rows × 21 columns, 797 positive labels (`ProdTaken = 1`, 19.3%), and no missing values. After dropping `Unnamed: 0` and `CustomerID`, 117 duplicate profiles are removed; the cleaned data contains 4,011 rows. The typo `Fe Male` is normalized to `Female`. The original CSV is preserved unchanged.

## Final model and evaluation

The final model is a pre-contact, profile-only XGBoost pipeline. It uses 13 features (9 numeric, 4 categorical) and excludes `TypeofContact`, `ProductPitched`, `DurationOfPitch`, `NumberOfFollowups`, and `PitchSatisfactionScore`, which are only known after contact or a sales pitch. The decision threshold is 0.50.

| Metric | Verified value |
|---|---:|
| Best 5-fold CV F1 | 0.7226 |
| Holdout accuracy | 0.9066 |
| Holdout positive-class precision | 0.8077 |
| Holdout positive-class recall | 0.6774 |
| Holdout positive-class F1 | 0.7368 |
| Holdout ROC-AUC | 0.9180 |
| Holdout log-loss | 0.2586 |
| Confusion matrix (TN, FP, FN, TP) | 623, 25, 50, 105 |

Training F1 is 1.00, a material overfitting warning. These results are a reproducible academic result, not evidence that the model is production-ready. Confirm the assignment’s exact interpretation of `ProdTaken` before making business claims.

## GitHub and workflow

The repository is public for assignment access: [Tourism_Package_Prediction](https://github.com/alexanderadhitya783-cloud/Tourism_Package_Prediction). The final GitHub Actions run [41268744961](https://github.com/alexanderadhitya783-cloud/Tourism_Package_Prediction/actions/runs/41268744961) passed all three jobs: dataset registration, data preparation, and model training. GitHub-run metrics match the local results above. The model artifact is present at `tourism_project/deployment/best_tourism_package_model_v1.joblib`.

The repository includes the course-provided customer-level CSV with `CustomerID` and profile/income/travel/pitch/outcome fields. A known-credential-pattern scan of repository history found no hits, and the CSV has no direct name, email, phone, or address columns. Follow the course dataset's sharing terms; do not add personal data or credentials.

## App and evidence status

The Streamlit app code and trained model are in the repository, but the app has **not** been deployed to Streamlit Community Cloud and no public app URL exists. Screenshots of the local app or local MLflow UI are local-development evidence only; do not present them as remote deployment evidence.
