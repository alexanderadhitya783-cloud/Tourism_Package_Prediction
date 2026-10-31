# Tourism Package Purchase Prediction (MLOps)



Predicts `ProdTaken` using XGBoost, scikit-learn preprocessing, MLflow experiment tracking, GitHub Actions CI/CD, and Streamlit.



## Project contents

- `tourism_project/data/tourism.csv` — dataset
- 
- `tourism_project/model_building/` — validation, preparation, and training scripts
- 
- `tourism_project/deployment/` — Streamlit app, requirements, and trained model
- 
- `.github/workflows/pipeline.yml` — automated CI/CD pipeline
- 


## Run

Install `tourism_project/requirements.txt`, run `data_register.py` and `prep.py` from the project root, start MLflow on port 5000, then run `train.py`. Launch the app with `streamlit run tourism_project/deployment/app.py`.



The supplied data contains 4,128 rows; preprocessing removes the unused row index and customer ID, normalizes gender spelling, and removes duplicate records. See the notebook for the full assignment walkthrough. Keep this repository private because it contains the dataset.



