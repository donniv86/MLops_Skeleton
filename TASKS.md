# MLOps Project Task Tracker

> Mark tasks with `[x]` as you complete them. Work top-to-bottom — each stage feeds the next.

---

## STAGE 0 — Prerequisites & Setup

### 0.1 Git & Version Control
- [x] Initialised Git repository (`git init`)
- [x] Created `.gitignore` (excludes `.venv/`, `data/`, `*.pkl`, `.env`, `__pycache__/`)
- [x] Made first commit with all project files
- [x] Created a GitHub/GitLab remote repository
- [x] Pushed local repo to remote (`git push -u origin main`)
- [x] Practiced creating a feature branch (`git checkout -b feature/my-branch`)
- [ ] Opened and merged a Pull Request on GitHub

### 0.2 Python Environment
- [x] Created `.venv` using Python 3.11 (`python3.11 -m venv .venv`)
- [x] Created `requirements.txt` with all project dependencies
- [x] Installed core packages into `.venv`
- [x] Registered `.venv` as a Jupyter kernel (`Python (MLOps .venv)`)

### 0.3 Project Structure
- [x] Created `notebooks/` directory
- [x] Created `plan-mlopsEndToEndRoadmap.prompt.md`
- [x] Created `data/raw/` and `data/processed/` directories
- [x] Created `models/` directory for saved model artefacts
- [x] Created `src/` directory for reusable Python modules
- [x] Created `.env.example` file documenting required environment variables

---

## STAGE 1 — Data Ingestion

### Notebook: `01_data_ingestion.ipynb`
- [x] Created data ingestion notebook
- [x] Ran notebook top-to-bottom without errors
- [x] Loaded `titanic` dataset via `seaborn.load_dataset()`
- [x] Inspected shape, dtypes, and column meanings
- [x] Plotted target variable distribution (class balance check)
- [x] Plotted numeric feature distributions
- [x] Plotted categorical feature frequencies
- [x] Plotted correlation heatmap
- [x] Ran missing value analysis and visualised with `missingno`
- [x] Checked for duplicate rows
- [x] Ran IQR outlier detection on all numeric columns
- [x] Applied data cleaning (drop ID cols, impute nulls)
- [x] Passed all 6 automated validation checks
- [x] Saved raw snapshot to `data/raw/titanic_raw.csv`
- [x] Saved cleaned data to `data/processed/titanic_clean.csv`
- [x] Saved cleaned data to `data/processed/titanic_clean.parquet`
- [x] Verified Parquet file reloads correctly

---

## STAGE 2 — Feature Engineering

### Notebook: `02_feature_engineering.ipynb`
- [x] Created feature engineering notebook
- [x] Loaded cleaned Parquet from Stage 1
- [x] Applied one-hot encoding to `sex` and `embarked`
- [x] Applied median scaling to numeric features (`age`, `fare`)
- [x] Created new interaction features (e.g. `family_size = sibsp + parch`)
- [x] Built a scikit-learn `Pipeline` combining all transformations
- [x] Verified Pipeline prevents train-test leakage (fit only on train split)
- [x] Saved the fitted Pipeline to `models/feature_pipeline.pkl`
- [x] Saved feature-engineered train/test splits to `data/processed/`

---

## STAGE 3 — Model Training & Evaluation

### Notebook: `03_model_training.ipynb`
- [ ] Created model training notebook
- [ ] Trained a Logistic Regression baseline
- [ ] Trained a Random Forest classifier
- [ ] Evaluated both models: Accuracy, Precision, Recall, F1, AUC-ROC
- [ ] Plotted Confusion Matrix for each model
- [ ] Plotted ROC Curve comparing both models
- [ ] Identified which metric matters most (class imbalance → prioritise Recall/F1)
- [ ] Selected best model based on evaluation metrics
- [ ] Saved best model to `models/titanic_model_v1.pkl`

---

## STAGE 4 — Hyperparameter Tuning

### Notebook: `04_hyperparameter_tuning.ipynb`
- [ ] Created hyperparameter tuning notebook
- [ ] Ran `GridSearchCV` on Random Forest
- [ ] Ran `Optuna` study to optimise hyperparameters
- [ ] Compared tuned vs untuned model metrics
- [ ] Saved best tuned model to `models/titanic_model_v2.pkl`

---

## STAGE 5 — Experiment Tracking with MLflow

### Notebook: `05_experiment_tracking.ipynb`
- [ ] Installed MLflow (`pip install mlflow`)
- [ ] Started a local MLflow tracking server (`mlflow ui`)
- [ ] Logged parameters (model name, hyperparameters) to MLflow
- [ ] Logged metrics (Accuracy, F1, AUC) to MLflow
- [ ] Logged model artefact to MLflow Model Registry
- [ ] Compared runs side-by-side in MLflow UI
- [ ] Registered best model in MLflow with stage = `Staging`
- [ ] Promoted model to `Production` in MLflow registry

---

## STAGE 6 — Model Serving with FastAPI

### Task: `src/api/main.py`
- [ ] Created `src/api/main.py` with FastAPI app
- [ ] Added `POST /predict` endpoint accepting passenger features
- [ ] Loaded model from `models/` inside the API
- [ ] Added input validation using Pydantic schema
- [ ] Added `GET /health` endpoint for liveness checks
- [ ] Tested endpoint locally with `uvicorn`
- [ ] Tested with `curl` or Python `requests.post()`
- [ ] Added basic logging for each prediction request

---

## STAGE 7 — Docker & Containerisation

### Task: `Dockerfile`
- [ ] Installed Docker Desktop
- [ ] Created `Dockerfile` for the FastAPI prediction service
- [ ] Created `.dockerignore` to exclude `.venv/`, `data/`, `*.ipynb`
- [ ] Built Docker image (`docker build -t mlops-titanic .`)
- [ ] Ran container locally (`docker run -p 8000:8000 mlops-titanic`)
- [ ] Verified `/predict` endpoint works inside the container
- [ ] Created `docker-compose.yml` to run API + any supporting services

---

## STAGE 8 — CI/CD with GitHub Actions

### Task: `.github/workflows/ci.yml`
- [ ] Created `.github/workflows/` directory
- [ ] Created `ci.yml` pipeline file
- [ ] Added step: install dependencies
- [ ] Added step: run data validation checks
- [ ] Added step: run model evaluation and check minimum F1 threshold
- [ ] Added step: build Docker image
- [ ] Pipeline passes on every `git push`
- [ ] Added branch protection rule — PRs must pass CI before merge

---

## STAGE 9 — Drift Detection

### Notebook: `06_drift_detection.ipynb`
- [ ] Created drift detection notebook
- [ ] Installed `alibi-detect`
- [ ] Simulated production data with different distribution than training
- [ ] Ran KS test on continuous features
- [ ] Ran Chi-squared test on categorical features
- [ ] Confirmed drift detection triggers at correct threshold
- [ ] Compared with your existing `code_03_XX Drift Detection Example.ipynb`
- [ ] Added automated drift alert that prints a warning when drift is detected

---

## STAGE 10 — Fairness & Bias Monitoring

### Notebook: `07_fairness_monitoring.ipynb`
- [ ] Created fairness monitoring notebook
- [ ] Computed Equal Opportunity Score for `sex` column
- [ ] Compared score against 0.8–1.25 threshold
- [ ] Investigated which group is disadvantaged
- [ ] Retrained model on a rebalanced dataset
- [ ] Measured whether rebalancing improved fairness score
- [ ] Compared with your existing `code_06_03 Equal Opportunity Score with sklego.ipynb`

---

## STAGE 11 — Explainability with SHAP

### Notebook: `08_explainability.ipynb`
- [ ] Installed `shap`
- [ ] Computed global feature importance using SHAP values
- [ ] Plotted SHAP summary plot — which features drive predictions most?
- [ ] Computed local explanation for a single passenger prediction
- [ ] Plotted SHAP waterfall chart for that individual prediction
- [ ] Documented findings: does any feature raise a fairness concern?

---

## STAGE 12 — Infrastructure Monitoring (Stretch Goal)

### Task: Prometheus + Grafana
- [ ] Added Prometheus metrics to FastAPI (request count, latency, error rate)
- [ ] Created `docker-compose.yml` with Prometheus + Grafana + API
- [ ] Configured Prometheus scrape config to pull from the API
- [ ] Created Grafana dashboard showing API latency and prediction counts
- [ ] Set up an alert rule: notify if error rate exceeds 5%

---

## STAGE 13 — Cloud Deployment (Stretch Goal)

- [ ] Created a free-tier AWS or GCP account
- [ ] Pushed Docker image to a container registry (ECR or GCR)
- [ ] Deployed container to a managed service (AWS App Runner or Cloud Run)
- [ ] Verified prediction endpoint is accessible at a public URL
- [ ] Set up basic cost alerts to avoid unexpected charges

---

## Project Milestones

| Milestone | Tasks Required | Status |
|---|---|---|
| **Environment Ready** | Stage 0 complete | 🔄 In Progress |
| **Data Pipeline Complete** | Stages 1–2 complete | ✅ Milestone Achieved |
| **First Model Trained** | Stage 3 complete | ⬜ Not Started |
| **Experiment Tracking Live** | Stage 5 complete | ⬜ Not Started |
| **Model Serving Locally** | Stage 6 complete | ⬜ Not Started |
| **Containerised** | Stage 7 complete | ⬜ Not Started |
| **CI/CD Pipeline Active** | Stage 8 complete | ⬜ Not Started |
| **Monitoring Active** | Stages 9–11 complete | ⬜ Not Started |
| **Cloud Deployed** | Stage 13 complete | ⬜ Not Started |

---

## Progress Legend

| Symbol | Meaning |
|---|---|
| `- [x]` | Done |
| `- [ ]` | Not yet started |
| 🔄 | In Progress |
| ⬜ | Not Started |
| ✅ | Milestone Achieved |
