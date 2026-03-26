# MLOps & DevOps Foundations — Hands-On Learning Project

> **Audience:** Undergraduate / graduate students exploring MLOps, DevOps, and production ML for the first time.  
> **Goal:** Take you from raw data all the way to hosting, monitoring, and governing a real ML model — step by step, hands-on.

---

## Current Progress

| Stage | Status | Notes |
|---|---|---|
| **Stage 0** — Git, Env, Structure | ✅ Complete | venv (Python 3.11), kernel registered, all folders created |
| **Stage 1** — Data Ingestion | ✅ Complete | 891 rows, seaborn titanic, EDA + validation + Parquet export |
| **Stage 2** — Feature Engineering | ✅ Complete | sklearn Pipeline, OHE + scaling, interaction features, leakage-free splits |
| **Stage 3** — Model Training | ⬜ Not Started | |
| **Stage 4** — Hyperparameter Tuning | ⬜ Not Started | |
| **Stage 5** — Experiment Tracking | ⬜ Not Started | |
| **Stage 6** — Model Serving | ⬜ Not Started | |
| **Stage 7** — Docker | ⬜ Not Started | |
| **Stage 8** — CI/CD | ⬜ Not Started | |
| **Stage 9** — Drift Detection | ⬜ Not Started | |
| **Stage 10** — Fairness Monitoring | ⬜ Not Started | |
| **Stage 11** — Explainability | ⬜ Not Started | |

---

## What Is This Project?

Most university courses teach you how to *train* a model. Industry requires you to know how to *run* one — reliably, at scale, fairly, and safely — for months or years after it first launches.

This project covers that gap. You will build a complete end-to-end ML pipeline using real datasets and industry-standard tools, following the same practices used at companies like Google, Uber, Netflix, and Airbnb.

```
[Raw Data] → [Clean & Validate] → [Feature Engineering] → [Train & Track]
                                                                  ↓
[Monitor & Retrain] ← [Serve Predictions] ← [Package & Deploy]
```

---

## Project Structure

```
MLops/
│
├── notebooks/                          # Hands-on Jupyter notebooks (work through in order)
│   ├── 01_data_ingestion.ipynb         # Stage 1 — Load & validate data via seaborn
│   ├── 02_feature_engineering.ipynb    # Stage 2 — Encode, scale, and transform features
│   ├── 03_model_training.ipynb         # Stage 3 — Train, evaluate, compare models
│   ├── 04_hyperparameter_tuning.ipynb  # Stage 4 — Optimise with Optuna
│   ├── 05_experiment_tracking.ipynb    # Stage 5 — Track experiments with MLflow
│   ├── 06_drift_detection.ipynb        # Stage 9 — Detect data & concept drift
│   ├── 07_fairness_monitoring.ipynb    # Stage 10 — Measure bias & fairness
│   └── 08_explainability.ipynb         # Stage 11 — Explain predictions with SHAP
│
├── Ex_Files_MLOps_Essentials_Model_Drift_Bias/
│   └── Exercise Files/                 # Original tutorial exercises (credit approval dataset)
│       ├── code_03_XX Drift Detection Example.ipynb
│       ├── code_06_03 Equal Opportunity Score with sklego.ipynb
│       ├── credit-approval-training-data.csv
│       ├── credit-approval-prod-data.csv
│       └── credit-approval-fair-data.csv
│
├── src/                                # (Created in Stage 6) Reusable Python modules
│   └── api/
│       └── main.py                     # FastAPI prediction service
│
├── data/                               # Auto-created by notebooks — NOT tracked in Git
│   ├── raw/                            # Original downloaded snapshots
│   └── processed/                      # Cleaned & feature-engineered data
│
├── models/                             # Saved model artefacts — NOT tracked in Git
│
├── .github/
│   └── workflows/
│       └── ci.yml                      # (Created in Stage 8) GitHub Actions CI/CD pipeline
│
├── .gitignore                          # Excludes venv, data, models, secrets
├── requirements.txt                    # All Python dependencies
├── TASKS.md                            # Checklist — tick off as you progress
└── plan-mlopsEndToEndRoadmap.prompt.md # Full conceptual roadmap reference
```

---

## Datasets Used

### 1. Titanic Survival Dataset (`seaborn.load_dataset('titanic')`)
Used in Notebooks 1–5, 8.

| Column | Description | Type |
|---|---|---|
| `survived` | **Target** — 1 = survived, 0 = did not | Binary |
| `pclass` | Ticket class (1 = First, 2 = Second, 3 = Third) | Categorical |
| `sex` | Gender — a **protected attribute** (like race in credit data) | Categorical |
| `age` | Age in years — has ~20% missing values | Continuous |
| `sibsp` | Siblings/spouses aboard | Numeric |
| `parch` | Parents/children aboard | Numeric |
| `fare` | Ticket price — has outliers | Continuous |
| `embarked` | Port: C = Cherbourg, Q = Queenstown, S = Southampton | Categorical |

**Why Titanic?** It has realistic data quality issues (missing values, class imbalance, potential gender bias, outliers) — all the same problems you will face in a real job.

> **Note:** The dataset is loaded via `seaborn.load_dataset('titanic')` — a stable, script-free CSV source. The `mstz/titanic` HuggingFace dataset uses a legacy loading script incompatible with `datasets ≥ 2.16`.

---

### 2. Credit Approval Dataset (Exercise Files — pre-loaded CSVs)
Used in Notebooks 6–7 (drift & fairness).

| Column | Description |
|---|---|
| `APPLICANT_ID` | Unique ID — drop before training |
| `AGE_RANGE` | Age bracket (1–3) |
| `INCOME_CATEGORY` | Income level (1–4) |
| `RACE` | Encoded race category (1–5) — **protected attribute** |
| `CREDIT_RATING` | Credit score bucket (1–6) |
| `APPROVED` | **Target** — 1 = approved, 0 = denied |

Three versions exist to simulate real-world scenarios:

| File | Scenario |
|---|---|
| `credit-approval-training-data.csv` | Historical data — model trained on this |
| `credit-approval-prod-data.csv` | Production data — distribution has **drifted** |
| `credit-approval-fair-data.csv` | Rebalanced data — used to reduce bias |

---

## Setup Instructions

### Prerequisites
- macOS / Linux / WSL2 on Windows
- Python 3.11 ([download here](https://www.python.org/downloads/))
- Git ([download here](https://git-scm.com/))
- VS Code with the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/donniv86/MLops_Skeleton.git
cd MLops_Skeleton
```

Or if you downloaded the ZIP, extract it and open the folder in VS Code.

---

### Step 2 — Create and Activate a Virtual Environment

> ⚠️ **Important:** Always use Python 3.11. Python 3.13+ is not yet fully supported by scipy and scikit-learn.

```bash
# Create the virtual environment
python3.11 -m venv .venv

# Activate it
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
```

You will see `(.venv)` appear in your terminal prompt. This means you are inside the isolated environment.

---

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all packages needed for every stage of the project.

---

### Step 4 — Register the Kernel with Jupyter

So notebooks can use your `.venv` instead of the system Python:

```bash
python -m ipykernel install --user --name mlops-venv --display-name "Python (MLOps .venv)"
```

---

### Step 5 — Open a Notebook

In VS Code, open any notebook in `notebooks/`. When prompted to select a kernel, choose **Python (MLOps .venv)**.

Or run Jupyter in the browser:
```bash
jupyter notebook
```

---

## Learning Path — Work Through in Order

### Month 1–2 | Foundations

| # | Notebook / Task | Key Concepts Learned |
|---|---|---|
| 0 | Git setup, `.gitignore`, `requirements.txt` | Version control, reproducible environments |
| 1 | `01_data_ingestion.ipynb` | seaborn dataset, EDA, missing values, data validation, Parquet |

---

### Month 3–4 | Data & Modelling

| # | Notebook / Task | Key Concepts Learned |
|---|---|---|
| 2 | `02_feature_engineering.ipynb` | Encoding, scaling, sklearn Pipelines, train-test leakage |
| 3 | `03_model_training.ipynb` | Algorithm comparison, F1/AUC-ROC, confusion matrix |
| 4 | `04_hyperparameter_tuning.ipynb` | GridSearchCV, Optuna, Bayesian optimisation |
| 5 | `05_experiment_tracking.ipynb` | MLflow tracking, model registry, staging vs production |

---

### Month 5–6 | Productionisation

| # | Notebook / Task | Key Concepts Learned |
|---|---|---|
| 6 | `src/api/main.py` | FastAPI, Pydantic validation, REST API, health checks |
| 7 | `Dockerfile` | Docker containers, reproducible deployments |
| 8 | `.github/workflows/ci.yml` | GitHub Actions, automated testing, CI/CD |

---

### Month 7–8 | Monitoring & Governance

| # | Notebook / Task | Key Concepts Learned |
|---|---|---|
| 9 | `06_drift_detection.ipynb` | Data drift, KS test, alibi-detect library |
| 10 | `07_fairness_monitoring.ipynb` | Equal Opportunity Score, demographic parity, bias correction |
| 11 | `08_explainability.ipynb` | SHAP values, global vs local explanations |

---

### Month 9–12 | Cloud & Scale (Stretch Goals)

| # | Task | Key Concepts Learned |
|---|---|---|
| 12 | Add Prometheus + Grafana | Infrastructure monitoring, alerting |
| 13 | Deploy to AWS/GCP free tier | Cloud hosting, container registries |
| 14 | Learn Terraform | Infrastructure as Code |
| 15 | Learn Apache Airflow | Pipeline orchestration, DAGs |

---

## Core Concepts Explained

### What is MLOps?

**DevOps** = practices that make software delivery fast and reliable (CI/CD, monitoring, version control).  
**MLOps** = DevOps, extended for Machine Learning systems.

The extra complexity in ML comes from the fact that there are **three things that can change** — not just code:

```
Code changes  →  model may behave differently
Data changes  →  model may go stale (drift)
World changes →  labels may shift (concept drift)
```

A model that performs perfectly in development can silently degrade in production. MLOps gives you the tools to detect, diagnose, and fix this.

---

### What is Model Drift?

When the real-world data your model sees in production is statistically different from the data it was trained on.

**This project demonstrates it concretely:**  
The credit approval model trained on `credit-approval-training-data.csv` achieves **92% accuracy**.  
When evaluated on `credit-approval-prod-data.csv` (simulating real production traffic 6 months later), accuracy drops to **37.5%**.

No code changed. The model didn't "break". The world changed.

---

### What is Fairness / Bias in ML?

An ML model is biased when it treats different demographic groups unequally — often inheriting historical discrimination present in the training data.

**Equal Opportunity Score** = ratio of True Positive Rate across groups:

$$\text{EOS} = \frac{\text{TPR(privileged group)}}{\text{TPR(unprivileged group)}}$$

- Score = **1.0** → perfectly fair
- Score < **0.8** → unacceptably biased against the unprivileged group
- Score > **1.25** → unacceptably biased in favour of the privileged group

In the credit approval dataset, `RACE` has an EOS below 0.8 — meaning applicants from certain racial groups are approved less often even when they are equally qualified. This is illegal under the Equal Credit Opportunity Act (ECOA).

---

### Why Not Just Use Accuracy?

| Scenario | Accuracy | Reality |
|---|---|---|
| 95% of loans are denied | A model that always says "deny" gets **95% accuracy** | Completely useless — approves nobody |
| Model is racially biased | High accuracy | Systemically unfair |
| Concept drift has occurred | Still high accuracy initially | Silently degrading |

**Always use F1, AUC-ROC, and fairness metrics alongside accuracy.**

---

### Train-Test Leakage — The Silent Killer

A common beginner mistake: fitting your scaler or encoder on the full dataset, then splitting into train/test.

```python
# ❌ WRONG — scaler has seen test data
scaler.fit(df[features])  # fitted on ALL data
X_train, X_test = train_test_split(df[features])

# ✅ CORRECT — scaler only sees training data
X_train, X_test = train_test_split(df[features])
scaler.fit(X_train)       # fitted ONLY on training data
scaler.transform(X_test)  # applied to test data
```

Using a scikit-learn `Pipeline` prevents this automatically.

---

## Key Tools Reference

| Tool | What it does | When you use it |
|---|---|---|
| `pandas` | Load, clean, transform tabular data | Always |
| `scikit-learn` | Build ML models, pipelines, metrics | Stages 2–4 |
| `datasets` | Load any dataset from HuggingFace Hub | Stage 1 |
| `pyarrow` | Read/write Parquet files | Stage 1 |
| `missingno` | Visualise missing data patterns | Stage 1 |
| `mlflow` | Track experiments, register models | Stage 5 |
| `alibi-detect` | Detect data and concept drift | Stage 9 |
| `scikit-lego` | Fairness metrics (equal opportunity score) | Stage 10 |
| `shap` | Explain model predictions | Stage 11 |
| `fastapi` | Serve predictions via REST API | Stage 6 |
| `docker` | Package app + dependencies into a container | Stage 7 |
| `optuna` | Bayesian hyperparameter optimisation | Stage 4 |
| `great-expectations` | Validate data quality automatically | Stage 1 |

---

## Common Mistakes to Avoid

| Mistake | Why it's a problem | Fix |
|---|---|---|
| Using Python 3.13+ for ML | scipy/sklearn don't have compiled wheels yet | Use Python 3.11 |
| Committing `.venv/` to Git | 100s of MB, machine-specific | Add to `.gitignore` |
| Committing `.env` or API keys | Security breach | Use `.env.example` template + secrets manager |
| Fitting preprocessors on test data | Leakage — inflated test metrics | Use sklearn `Pipeline` |
| Using accuracy on imbalanced data | Misleading — 95% accuracy ≠ good model | Use F1, AUC-ROC, recall |
| Not saving raw data before cleaning | Can never audit what went wrong | Always save `data/raw/` snapshot |
| Ignoring fairness metrics | Legal risk, ethical harm | Always compute EOS / demographic parity |
| Not versioning data | Can't reproduce a model months later | Use DVC or Delta Lake |
| Deploying without a health check | No way to know if service is alive | Add `/health` endpoint |
| Retraining without drift detection | May retrain on corrupt or shifted data | Run drift check first |

---

## Progress Tracker

See [TASKS.md](TASKS.md) for the full checklist of tasks to complete as you work through the project. Tick items off as you go.

**Current milestone status:**

| Milestone | Status |
|---|---|
| Environment Ready | 🔄 In Progress |
| Data Pipeline Complete | ⬜ Not Started |
| First Model Trained | ⬜ Not Started |
| Experiment Tracking Live | ⬜ Not Started |
| Model Serving Locally | ⬜ Not Started |
| Containerised | ⬜ Not Started |
| CI/CD Pipeline Active | ⬜ Not Started |
| Monitoring Active | ⬜ Not Started |
| Cloud Deployed | ⬜ Not Started |

---

## Further Reading & Resources

### Foundational Papers
- [Hidden Technical Debt in Machine Learning Systems (Google, 2015)](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
- [The ML Test Score: A Rubric for ML Production Readiness (Google, 2017)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf)

### Learning Resources
- [roadmap.sh/mlops](https://roadmap.sh/mlops) — visual MLOps learning roadmap
- [ml-ops.org](https://ml-ops.org/content/mlops-principles) — MLOps principles in depth
- [Made With ML](https://madewithml.com/) — free MLOps course by a former Apple/Google engineer
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/) — covers the full production stack

### Tools Documentation
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [alibi-detect Docs](https://docs.seldon.io/projects/alibi-detect/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [HuggingFace Datasets Docs](https://huggingface.co/docs/datasets)
- [SHAP Docs](https://shap.readthedocs.io/)

---

## Contributing

This is a personal learning project. If you are a classmate:
1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-name-notebook-X`
3. Make your changes and commit: `git commit -m "feat: completed stage X notebook"`
4. Open a Pull Request — describe what you learned and what decisions you made

---

## License

Educational use only. Dataset credits:
- Titanic dataset: [Kaggle / mstz](https://huggingface.co/datasets/mstz/titanic) — CC licence
- Credit approval data: LinkedIn Learning — MLOps Essentials course exercise files

---

*Built as a hands-on companion to the [MLOps Essentials: Model Drift & Bias](https://www.linkedin.com/learning/) course, extended with a complete end-to-end production pipeline.*
