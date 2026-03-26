# Complete MLOps & DevOps Roadmap — Raw Data to Production

## The Master Pipeline (Everything connected)

```
[RAW DATA] → [DATA ENGINEERING] → [FEATURE ENGINEERING] → [MODEL TRAINING]
     ↓               ↓                     ↓                      ↓
[Storage &      [Validation &          [Feature              [Experiment
 Versioning]     Cleaning]              Store]                Tracking]
                                                                  ↓
[MONITORING] ← [PRODUCTION SERVING] ← [DEPLOYMENT] ← [MODEL PACKAGING
     ↓               ↓                                  & REGISTRY]
[Drift Alert]   [Bias / Fairness]
     ↓
[RETRAINING LOOP] ─────────────────────────────────────────────────────┐
                                                                        ↑
                                              Automatic feedback back to training
```

---

## STAGE 0 — Programming & DevOps Prerequisites

### 0.1 Version Control with Git
| Concept | What to learn |
|---|---|
| Basic Git | `git init`, `commit`, `push`, `pull`, `branch`, `merge` |
| Branching strategy | Feature branches, main/develop, pull requests |
| GitHub / GitLab | Remote repos, issues, code review via PRs |
| `.gitignore` | Exclude large files, secrets, notebooks outputs |

Industry rule: **You should never directly commit to `main`**. Every change goes through a Pull Request (PR) reviewed by a teammate.

### 0.2 Python Packaging & Environments
| Concept | What to learn |
|---|---|
| Virtual environments | `conda`, `venv` — isolate project dependencies |
| `requirements.txt` / `pyproject.toml` | Declare and pin dependencies |
| Python project structure | `src/`, `tests/`, `notebooks/`, `configs/` |

### 0.3 Linux / Bash
- File navigation, permissions (`chmod`, `chown`)
- Shell scripts to automate workflows
- SSH into remote servers
- Process management (`ps`, `kill`, `nohup`)
- Environment variables (`export`, `.env` files)

### 0.4 Networking Basics
- TCP/IP, DNS, ports
- HTTP/HTTPS, REST APIs (GET/POST/JSON)
- Understanding of firewalls and load balancers at a conceptual level

---

## STAGE 1 — Raw Data Layer

### 1.1 Data Sources
| Source | Examples | Tool |
|---|---|---|
| Relational databases | MySQL, PostgreSQL | SQLAlchemy, psycopg2 |
| Data warehouses | Snowflake, BigQuery, Redshift | dbt, SQL |
| File storage | AWS S3, Azure Blob, GCS | boto3, fsspec |
| Streaming | Apache Kafka, AWS Kinesis | kafka-python |
| APIs | REST endpoints, IoT sensors | requests |
| Data lakes | Delta Lake, Apache Iceberg | PySpark |

### 1.2 Data Versioning
| Tool | What it does |
|---|---|
| **DVC** (Data Version Control) | Git extension for versioning large datasets and models |
| **Delta Lake** | Versioned data lake (keeps history of every table change) |
| **LakeFS** | Git-like version control for entire data lakes |

### 1.3 Data Catalogues & Lineage
| Tool | Purpose |
|---|---|
| Apache Atlas | Track data lineage across pipelines |
| OpenMetadata | Open-source data catalogue |
| AWS Glue Data Catalog | Metadata store for AWS data |

---

## STAGE 2 — Data Engineering

### 2.1 Exploratory Data Analysis (EDA)
| Check | Library | What to look for |
|---|---|---|
| Shape, types, nulls | `pandas` | Missing value %, dtypes |
| Distributions | `matplotlib`, `seaborn` | Skew, outliers, class imbalance |
| Correlations | `pandas.corr()` | Multicollinearity between features |
| Target imbalance | `value_counts()` | e.g. 95% denied, 5% approved |

### 2.2 Data Cleaning & Preprocessing
| Problem | Fix |
|---|---|
| Missing values | Imputation (mean/median/mode) or drop |
| Duplicate rows | `df.drop_duplicates()` |
| Outliers | IQR clipping, Z-score filtering |
| Wrong data types | `df.astype()`, date parsing |
| Inconsistent categories | Standardise labels |

### 2.3 Data Validation
| Tool | What it checks |
|---|---|
| **Great Expectations** | Schema, value ranges, no nulls, distribution checks |
| **Pandera** | Type and statistical validation for DataFrames |
| **TFDV** | Distribution statistics, schema inference |

### 2.4 Data Pipelines & Orchestration
| Tool | Level | What it does |
|---|---|---|
| **Apache Airflow** | Industry standard | DAG-based task orchestration |
| **Prefect** | Modern alternative | Python-native workflows |
| **Apache Spark** | Big Data | Distributed processing of TB-scale data |
| **dbt** | SQL-focused | Data transformation in warehouses (ELT) |

---

## STAGE 3 — Feature Engineering & Feature Stores

### 3.1 Feature Engineering Techniques
| Technique | Example (credit data) |
|---|---|
| Binarisation | `AGE_RANGE == 3` → 1 |
| One-hot encoding | `RACE` categories → 5 binary columns |
| Scaling | Normalise `CREDIT_RATING` to 0–1 range |
| Interaction features | `INCOME × CREDIT_RATING` |
| Time-based features | "Days since last credit application" |

⚠️ **Critical pitfall**: Feature transformations must be fitted **only on training data** to avoid train-test leakage. Always use a sklearn `Pipeline`.

### 3.2 Feature Stores
| Tool | Type |
|---|---|
| **Feast** | Open-source feature store |
| **Tecton** | Enterprise feature store |
| AWS SageMaker Feature Store | Managed cloud option |

---

## STAGE 4 — Model Development

### 4.1 Algorithm Selection
| Algorithm | When to use |
|---|---|
| Logistic Regression | Binary classification baseline |
| Gaussian Naive Bayes | Fast, works on small data |
| Decision Tree / Random Forest | Interpretable, handles non-linearity |
| XGBoost / LightGBM | Industry workhorse for tabular data |
| Neural Networks (PyTorch) | Images, text, sequences |

### 4.2 Model Evaluation Metrics
| Metric | When it matters |
|---|---|
| **Accuracy** | Only when classes are balanced |
| **Precision** | When false positives are costly |
| **Recall** | When false negatives are costly |
| **F1 Score** | Balance between precision and recall |
| **AUC-ROC** | Overall model discrimination ability |
| **Confusion Matrix** | Full breakdown of TP/FP/TN/FN |

### 4.3 Hyperparameter Tuning
| Tool | Method |
|---|---|
| sklearn `GridSearchCV` | Try all combinations |
| sklearn `RandomizedSearchCV` | Random sample, faster |
| **Optuna** | Bayesian optimisation, industry standard |
| **Ray Tune** | Distributed hyperparameter search |

### 4.4 Experiment Tracking
| Tool | What it tracks |
|---|---|
| **MLflow** | Parameters, metrics, artefacts, model versions |
| **Weights & Biases (W&B)** | Same + richer visualisations |
| **DVC** | Code + data + metrics combined in Git |

---

## STAGE 5 — Model Packaging & Registry

### 5.1 Serialising a Model
```python
import joblib
joblib.dump(credit_classifier, "credit_model_v1.pkl")   # save
model = joblib.load("credit_model_v1.pkl")              # reload
```

### 5.2 Model Registry States
| State | Meaning |
|---|---|
| `Staging` | Candidate passed automated tests, awaiting human approval |
| `Production` | Currently serving live traffic |
| `Archived` | Old version, kept for rollback |

| Tool | Type |
|---|---|
| **MLflow Model Registry** | Open-source |
| AWS SageMaker Model Registry | Managed cloud |
| Azure ML Model Registry | Managed cloud |

---

## STAGE 6 — DevOps & CI/CD

### 6.1 Docker (Containerisation)
Packages code + all dependencies into a portable container image.

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 Kubernetes (Orchestration)
| Concept | What it means |
|---|---|
| Pod | Smallest deployable unit |
| Deployment | Manages replicas of a pod |
| Service | Exposes pods to network traffic |
| Ingress | Routes external HTTP traffic to services |
| HPA | Horizontal Pod Autoscaler — add more pods under load |

### 6.3 CI/CD Pipeline
```
Push code → Run unit tests → Run data validation → Train model →
Evaluate metrics → If metrics pass → Build Docker image →
Push to container registry → Deploy to Kubernetes
```

| Tool | Use |
|---|---|
| **GitHub Actions** | Most beginner-friendly |
| **GitLab CI** | For GitLab users |
| **Jenkins** | Common in enterprises |

### 6.4 Infrastructure as Code (IaC)
| Tool | What it provisions |
|---|---|
| **Terraform** | Any cloud resource |
| **Ansible** | Configure existing machines |
| **Helm** | Package and deploy Kubernetes applications |

---

## STAGE 7 — Model Serving / Hosting

### 7.1 FastAPI Prediction Endpoint Example
```python
from fastapi import FastAPI
import joblib, numpy as np

app = FastAPI()
model = joblib.load("credit_model_v1.pkl")

@app.post("/predict")
def predict(age_range: int, income: int, race: int, credit: int):
    features = np.array([[age_range, income, race, credit]])
    prediction = model.predict(features)[0]
    return {"approved": int(prediction)}
```

### 7.2 Serving Patterns
| Pattern | Description | When to use |
|---|---|---|
| **Real-time / Online** | One record at a time via REST API | Credit scoring, fraud detection |
| **Batch inference** | Thousands of records overnight | Monthly reports |
| **Streaming** | Predict on Kafka/Kinesis data | IoT, real-time fraud |
| **Edge inference** | Model runs on-device | Offline, low latency, privacy |

### 7.3 Model Serving Frameworks
| Tool | Best for |
|---|---|
| **FastAPI** | Custom Python APIs |
| **BentoML** | Package + serve any ML model easily |
| **TorchServe** | Serving PyTorch models |
| **Triton Inference Server** | High-performance GPU inference |
| **Seldon Core** | K8s-native model serving |

### 7.4 Deployment Strategies
| Strategy | How it works |
|---|---|
| **Blue/Green** | Run old and new in parallel. Switch all traffic instantly. |
| **Canary** | Route 5% traffic to new model, gradually increase. |
| **Shadow** | New model logs predictions but doesn't serve them. |
| **A/B Testing** | Split users. Compare business metrics. |

---

## STAGE 8 — Monitoring & Observability

### 8.1 The Four Layers of Monitoring
```
┌────────────────────────────────────────────────────────────┐
│  Layer 4: BUSINESS METRICS  (revenue, user satisfaction)   │
├────────────────────────────────────────────────────────────┤
│  Layer 3: ML MODEL METRICS  (accuracy, drift, fairness)    │  ← Notebooks 1 & 2
├────────────────────────────────────────────────────────────┤
│  Layer 2: APPLICATION METRICS  (API latency, error rate)   │
├────────────────────────────────────────────────────────────┤
│  Layer 1: INFRASTRUCTURE METRICS (CPU, RAM, disk, GPU)     │
└────────────────────────────────────────────────────────────┘
```

### 8.2 Types of Drift
| Type | Technical Name | What changes |
|---|---|---|
| **Data drift** | Covariate shift | Input feature distributions |
| **Label drift** | Prior probability shift | Target variable distribution |
| **Concept drift** | Posterior shift | Relationship between inputs and output |
| **Prediction drift** | — | Model output distribution |

### 8.3 Drift Detection Tests
| Test | Feature type | What it measures |
|---|---|---|
| Kolmogorov-Smirnov (KS) | Continuous | Distribution shape difference |
| Chi-squared | Categorical | Category frequency difference |
| Population Stability Index (PSI) | Any | Overall distribution shift magnitude |
| Maximum Mean Discrepancy (MMD) | High-dimensional | Whole-dataset drift |

### 8.4 Fairness Metrics
| Metric | Formula | Good value |
|---|---|---|
| **Equal Opportunity Score** | TPR(privileged) / TPR(unprivileged) | 0.8 – 1.25 |
| **Demographic Parity** | P(approve\|A) / P(approve\|B) | ≈ 1.0 |
| **Equalised Odds** | Both TPR and FPR equal across groups | ≈ 1.0 |

### 8.5 Infrastructure Monitoring Stack
| Tool | What it monitors |
|---|---|
| **Prometheus** | Time-series metrics (CPU, RAM, API latency) |
| **Grafana** | Visualises Prometheus metrics in dashboards |
| **ELK Stack** | Centralised log aggregation and search |
| **Jaeger / Zipkin** | Distributed tracing |

### 8.6 Dedicated ML Monitoring Tools
| Tool | Speciality |
|---|---|
| **Evidently AI** | Open-source drift reports |
| **WhyLabs** | Cloud-based ML observability |
| **Arize AI** | Enterprise ML observability |
| **nannyml** | Performance estimation without ground truth labels |

---

## STAGE 9 — Retraining & Feedback Loops

| Strategy | When to use |
|---|---|
| **Scheduled retraining** | Weekly/monthly regardless of drift |
| **Drift-triggered** | Only when drift exceeds threshold |
| **Performance-triggered** | When accuracy drops below threshold |
| **Online learning** | Continuous updates (complex, not always stable) |

---

## STAGE 10 — Governance, Security & Compliance

### 10.1 Explainability
| Tool | Method |
|---|---|
| **SHAP** | Feature importance per prediction |
| **LIME** | Local approximation explains individual predictions |
| **InterpretML** | Microsoft's open-source explainability toolkit |

### 10.2 Security Threats
| Threat | Mitigation |
|---|---|
| Model inversion attacks | Limit API calls, differential privacy |
| Adversarial inputs | Input validation, adversarial training |
| Data poisoning | Validate data sources, anomaly detection |
| Secrets in code | `.env` files, secrets managers (Vault, AWS Secrets Manager) |
| Model theft | Rate limiting, watermarking |

---

## Complete Topics Checklist
| Stage | Topics | Covered in Tutorial |
|---|---|---|
| **Raw Data** | SQL, storage, DVC, lineage | No |
| **Data Engineering** | EDA, cleaning, Great Expectations, Airflow | Partial |
| **Feature Engineering** | Encoding, scaling, feature stores | Partial |
| **Model Development** | Algorithm selection, metrics, hyperparameter tuning | Partial |
| **Experiment Tracking** | MLflow, W&B | No |
| **Model Packaging** | joblib, ONNX, model registry | No |
| **DevOps** | Git, Docker, Kubernetes, CI/CD, Terraform | No |
| **Model Serving** | FastAPI, BentoML, canary/blue-green | No |
| **Drift Monitoring** | alibi-detect, Evidently | **Yes — Notebook 1** |
| **Fairness Monitoring** | Equal opportunity, demographic parity, sklego | **Yes — Notebook 2** |
| **Infrastructure Monitoring** | Prometheus, Grafana, ELK | No |
| **Retraining Loops** | Triggers, CT pipelines | No |
| **Governance & Explainability** | SHAP, LIME, model cards, GDPR | No |
| **Security** | Adversarial ML, secrets management | No |

---

## Recommended Learning Order

**Month 1–2 — Foundations**
- Git & GitHub (hands-on with your own repos)
- Python packaging, virtual environments
- SQL basics
- Linux & Bash scripting

**Month 3–4 — Data & Modelling**
- Pandas EDA, cleaning, validation with Great Expectations
- scikit-learn Pipelines, feature engineering, evaluation metrics
- MLflow for experiment tracking
- Extend credit approval notebook with SHAP explanations

**Month 5–6 — Productionisation**
- Build a FastAPI prediction endpoint for the credit model
- Containerise it with Docker
- Set up a basic GitHub Actions CI/CD pipeline

**Month 7–8 — Monitoring & Governance**
- Add Evidently AI drift reports to the credit model pipeline
- Add Prometheus + Grafana for API metrics
- Add SHAP explanations to every prediction

**Month 9–12 — Cloud & Scale**
- Deploy the full pipeline on AWS or GCP (free tier)
- Learn Kubernetes basics (Minikube locally first)
- Learn Terraform to provision cloud resources as code
- Learn Airflow to orchestrate the entire pipeline
