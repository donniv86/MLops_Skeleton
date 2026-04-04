"""
Stage 6 — Model Serving with FastAPI
=====================================
Serves the @champion model from the MLflow Model Registry.

Endpoints:
  GET  /health  — liveness probe  (always 200)
  GET  /ready   — readiness probe (200 if model loaded, 503 if not)
  POST /predict — survival prediction for a Titanic passenger

Run locally:
  uvicorn src.api.main:app --reload --port 8000

Test:
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"pclass":1,"age":29.0,"sibsp":0,"parch":0,"fare":211.3,
            "family_size":1,"fare_per_person":211.3,
            "sex_female":1,"sex_male":0,
            "embarked_C":1,"embarked_Q":0,"embarked_S":0,
            "is_alone":1,"alone":1}'
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from src.api.config import settings

# ---------------------------------------------------------------------------
# Logging — structured JSON to stdout
# ---------------------------------------------------------------------------
logger.remove()
logger.add(
    sys.stdout,
    level=settings.log_level,
    serialize=True,  # JSON output
    backtrace=False,
    diagnose=False,
)

# ---------------------------------------------------------------------------
# Feature column order must match training pipeline output (Stage 2)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "pclass", "age", "sibsp", "parch", "fare",
    "family_size", "fare_per_person",
    "sex_female", "sex_male",
    "embarked_C", "embarked_Q", "embarked_S",
    "is_alone", "alone",
]

# Shared app state — populated during lifespan startup
_model_store: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup, clean up at shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Official FastAPI pattern (2024+) for shared resource loading.
    Model is loaded BEFORE the first request is handled.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.model_name}@{settings.model_alias}"

    try:
        logger.info(f"Loading model from MLflow registry: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        # Retrieve version metadata for response headers
        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
        _model_store["model"] = model
        _model_store["alias"] = settings.model_alias
        _model_store["version"] = mv.version
        logger.info(
            f"Model loaded — alias=@{settings.model_alias} version={mv.version}"
        )

    except Exception as exc:
        # Fall back to local pkl so the service still starts in offline mode
        fallback = Path(settings.fallback_model_path)
        if fallback.exists():
            import pickle
            with open(fallback, "rb") as f:
                model = pickle.load(f)
            _model_store["model"] = model
            _model_store["alias"] = "fallback-local"
            _model_store["version"] = "local"
            logger.warning(
                f"MLflow unreachable ({exc}). Loaded fallback: {fallback}"
            )
        else:
            logger.error(
                f"MLflow unreachable and no fallback pkl found at {fallback}. "
                "Service will return 503 on /predict."
            )

    yield  # ← application handles requests between here and shutdown

    # Cleanup on shutdown
    _model_store.clear()
    logger.info("Model unloaded — shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Titanic Survival Predictor",
    description="MLOps Stage 6 — serves @champion model from MLflow registry",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PassengerFeatures(BaseModel):
    """Input schema — all 14 features produced by the Stage 2 pipeline."""
    model_config = {"json_schema_extra": {
        "example": {
            "pclass": 1, "age": 29.0, "sibsp": 0, "parch": 0, "fare": 211.3,
            "family_size": 1, "fare_per_person": 211.3,
            "sex_female": 1, "sex_male": 0,
            "embarked_C": 1, "embarked_Q": 0, "embarked_S": 0,
            "is_alone": 1, "alone": 1,
        }
    }}

    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1/2/3)")
    age: float = Field(..., ge=0.0, le=120.0)
    sibsp: int = Field(..., ge=0)
    parch: int = Field(..., ge=0)
    fare: float = Field(..., ge=0.0)
    family_size: int = Field(..., ge=0)
    fare_per_person: float = Field(..., ge=0.0)
    sex_female: int = Field(..., ge=0, le=1)
    sex_male: int = Field(..., ge=0, le=1)
    embarked_C: int = Field(..., ge=0, le=1)
    embarked_Q: int = Field(..., ge=0, le=1)
    embarked_S: int = Field(..., ge=0, le=1)
    is_alone: int = Field(..., ge=0, le=1)
    alone: int = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    """Output schema."""
    survived: int
    probability: float
    model_alias: str
    model_version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
def health():
    """Liveness probe — returns 200 as long as the process is running."""
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
def ready():
    """
    Readiness probe — returns 200 only when model is loaded.
    Returns 503 Service Unavailable if model failed to load at startup.
    Docker HEALTHCHECK and k8s readinessProbe should hit this endpoint.
    """
    if "model" not in _model_store:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — service is not ready",
        )
    return {
        "status": "ready",
        "model_loaded": True,
        "model_alias": _model_store.get("alias"),
        "model_version": _model_store.get("version"),
    }


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(payload: PassengerFeatures, request: Request):
    """
    Predict Titanic survival for a single passenger.

    Returns:
      - survived: 0 or 1
      - probability: float between 0 and 1 (probability of survival)
      - model_alias: which alias was used (e.g. 'champion')
      - model_version: MLflow model version number
    """
    if "model" not in _model_store:
        raise HTTPException(
            status_code=503,
            detail="Model not ready — try again in a moment",
        )

    model = _model_store["model"]

    # Build a single-row DataFrame in the exact column order the pipeline expects
    row = pd.DataFrame(
        [[getattr(payload, col) for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )
    # alone column was bool in Parquet — ensure int for sklearn
    row["alone"] = row["alone"].astype(int)

    try:
        prediction = model.predict(row)
        survived = int(prediction[0])

        # Attempt to get probability — not all model flavours support predict_proba
        try:
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(row)[0][1])
            else:
                # MLflow pyfunc wrapper — unwrap sklearn model if available
                inner = getattr(model, "_model_impl", None)
                if inner is not None and hasattr(inner, "predict_proba"):
                    proba = float(inner.predict_proba(row)[0][1])
                else:
                    proba = float(survived)  # binary 0/1 fallback
        except Exception:
            proba = float(survived)

        logger.info(
            "prediction",
            survived=survived,
            probability=round(proba, 4),
            pclass=payload.pclass,
            age=payload.age,
            sex_female=payload.sex_female,
            model_alias=_model_store["alias"],
            model_version=_model_store["version"],
        )

        return PredictResponse(
            survived=survived,
            probability=round(proba, 4),
            model_alias=_model_store["alias"],
            model_version=_model_store["version"],
        )

    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
