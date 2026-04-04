"""
tests/test_api.py — FastAPI TestClient tests for Stage 6.

Run with:
    pytest tests/test_api.py -v
"""
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared test payload — a valid first-class female passenger (should survive)
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
    "pclass": 1,
    "age": 29.0,
    "sibsp": 0,
    "parch": 0,
    "fare": 211.3,
    "family_size": 1,
    "fare_per_person": 211.3,
    "sex_female": 1,
    "sex_male": 0,
    "embarked_C": 1,
    "embarked_Q": 0,
    "embarked_S": 0,
    "is_alone": 1,
    "alone": 1,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client_with_model():
    """
    Returns a TestClient with a real local pkl model loaded via the fallback
    path. This avoids needing a live MLflow server to run tests.
    MLflow is patched to raise so the lifespan uses the fallback pkl path.
    """
    models_dir = Path(__file__).parent.parent / "models"
    pkl_path = models_dir / "titanic_model_v2.pkl"

    if not pkl_path.exists():
        pytest.skip(f"Model file not found at {pkl_path} — run Stage 4 first")

    from src.api import main as api_main

    # Patch mlflow so it fails → lifespan uses fallback pkl
    with patch("src.api.main.mlflow.pyfunc.load_model", side_effect=Exception("mlflow unavailable in tests")), \
         patch.object(api_main.settings, "fallback_model_path", str(pkl_path)):
        with TestClient(api_main.app, raise_server_exceptions=True) as c:
            yield c

    api_main._model_store.clear()


@pytest.fixture()   # function scope — fresh state per test, no cross-contamination
def client_no_model():
    """
    TestClient where the model store is explicitly empty after startup.
    Simulates the state where model loading failed at startup.
    Saves and restores _model_store so module-scope fixtures stay intact.
    """
    from src.api import main as api_main

    # Save whatever was previously in the store (client_with_model may have loaded it)
    saved_store = dict(api_main._model_store)

    with TestClient(api_main.app, raise_server_exceptions=False) as c:
        # Lifespan may have loaded a fallback — clear to simulate "not ready"
        api_main._model_store.clear()
        yield c

    # Restore previous model store state so other module-scope fixtures still work
    api_main._model_store.clear()
    api_main._model_store.update(saved_store)


# ---------------------------------------------------------------------------
# /health — liveness (always 200)
# ---------------------------------------------------------------------------
def test_health_returns_200(client_with_model):
    response = client_with_model.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_works_even_without_model(client_no_model):
    """Liveness must always return 200 — even if model failed to load."""
    response = client_no_model.get("/health")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# /ready — readiness probe
# ---------------------------------------------------------------------------
def test_ready_returns_200_when_model_loaded(client_with_model):
    response = client_with_model.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert "model_alias" in body
    assert "model_version" in body


def test_ready_returns_503_when_model_not_loaded(client_no_model):
    response = client_no_model.get("/ready")
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# POST /predict — valid payload
# ---------------------------------------------------------------------------
def test_predict_valid_payload_returns_200(client_with_model):
    response = client_with_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "survived" in body
    assert body["survived"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert "model_alias" in body        # alias is present (fallback-local or champion)
    assert "model_version" in body      # version is present


def test_predict_third_class_male(client_with_model):
    """Third-class male from Southampton — expected to not survive."""
    payload = {**VALID_PAYLOAD, "pclass": 3, "sex_female": 0, "sex_male": 1,
               "embarked_C": 0, "embarked_S": 1, "fare": 7.9, "fare_per_person": 7.9}
    response = client_with_model.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["survived"] in (0, 1)


# ---------------------------------------------------------------------------
# POST /predict — invalid payload → 422 validation error
# ---------------------------------------------------------------------------
def test_predict_invalid_pclass_returns_422(client_with_model):
    """pclass must be 1, 2, or 3 — pclass=99 should be rejected."""
    bad_payload = {**VALID_PAYLOAD, "pclass": 99}
    response = client_with_model.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_negative_age_returns_422(client_with_model):
    bad_payload = {**VALID_PAYLOAD, "age": -5.0}
    response = client_with_model.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_missing_field_returns_422(client_with_model):
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "fare"}
    response = client_with_model.post("/predict", json=incomplete)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict — 503 when no model loaded
# ---------------------------------------------------------------------------
def test_predict_503_when_model_not_loaded(client_no_model):
    response = client_no_model.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
