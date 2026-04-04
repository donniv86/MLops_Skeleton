"""
API configuration using pydantic-settings.
Values are read from environment variables (or .env file).
See .env.example for reference.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "titanic-classifier"
    model_alias: str = "champion"

    # Logging
    log_level: str = "INFO"

    # Paths (fallback to local pkl if MLflow registry is unavailable)
    models_dir: str = "models"
    fallback_model_path: str = "models/titanic_model_v2.pkl"


settings = Settings()
