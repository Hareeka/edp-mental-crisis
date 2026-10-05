from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MC_", extra="ignore")

    app_name: str = "MindCompanion"
    database_url: str = f"sqlite:///{BACKEND_DIR / 'mindcompanion.db'}"
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60 * 12
    # Fernet key for encrypting stored message text. Generated per process if unset (dev only).
    encryption_key: str | None = None
    nlp_backend: str = "transformer"  # "transformer" | "lexicon"
    risk_model_path: Path = BACKEND_DIR / "artifacts" / "risk_model.joblib"
    store_message_text: bool = True
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # Optional OpenAI-compatible endpoint for Low/Moderate replies. High risk always uses fixed templates.
    llm_api_base: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 15.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
