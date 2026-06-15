from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    metadata_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 900
    chunk_overlap: int = 150
    logfire_enabled: bool = False
    logfire_service_name: str = "rag-dataprep-agent"
    logfire_environment: str | None = None


def load_settings(env_file: str | Path = ".env") -> Settings:
    load_dotenv(env_file)
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        logfire_enabled=_env_flag("LOGFIRE_ENABLED"),
        logfire_service_name=os.getenv("LOGFIRE_SERVICE_NAME", "rag-dataprep-agent"),
        logfire_environment=os.getenv("LOGFIRE_ENVIRONMENT"),
    )
