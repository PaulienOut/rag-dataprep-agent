from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    metadata_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 900
    chunk_overlap: int = 150


def load_settings(env_file: str | Path = ".env") -> Settings:
    load_dotenv(env_file)
    return Settings(openai_api_key=os.getenv("OPENAI_API_KEY"))
