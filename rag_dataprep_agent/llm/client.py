from __future__ import annotations

from openai import OpenAI


def build_openai_client(api_key: str | None) -> OpenAI | None:
    if not api_key:
        return None
    return OpenAI(api_key=api_key)
