from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import logfire
from openai import OpenAI

from rag_dataprep_agent.config import Settings

_configured = False


def configure_logfire(settings: Settings) -> None:
    """Configure Logfire once when monitoring is enabled."""
    global _configured
    if _configured or not settings.logfire_enabled:
        return

    logfire.configure(
        send_to_logfire="if-token-present",
        service_name=settings.logfire_service_name,
        environment=settings.logfire_environment,
    )
    _configured = True


def instrument_openai_client(client: OpenAI | None, settings: Settings) -> None:
    if client is not None and settings.logfire_enabled:
        logfire.instrument_openai(client)


def span(settings: Settings, msg_template: str, **attributes: Any):
    if not settings.logfire_enabled:
        return nullcontext()
    return logfire.span(msg_template, **attributes)


def info(settings: Settings, msg_template: str, **attributes: Any) -> None:
    if settings.logfire_enabled:
        logfire.info(msg_template, **attributes)
