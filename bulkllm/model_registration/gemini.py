import logging
import os
from functools import cache
from typing import Any

import litellm
import requests

logger = logging.getLogger(__name__)


@cache
def get_gemini_models() -> dict[str, Any]:
    """Return models from the Google Gemini list endpoint."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    params = {"key": api_key} if api_key else None
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:  # noqa: PERF203 - broad catch ok here
        logger.warning("Failed to fetch Gemini models: %s", exc)
        return {}
    models: dict[str, Any] = {}
    for item in data.get("models", []):
        name = item.get("name")
        if name:
            model_id = name.split("/")[-1]
            models[f"gemini/{model_id}"] = {
                "litellm_provider": "gemini",
                "mode": "chat",
            }
    return models


@cache
def register_gemini_models_with_litellm() -> None:
    """Fetch and register Gemini models with LiteLLM."""
    litellm.register_model(get_gemini_models())
