import logging
import os
from functools import cache
from typing import Any

import litellm
import requests

logger = logging.getLogger(__name__)


@cache
def get_openai_models() -> dict[str, Any]:
    """Return models from the OpenAI list endpoint."""
    url = "https://api.openai.com/v1/models"
    api_key = os.getenv("OPENAI_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:  # noqa: PERF203 - broad catch ok here
        logger.warning("Failed to fetch OpenAI models: %s", exc)
        return {}
    models: dict[str, Any] = {}
    for item in data.get("data", []):
        model_id = item.get("id")
        if model_id:
            models[f"openai/{model_id}"] = {
                "litellm_provider": "openai",
                "mode": "chat",
            }
    return models


@cache
def register_openai_models_with_litellm() -> None:
    """Fetch and register OpenAI models with LiteLLM."""
    litellm.register_model(get_openai_models())
