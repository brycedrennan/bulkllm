import logging
import os
from functools import cache
from typing import Any

import requests

from bulkllm.model_registration.utils import bulkllm_register_models

logger = logging.getLogger(__name__)


@cache
def get_anthropic_models() -> dict[str, Any]:
    """Return models from the Anthropic list endpoint."""
    url = "https://api.anthropic.com/v1/models"
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:  # noqa: PERF203 - broad catch ok here
        logger.warning("Failed to fetch Anthropic models: %s", exc)
        return {}
    models: dict[str, Any] = {}
    for item in data.get("data", []):
        model_id = item.get("id")
        if model_id:
            models[f"anthropic/{model_id}"] = {
                "litellm_provider": "anthropic",
                "mode": "chat",
            }
    return models


@cache
def register_anthropic_models_with_litellm() -> None:
    """Fetch and register Anthropic models with LiteLLM."""
    bulkllm_register_models(get_anthropic_models())
