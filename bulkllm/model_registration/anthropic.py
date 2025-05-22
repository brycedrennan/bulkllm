import logging
import os
from functools import cache
from typing import Any

import requests

from bulkllm.model_registration.utils import (
    bulkllm_register_models,
    load_cached_provider_data,
    save_cached_provider_data,
)

logger = logging.getLogger(__name__)


@cache
def get_anthropic_models(*, use_cached: bool = True) -> dict[str, Any]:
    """Return models from the Anthropic list endpoint or cached data."""
    if use_cached:
        try:
            data = load_cached_provider_data("anthropic")
        except FileNotFoundError:
            use_cached = False
    if not use_cached:
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
            save_cached_provider_data("anthropic", data)
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
    bulkllm_register_models(get_anthropic_models(), source="anthropic")
