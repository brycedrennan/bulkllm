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


def convert_openai_to_litellm(openai_model: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an OpenAI model dict to LiteLLM format."""
    model_id = openai_model.get("id")
    if not model_id:
        logger.warning("Skipping model due to missing id: %s", openai_model)
        return None

    litellm_model_name = f"openai/{model_id}"

    model_info = {
        "litellm_provider": "openai",
        "mode": "chat",
    }

    for field in ["object", "created", "owned_by", "root", "parent"]:
        value = openai_model.get(field)
        if value is not None:
            model_info[field] = value

    return {"model_name": litellm_model_name, "model_info": model_info}


@cache
def get_openai_models(*, use_cached: bool = True) -> dict[str, Any]:
    """Return models from the OpenAI list endpoint or cached data."""
    if use_cached:
        try:
            data = load_cached_provider_data("openai")
        except FileNotFoundError:
            use_cached = False
    if not use_cached:
        url = "https://api.openai.com/v1/models"
        api_key = os.getenv("OPENAI_API_KEY", "")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            save_cached_provider_data("openai", data)
        except requests.RequestException as exc:  # noqa: PERF203 - broad catch ok here
            logger.warning("Failed to fetch OpenAI models: %s", exc)
            return {}
    models: dict[str, Any] = {}
    for item in data.get("data", []):
        converted = convert_openai_to_litellm(item)
        if converted:
            models[converted["model_name"]] = converted["model_info"]
    return models


@cache
def register_openai_models_with_litellm() -> None:
    """Fetch and register OpenAI models with LiteLLM."""
    bulkllm_register_models(get_openai_models(), source="openai")
