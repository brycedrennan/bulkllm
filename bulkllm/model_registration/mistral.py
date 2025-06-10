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


def convert_mistral_to_litellm(mistral_model: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a Mistral model dict to LiteLLM format."""
    model_id = mistral_model.get("id")
    if not model_id:
        logger.warning("Skipping model due to missing id: %s", mistral_model)
        return None

    litellm_model_name = f"mistral/{model_id}"

    model_info = {
        "litellm_provider": "mistral",
        "mode": "chat",
    }

    context = mistral_model.get("max_context_length")
    if context is not None:
        model_info["max_input_tokens"] = context

    caps = mistral_model.get("capabilities", {})
    if caps.get("function_calling"):
        model_info["supports_function_calling"] = True
    if caps.get("vision"):
        model_info["supports_vision"] = True

    return {"model_name": litellm_model_name, "model_info": model_info}


@cache
def get_mistral_models(*, use_cached: bool = True) -> dict[str, Any]:
    """Return models from the Mistral list endpoint or cached data."""
    if use_cached:
        try:
            data = load_cached_provider_data("mistral")
        except FileNotFoundError:
            use_cached = False
    if not use_cached:
        url = "https://api.mistral.ai/v1/models"
        api_key = os.getenv("MISTRAL_API_KEY", "")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            save_cached_provider_data("mistral", data)
        except requests.RequestException as exc:  # noqa: PERF203 - broad catch ok here
            logger.warning("Failed to fetch Mistral models: %s", exc)
            return {}
    models: dict[str, Any] = {}
    for item in data.get("data", []):
        converted = convert_mistral_to_litellm(item)
        if converted:
            models[converted["model_name"]] = converted["model_info"]
    return models


@cache
def register_mistral_models_with_litellm() -> None:
    """Fetch and register Mistral models with LiteLLM."""
    bulkllm_register_models(get_mistral_models(), source="mistral")
