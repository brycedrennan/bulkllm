import logging
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# Track models registered by this package.
ADDED_MODELS: list[tuple[str, str | None]] = []


def bulkllm_register_models(
    model_cost_map: dict[str, Any], warn_existing: bool = True, *, source: str | None = None
) -> None:
    """Register multiple models with LiteLLM, warning if already present."""

    for model_name in model_cost_map:
        model_info = None
        try:
            model_info = litellm.get_model_info(model_name)
        except Exception:  # noqa: PERF203,BLE001 - broad catch ok here
            model_info = None
        if model_info:
            if warn_existing:
                logger.warning(f"Model '{model_name}' already registered")
        else:
            entry = (model_name, source)
            if entry not in ADDED_MODELS:
                ADDED_MODELS.append(entry)

    litellm.register_model(model_cost_map)


def print_added_models() -> None:
    for model_name, model_source in ADDED_MODELS:
        print(f"{model_name} - {model_source}")
