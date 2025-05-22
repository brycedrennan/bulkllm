import json
import logging
from pathlib import Path
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# Track models registered by this package.
ADDED_MODELS: list[tuple[str, str | None]] = []


DATA_DIR = Path(__file__).resolve().parent / "data"


def get_data_file(provider: str) -> Path:
    """Return path to the cached JSON for a provider."""
    return DATA_DIR / f"{provider}.json"


def load_cached_provider_data(provider: str) -> dict[str, Any]:
    """Load cached raw API response for ``provider``."""
    path = get_data_file(provider)
    with open(path) as f:
        return json.load(f)


def save_cached_provider_data(provider: str, data: dict[str, Any]) -> None:
    """Write raw API response for ``provider`` to cache."""
    path = get_data_file(provider)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving cached provider data for {provider} to {path}")
    with open(path, "w") as f:
        json.dump(data, f)


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
