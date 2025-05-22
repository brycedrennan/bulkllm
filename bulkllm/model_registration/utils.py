import logging
from typing import Any

import litellm

logger = logging.getLogger(__name__)


def bulkllm_register_models(model_cost_map: dict[str, Any], warn_existing: bool = True):
    for model_name in model_cost_map:
        if warn_existing:
            try:
                model_info = litellm.get_model_info(model_name)
            except Exception:  # noqa
                model_info = None
        if model_info:
            logger.warning(f"Model '{model_name}' already registered")
    litellm.register_model(model_cost_map)
