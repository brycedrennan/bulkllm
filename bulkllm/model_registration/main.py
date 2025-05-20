import logging
from functools import cache

import litellm

from bulkllm.model_registration.openrouter import register_openrouter_models_with_litellm

logger = logging.getLogger(__name__)

manual_model_registrations = {
    "gemini/gemini-2.5-flash-preview-04-17": {
        "max_tokens": 65536,
        "max_input_tokens": 1048576,
        "max_output_tokens": 65536,
        "max_images_per_prompt": 3000,
        "max_videos_per_prompt": 10,
        "max_video_length": 1,
        "max_audio_length_hours": 8.4,
        "max_audio_per_prompt": 1,
        "max_pdf_size_mb": 30,
        "input_cost_per_audio_token": 0.0000001,
        "input_cost_per_token": 0.00000015,
        "output_cost_per_token": 0.00000060,
        "litellm_provider": "gemini",
        "mode": "chat",
        "rpm": 10,
        "tpm": 250000,
        "supports_system_messages": True,
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_reasoning": True,
        "supports_response_schema": True,
        "supports_audio_output": False,
        "supports_tool_choice": True,
        "supported_modalities": ["text", "image", "audio", "video"],
        "supported_output_modalities": ["text"],
        "source": "https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-preview",
    },
    "openrouter/google/gemini-2.0-pro-exp-02-05:free": {
        "max_tokens": 8000,
        "input_cost_per_token": 0 / 1_000_000,
        "output_cost_per_token": 0 / 1_000_000,
        "litellm_provider": "openrouter",
        "mode": "chat",
    },
    "openrouter/openrouter/quasar-alpha": {
        "max_tokens": 32_000,
        "input_cost_per_token": 0.0 / 1_000_000,
        "output_cost_per_token": 0.0 / 1_000_000,
        "litellm_provider": "openrouter",
        "mode": "chat",
    },
}


def manual_registration():
    for model_name in manual_model_registrations:
        try:
            model_info = litellm.get_model_info(model_name)
        except Exception:  # noqa
            model_info = None
        if model_info:
            logger.warning(f"Model {model_name} already registered")
    litellm.register_model(manual_model_registrations)


@cache
def register_models():
    logger.info("Registering models with LiteLLM")
    register_openrouter_models_with_litellm()
    manual_registration()
