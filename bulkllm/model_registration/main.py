import logging
from functools import cache

from bulkllm.model_registration.anthropic import (
    register_anthropic_models_with_litellm,
)
from bulkllm.model_registration.gemini import register_gemini_models_with_litellm
from bulkllm.model_registration.mistral import register_mistral_models_with_litellm
from bulkllm.model_registration.openai import register_openai_models_with_litellm
from bulkllm.model_registration.openrouter import register_openrouter_models_with_litellm
from bulkllm.model_registration.utils import bulkllm_register_models
from bulkllm.model_registration.xai import register_xai_models_with_litellm

logger = logging.getLogger(__name__)

manual_model_registrations = {
    # "gemini/gemini-2.5-flash-preview-04-17": {
    #     "max_tokens": 65536,
    #     "max_input_tokens": 1048576,
    #     "max_output_tokens": 65536,
    #     "max_images_per_prompt": 3000,
    #     "max_videos_per_prompt": 10,
    #     "max_video_length": 1,
    #     "max_audio_length_hours": 8.4,
    #     "max_audio_per_prompt": 1,
    #     "max_pdf_size_mb": 30,
    #     "input_cost_per_audio_token": 0.0000001,
    #     "input_cost_per_token": 0.00000015,
    #     "output_cost_per_token": 0.00000060,
    #     "litellm_provider": "gemini",
    #     "mode": "chat",
    #     "rpm": 10,
    #     "tpm": 250000,
    #     "supports_system_messages": True,
    #     "supports_function_calling": True,
    #     "supports_vision": True,
    #     "supports_reasoning": True,
    #     "supports_response_schema": True,
    #     "supports_audio_output": False,
    #     "supports_tool_choice": True,
    #     "supported_modalities": ["text", "image", "audio", "video"],
    #     "supported_output_modalities": ["text"],
    #     "source": "https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-preview",
    # },
    # "openrouter/google/gemini-2.0-pro-exp-02-05:free": {
    #     "max_tokens": 8000,
    #     "input_cost_per_token": 0 / 1_000_000,
    #     "output_cost_per_token": 0 / 1_000_000,
    #     "litellm_provider": "openrouter",
    #     "mode": "chat",
    # },
    # "openrouter/openrouter/quasar-alpha": {
    #     "max_tokens": 32_000,
    #     "input_cost_per_token": 0.0 / 1_000_000,
    #     "output_cost_per_token": 0.0 / 1_000_000,
    #     "litellm_provider": "openrouter",
    #     "mode": "chat",
    # },
    "anthropic/claude-opus-4-1-20250805": {
        "max_tokens": 32000,
        "max_input_tokens": 200000,
        "max_output_tokens": 32000,
        # "input_cost_per_token": 1.5e-05,
        "input_cost_per_token": 15 / 1_000_000,
        # "output_cost_per_token": 7.5e-05,
        "output_cost_per_token": 75 / 1_000_000,
        "search_context_cost_per_query": {
            "search_context_size_low": 0.01,
            "search_context_size_medium": 0.01,
            "search_context_size_high": 0.01,
        },
        "cache_creation_input_token_cost": 1.875e-05,
        "cache_read_input_token_cost": 1.5e-06,
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_function_calling": True,
        "supports_vision": True,
        "tool_use_system_prompt_tokens": 159,
        "supports_assistant_prefill": True,
        "supports_pdf_input": True,
        "supports_prompt_caching": True,
        "supports_response_schema": True,
        "supports_tool_choice": True,
        "supports_reasoning": True,
        "supports_computer_use": True,
    },
    # Ensure GPT-5 Chat is available and prioritized even if provider fetch fails
    "openai/gpt-5-chat-latest": {
        "max_tokens": 128000,
        "max_input_tokens": 400000,
        "max_output_tokens": 128000,
        "input_cost_per_token": 10 / 1_000_000,
        "output_cost_per_token": 30 / 1_000_000,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_prompt_caching": True,
        "supports_response_schema": True,
        "supports_reasoning": True,
    },
}


@cache
def register_models():
    """Register built-in and manual models with LiteLLM."""
    logger.info("Registering models with LiteLLM")
    # Register manual models first so they take precedence over API models
    bulkllm_register_models(manual_model_registrations, source="manual")
    register_openrouter_models_with_litellm()
    register_openai_models_with_litellm()
    register_anthropic_models_with_litellm()
    register_gemini_models_with_litellm()
    register_mistral_models_with_litellm()
    register_xai_models_with_litellm()
