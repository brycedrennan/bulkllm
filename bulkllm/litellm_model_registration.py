import json
import logging
import time
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import litellm
import requests

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


@cache
def register_models():
    import litellm

    logger.info("Registering models with LiteLLM")
    register_openrouter_models_with_litellm()

    for model_name in manual_model_registrations:
        try:
            model_info = litellm.get_model_info(model_name)
        except Exception:  # noqa
            model_info = None
        if model_info:
            logger.warning(f"Model {model_name} already registered")
    litellm.register_model(manual_model_registrations)


"""
OpenRouter integration module for accessing and managing LLM models via OpenRouter API.

This module provides functionality to:
1. Fetch available models from OpenRouter API
2. Convert OpenRouter model specifications to LiteLLM format
3. Register OpenRouter models with LiteLLM for use in the evaluation framework
4. Cache model information to reduce API calls
5. Calculate token usage costs across different models

The module handles model capability detection (vision, audio, etc.), manages pricing information,
and provides utilities for cost estimation based on token usage.
"""


def convert_openrouter_to_litellm(openrouter_model: dict[str, Any]) -> dict[str, Any] | None:
    """Converts an OpenRouter model dictionary to the LiteLLM format."""

    model_id = openrouter_model.get("id")
    if not model_id:
        print(f"Skipping model due to missing id: {openrouter_model.get('name', 'Unknown')}")
        return None

    litellm_model_name = f"openrouter/{model_id}"

    architecture = openrouter_model.get("architecture", {})
    input_modalities: list[str] = architecture.get("input_modalities", [])
    output_modalities: list[str] = architecture.get("output_modalities", [])

    # Determine mode
    mode = "chat"  # Default
    if "text" in input_modalities and "image" in output_modalities:
        mode = "image_generation"
    elif "text" in input_modalities and "audio" in output_modalities:
        mode = "audio_speech"
    elif "audio" in input_modalities and "text" in output_modalities:
        mode = "audio_transcription"
    elif "image" in input_modalities and "text" in output_modalities:
        mode = "vision"  # LiteLLM uses supports_vision flag, but map mode if possible
    elif "text" not in input_modalities and "text" not in output_modalities:
        # Attempt to infer from modality string if input/output lists are empty/missing
        modality_str = architecture.get("modality", "")
        if "text->image" in modality_str:
            mode = "image_generation"
        elif "text->audio" in modality_str:
            mode = "audio_speech"
        elif "audio->text" in modality_str:
            mode = "audio_transcription"
        elif "image->text" in modality_str:
            mode = "vision"

    pricing = openrouter_model.get("pricing", {})
    input_cost = float(pricing.get("prompt", 0.0))
    output_cost = float(pricing.get("completion", 0.0))

    context_length = openrouter_model.get("context_length")
    top_provider = openrouter_model.get("top_provider", {})
    max_completion_tokens = top_provider.get("max_completion_tokens")

    max_input_tokens = context_length
    # Use max_completion_tokens if available, otherwise fallback to context_length
    max_output_tokens = max_completion_tokens if max_completion_tokens is not None else context_length
    # LiteLLM legacy 'max_tokens': prefer max_output, then max_input
    max_tokens = max_output_tokens if max_output_tokens is not None else max_input_tokens

    supports_vision = "image" in input_modalities
    supports_audio_input = "audio" in input_modalities
    supports_audio_output = "audio" in output_modalities
    supports_web_search = float(pricing.get("web_search", 0.0)) > 0
    # Assume True for chat models, False otherwise. OpenRouter doesn't specify this directly.
    supports_system_messages = mode == "chat"

    # Fields not directly available from OpenRouter API for basic models endpoint
    supports_function_calling = False  # Assume False
    supports_parallel_function_calling = False  # Assume False
    supports_prompt_caching = False  # Assume False
    supports_response_schema = False  # Assume False

    model_info = {
        "max_tokens": max_tokens,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "input_cost_per_token": input_cost,
        "output_cost_per_token": output_cost,
        "litellm_provider": "openrouter",
        "mode": mode,
        "supports_function_calling": supports_function_calling,
        "supports_parallel_function_calling": supports_parallel_function_calling,
        "supports_vision": supports_vision,
        "supports_audio_input": supports_audio_input,
        "supports_audio_output": supports_audio_output,
        "supports_prompt_caching": supports_prompt_caching,
        "supports_response_schema": supports_response_schema,
        "supports_system_messages": supports_system_messages,
        "supports_web_search": supports_web_search,
        # "search_context_cost_per_query": search_context_cost_per_query, # Omit if None
        # "deprecation_date": deprecation_date # Omit if None
    }
    # Clean None values from model_info
    model_info = {k: v for k, v in model_info.items() if v is not None}

    return {"model_name": litellm_model_name, "model_info": model_info}


def get_cache_file_path() -> Path:
    """Returns the path to the cache file."""
    cache_dir = Path.home() / ".cache" / "cognitive-integrity-eval"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "openrouter_models_cache.json"


def read_cache() -> dict[str, Any] | None:
    """Read cached OpenRouter models if available and not expired."""
    cache_file = get_cache_file_path()

    if not cache_file.exists():
        return None

    with open(cache_file) as f:
        cache_data = json.load(f)

    # Check if cache is expired (older than 24 hours)
    cache_timestamp = cache_data.get("timestamp", 0)
    cache_age = time.time() - cache_timestamp
    cache_max_age = 24 * 60 * 60  # 24 hours in seconds

    if cache_age > cache_max_age:
        return None

    return cache_data.get("models")


def write_cache(models: list[dict[str, Any]]) -> None:
    """Write OpenRouter models to cache with current timestamp."""
    cache_file = get_cache_file_path()

    try:
        cache_data = {"timestamp": time.time(), "models": models}

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
    except Exception as e:  # noqa
        print(f"Error writing cache: {e}")


@cache
def get_openrouter_models():
    # Try to get models from cache first
    cached_models = read_cache()
    if cached_models is not None:
        return cached_models

    # If no valid cache exists, fetch from API
    url = "https://openrouter.ai/api/v1/models"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        models_data = response.json()
    except requests.RequestException as exc:  # noqa: PERF203 - broad catch ok here
        logger.warning("Failed to fetch OpenRouter models (offline mode?): %s", exc)
        return {}

    litellm_models = {}
    for model in models_data.get("data", []):
        converted_model = convert_openrouter_to_litellm(model)

        if converted_model:
            litellm_models[converted_model["model_name"]] = converted_model["model_info"]

    # Cache the results
    write_cache(litellm_models)

    return litellm_models


@cache
def register_openrouter_models_with_litellm():
    litellm_model_names_pre_registration = set(litellm.model_cost.keys())
    litellm_models = get_openrouter_models()
    model_names_for_registration = set(litellm_models.keys())
    litellm.register_model(litellm_models)

    litellm_model_names_post_registration = set(litellm.model_cost.keys())

    failed_to_register = model_names_for_registration - litellm_model_names_post_registration
    successfully_registered = litellm_model_names_post_registration - litellm_model_names_pre_registration

    print(f"Registered {len(successfully_registered)} models successfully")
    print(f"Failed to register {len(failed_to_register)} models")
    print(f"Failed to register models: {failed_to_register}")

    # print(litellm.model_cost['openrouter/mistralai/mistral-small-3.1-24b-instruct'])


def calculate_costs_for_all_models(input_tokens=1612249, completion_tokens=1464563, output_csv=None):
    """
    Calculate costs for specified token amounts across all litellm models.

    Args:
        input_tokens (int): Number of input tokens to calculate cost for (default: 1,612,249)
        completion_tokens (int): Number of completion tokens to calculate cost for (default: 1,464,563)
        output_csv (str): Path to save the CSV output (default: None, which prints to console)

    Returns:
        dict: Dictionary mapping model names to calculated costs
    """
    import csv

    import litellm

    # Register OpenRouter models with litellm
    register_openrouter_models_with_litellm()

    costs = {}

    # Calculate costs for each model
    for model_name, model_info in litellm.model_cost.items():
        input_cost = 0
        output_cost = 0

        # Get input cost if available
        if "input_cost_per_token" in model_info and model_info["input_cost_per_token"] is not None:
            input_cost = model_info["input_cost_per_token"] * input_tokens

        # Get output cost if available
        if "output_cost_per_token" in model_info and model_info["output_cost_per_token"] is not None:
            output_cost = model_info["output_cost_per_token"] * completion_tokens

        total_cost = input_cost + output_cost
        costs[model_name] = {"input_cost": input_cost, "output_cost": output_cost, "total_cost": total_cost}

    # Sort models by total cost
    sorted_costs = dict(sorted(costs.items(), key=lambda item: item[1]["total_cost"], reverse=True))

    # Generate default CSV filename if output_csv is None
    if output_csv is True or output_csv == "":
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_csv = f"model_costs_{timestamp}.csv"

    # Write results to CSV if output_csv is specified
    if output_csv:
        with open(output_csv, "w", newline="") as csvfile:
            fieldnames = ["model_name", "input_tokens", "completion_tokens", "input_cost", "output_cost", "total_cost"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for model_name, cost_info in sorted_costs.items():
                writer.writerow(
                    {
                        "model_name": model_name,
                        "input_tokens": input_tokens,
                        "completion_tokens": completion_tokens,
                        "input_cost": cost_info["input_cost"],
                        "output_cost": cost_info["output_cost"],
                        "total_cost": cost_info["total_cost"],
                    }
                )

            print(f"Cost data written to {output_csv}")
    else:
        # Print to console if no CSV output is specified
        for model_name, cost_info in sorted_costs.items():
            print(f"{model_name}: Total Cost = ${cost_info['total_cost']:.2f}")

    return sorted_costs
