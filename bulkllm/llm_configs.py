import logging
import re
from datetime import date
from functools import cache

import litellm

from bulkllm.schema import LLMConfig

logger = logging.getLogger(__name__)

default_max_tokens = None

default_temperature = 1

default_system_prompt = "You are a helpful AI assistant."
default_system_prompt = ""

default_models = [
    LLMConfig(
        slug="gemini-2.5-flash-preview-04-17",
        display_name="Gemini 2.5 Flash Preview 20250417",
        company_name="Google",
        litellm_model_name="gemini/gemini-2.5-flash-preview-04-17",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 4, 17),
    ),
    LLMConfig(
        slug="openai-o3-2025-04-16-low",
        display_name="o3 (Low) 20250416",
        company_name="openai",
        litellm_model_name="openai/o3-2025-04-16",
        temperature=1,
        max_completion_tokens=8000 - 1,
        thinking_config={},
        system_prompt=default_system_prompt,
        reasoning_effort="low",
        release_date=date(2025, 4, 16),
        is_reasoning=True,
    ),
    LLMConfig(
        slug="openai-o3-2025-04-16-medium",
        display_name="o3 (Medium) 20250416",
        company_name="openai",
        litellm_model_name="openai/o3-2025-04-16",
        temperature=1,
        max_completion_tokens=8000,
        thinking_config={},
        system_prompt=default_system_prompt,
        reasoning_effort="medium",
        release_date=date(2025, 4, 16),
        is_reasoning=True,
    ),
    LLMConfig(
        slug="openai-o3-2025-04-16-high",
        display_name="o3 (High) 20250416",
        company_name="openai",
        litellm_model_name="openai/o3-2025-04-16",
        temperature=1,
        max_completion_tokens=8000 - 2,
        thinking_config={},
        system_prompt=default_system_prompt,
        reasoning_effort="high",
        release_date=date(2025, 4, 16),
        is_reasoning=True,
    ),
    LLMConfig(
        slug="openai-o4-mini-2025-04-16-low",
        display_name="o4-mini (Low) 20250416",
        company_name="openai",
        litellm_model_name="openai/o4-mini-2025-04-16",
        temperature=1,
        max_completion_tokens=8000 - 1,
        thinking_config={},
        system_prompt=default_system_prompt,
        reasoning_effort="low",
        release_date=date(2025, 4, 16),
        is_reasoning=True,
    ),
    LLMConfig(
        slug="openai-o4-mini-2025-04-16-medium",
        display_name="o4-mini (Medium) 20250416",
        company_name="openai",
        litellm_model_name="openai/o4-mini-2025-04-16",
        temperature=1,
        max_completion_tokens=8000,
        thinking_config={},
        system_prompt=default_system_prompt,
        reasoning_effort="medium",
        release_date=date(2025, 4, 16),
        is_reasoning=True,
    ),
    LLMConfig(
        slug="openai-o4-mini-2025-04-16-high",
        display_name="o4-mini (High) 20250416",
        company_name="openai",
        litellm_model_name="openai/o4-mini-2025-04-16",
        temperature=1,
        max_completion_tokens=8000 - 2,
        thinking_config={},
        system_prompt=default_system_prompt,
        reasoning_effort="high",
        release_date=date(2025, 4, 16),
        is_reasoning=True,
    ),
    LLMConfig(
        slug="openai-o3-mini-2025-01-31-low",
        display_name="o3-mini (Low) 20250131",
        company_name="openai",
        litellm_model_name="openai/o3-mini-2025-01-31",
        temperature=1,
        max_tokens=8000 - 1,
        thinking_config={},
        system_prompt=default_system_prompt,
        reasoning_effort="low",
        release_date=date(2025, 1, 31),
        is_reasoning=True,
    ),
    LLMConfig(
        slug="openai-o3-mini-2025-01-31-medium",
        display_name="o3-mini (Medium) 20250131",
        company_name="openai",
        litellm_model_name="openai/o3-mini-2025-01-31",
        temperature=1,
        max_tokens=8000,
        thinking_config={},
        system_prompt=default_system_prompt,
        reasoning_effort="medium",
        release_date=date(2025, 1, 31),
        is_reasoning=True,
    ),
    LLMConfig(
        slug="openai-o3-mini-2025-01-31-high",
        display_name="o3-mini (High) 20250131",
        company_name="openai",
        litellm_model_name="openai/o3-mini-2025-01-31",
        temperature=1,
        max_tokens=8000 - 2,
        thinking_config={},
        system_prompt=default_system_prompt,
        reasoning_effort="high",
        release_date=date(2025, 1, 31),
        is_reasoning=True,
    ),
    LLMConfig(
        slug="openai-gpt-4.1-2025-04-14",
        display_name="GPT-4.1 20250414",
        company_name="openai",
        litellm_model_name="openai/gpt-4.1-2025-04-14",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 4, 14),
    ),
    LLMConfig(
        slug="openai-gpt-4.1-mini-2025-04-14",
        display_name="GPT-4.1 Mini 20250414",
        company_name="openai",
        litellm_model_name="openai/gpt-4.1-mini-2025-04-14",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 4, 14),
    ),
    LLMConfig(
        slug="openai-gpt-4.1-nano-2025-04-14",
        display_name="GPT-4.1 Nano 20250414",
        company_name="openai",
        litellm_model_name="openai/gpt-4.1-nano-2025-04-14",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 4, 14),
    ),
    LLMConfig(
        slug="openai-gpt-4o-2024-05-13",
        display_name="GPT-4o 20240513",
        company_name="openai",
        litellm_model_name="openai/gpt-4o-2024-05-13",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 5, 13),
    ),
    LLMConfig(
        slug="openai-gpt-4o-2024-08-06",
        display_name="GPT-4o 20240806",
        company_name="openai",
        litellm_model_name="openai/gpt-4o-2024-08-06",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 8, 6),
    ),
    LLMConfig(
        slug="openai-gpt-4o-2024-11-20",
        display_name="GPT-4o 20241120",
        company_name="openai",
        litellm_model_name="openai/gpt-4o-2024-11-20",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 11, 20),
    ),
    LLMConfig(
        slug="openai-gpt-4o-mini-2024-07-18",
        display_name="GPT-4o mini 20240718",
        company_name="openai",
        litellm_model_name="openai/gpt-4o-mini-2024-07-18",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 7, 18),
    ),
    LLMConfig(
        slug="openai-gpt-3.5-turbo-0125",
        display_name="GPT-3.5 Turbo 20230125",
        company_name="openai",
        litellm_model_name="openai/gpt-3.5-turbo-0125",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2023, 1, 25),
    ),
    LLMConfig(
        slug="openai-gpt-3.5-turbo-1106",
        display_name="GPT-3.5 Turbo 20231106",
        company_name="openai",
        litellm_model_name="openai/gpt-3.5-turbo-1106",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2023, 11, 6),
    ),
    LLMConfig(
        slug="openrouter-deepseek-deepseek-r1",
        display_name="DeepSeek R1 20250120",
        company_name="DeepSeek",
        litellm_model_name="openrouter/deepseek/deepseek-r1",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        is_reasoning=True,
        release_date=date(2025, 1, 20),
    ),
    LLMConfig(
        slug="openrouter-deepseek-deepseek-chat-v3-0324",
        display_name="DeepSeek V3 20250324",
        company_name="DeepSeek",
        litellm_model_name="openrouter/deepseek/deepseek-chat-v3-0324",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 3, 24),
    ),
    LLMConfig(
        slug="openrouter-deepseek-deepseek-chat",
        display_name="DeepSeek V3 20241226",
        company_name="DeepSeek",
        litellm_model_name="openrouter/deepseek/deepseek-chat",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 12, 26),
    ),
    LLMConfig(
        slug="openrouter-amazon-nova-pro-v1",
        display_name="Nova Pro V1",
        company_name="Amazon",
        litellm_model_name="openrouter/amazon/nova-pro-v1",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 12, 3),
    ),
    LLMConfig(
        slug="openrouter-qwen-qwq-32b",
        display_name="Qwen QwQ-32B",
        company_name="Alibaba",
        litellm_model_name="openrouter/qwen/qwq-32b",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        is_reasoning=True,
        release_date=date(2025, 3, 6),
    ),
    LLMConfig(
        slug="openrouter-meta-llama-llama-3.3-70b-instruct",
        display_name="Llama 3.3 70b Instruct",
        company_name="Meta",
        litellm_model_name="openrouter/meta-llama/llama-3.3-70b-instruct",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 12, 6),
    ),
    LLMConfig(
        slug="openrouter-mistralai-mistral-large-2411",
        display_name="Mistral Large 2411",
        company_name="MistralAI",
        litellm_model_name="openrouter/mistralai/mistral-large-2411",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 11, 18),
    ),
    LLMConfig(
        slug="openrouter-mistralai-mistral-small-3.1-24b-instruct",
        display_name="Mistral Small 3.1 24b Instruct 20250317",
        company_name="MistralAI",
        litellm_model_name="openrouter/mistralai/mistral-small-3.1-24b-instruct",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 3, 17),
    ),
    LLMConfig(
        slug="openrouter-google-gemma-3-27b-it",
        display_name="Gemma 3 27b IT",
        company_name="Google",
        litellm_model_name="openrouter/google/gemma-3-27b-it",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 3, 12),
    ),
    LLMConfig(
        slug="anthropic-claude-3.7-sonnet-20250219",
        display_name="Claude 3.7 Sonnet 20250219",
        company_name="Anthropic",
        litellm_model_name="anthropic/claude-3-7-sonnet-20250219",
        temperature=default_temperature,
        max_tokens=8000,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 2, 19),
    ),
    LLMConfig(
        slug="anthropic-claude-3.7-sonnet-20250219-thinking",
        display_name="Claude 3.7 Sonnet (thinking) 20250219",
        company_name="Anthropic",
        litellm_model_name="anthropic/claude-3-7-sonnet-20250219",
        temperature=default_temperature,
        max_tokens=8000 + 1,
        thinking_config={"type": "enabled", "budget_tokens": 2048},
        system_prompt=default_system_prompt,
        release_date=date(2025, 2, 19),
        is_reasoning=True,
    ),
    LLMConfig(
        slug="anthropic-claude-3.5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet 20241022",
        company_name="Anthropic",
        litellm_model_name="anthropic/claude-3-5-sonnet-20241022",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 10, 22),
    ),
    LLMConfig(
        slug="anthropic-claude-3.5-sonnet-20240620",
        display_name="Claude 3.5 Sonnet 20240620",
        company_name="Anthropic",
        litellm_model_name="anthropic/claude-3-5-sonnet-20240620",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 6, 20),
    ),
    LLMConfig(
        slug="anthropic-claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku 20241022",
        company_name="Anthropic",
        litellm_model_name="anthropic/claude-3-5-haiku-20241022",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 10, 22),
    ),
    LLMConfig(
        slug="gemini-gemini-1.5-flash-002",
        display_name="Gemini 1.5 Flash 002",
        company_name="Google",
        litellm_model_name="gemini/gemini-1.5-flash-002",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 9, 24),
    ),
    LLMConfig(
        slug="gemini-gemini-1.5-pro-002",
        display_name="Gemini 1.5 Pro 002",
        company_name="Google",
        litellm_model_name="gemini/gemini-1.5-pro-002",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 9, 24),
    ),
    LLMConfig(
        slug="gemini-gemini-2.0-flash-lite",
        display_name="Gemini 2.0 Flash Lite",
        company_name="Google",
        litellm_model_name="gemini/gemini-2.0-flash-lite",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 2, 5),
    ),
    LLMConfig(
        slug="gemini-gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        company_name="Google",
        litellm_model_name="gemini/gemini-2.0-flash",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 2, 5),
    ),
    LLMConfig(
        slug="gemini-gemini-2.5-pro-preview-03-25",
        display_name="Gemini 2.5 Pro Preview 20250325",
        company_name="Google",
        litellm_model_name="gemini/gemini-2.5-pro-preview-03-25",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 3, 25),
    ),
    LLMConfig(
        slug="openrouter-meta-llama-llama-4-maverick",
        display_name="Llama 4 Maverick",
        company_name="Meta",
        litellm_model_name="openrouter/meta-llama/llama-4-maverick",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 4, 5),
    ),
    LLMConfig(
        slug="openrouter-meta-llama-llama-4-scout",
        display_name="Llama 4 Scout",
        company_name="Meta",
        litellm_model_name="openrouter/meta-llama/llama-4-scout",
        temperature=default_temperature,
        max_tokens=default_max_tokens,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 4, 5),
    ),
    LLMConfig(
        slug="xai-grok-2-1212",
        display_name="Grok 2 20241212",
        company_name="xai",
        litellm_model_name="xai/grok-2-1212",
        temperature=default_temperature,
        max_tokens=8000,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2024, 12, 12),
    ),
    LLMConfig(
        slug="xai-grok-3-20250409",
        display_name="Grok 3 Beta",
        company_name="xai",
        litellm_model_name="xai/grok-3-beta",
        temperature=default_temperature,
        max_tokens=8000,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 4, 9),
    ),
    LLMConfig(
        slug="xai-grok-3-mini-low-20250409",
        display_name="Grok 3 Mini Beta (low)",
        company_name="xai",
        litellm_model_name="xai/grok-3-mini-beta",
        temperature=default_temperature,
        max_tokens=32000,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 4, 9),
        reasoning_effort="low",
        is_reasoning=True,
    ),
    LLMConfig(
        slug="xai-grok-3-mini-high-20250409",
        display_name="Grok 3 Mini Beta (high)",
        company_name="xai",
        litellm_model_name="xai/grok-3-mini-beta",
        temperature=default_temperature,
        max_tokens=32000,
        thinking_config={},
        system_prompt=default_system_prompt,
        release_date=date(2025, 4, 9),
        reasoning_effort="high",
        is_reasoning=True,
    ),
]


def create_model_configs(system_prompt: str | None = "You are a helpful AI assistant."):
    new_configs = []
    for llm_config in default_models:
        new_config = llm_config.model_copy()
        new_config.system_prompt = system_prompt
        new_configs.append(new_config)
    return new_configs


@cache
def model_info():
    """Gather info for each model config."""
    from bulkllm.model_registration.main import register_models
    from bulkllm.rate_limiter import RateLimiter

    register_models()

    # Initialize rate limiter to fetch rate limits
    rate_limiter = RateLimiter()
    prompt_tokens = 1_700_000  # Fixed input tokens for all models

    # Company → colour and logo mappings used for charts / table
    company_colors = {
        "openai": "rgba(116, 170, 156, 1)",
        "anthropic": "rgb(204, 120, 92)",
        "google": "yellow",
        "gemini": "yellow",
        "vertex": "yellow",
        "deepseek": "rgba(83, 106, 245, 1)",
        "xai": "purple",
        "meta": "rgba(49, 111, 246, 1)",
        "llama": "rgba(49, 111, 246, 1)",
        "amazon": "rgba(255, 153, 0, 1)",
    }

    # Function to create a simpleicons logo URL for a company slug.
    def get_company_icon(c_slug: str) -> str:
        """Return an SVG logo URL from simpleicons CDN (falls back to generic icon)."""
        base = "https://cdn.simpleicons.org/"
        # Some company names differ from slug we want in URL
        simpleicons_overrides = {
            "openai": "openai",
            "anthropic": "anthropic",
            "google": "google",
            "gemini": "google",  # gemini not in simpleicons, use google icon
            "vertex": "googlecloud",
            "meta": "meta",
            "llama": "meta",
            "amazon": "amazon",
            "deepseek": "openrouter",  # fallback generic
            "xai": "x",
        }
        icon_slug = simpleicons_overrides.get(c_slug, c_slug)
        return f"{base}{icon_slug}/FFFFFF"

    # Default color for unknown companies
    default_color = "gray"

    model_entries: list[dict] = []

    for llm in default_models:
        completion_tokens = 4_800_000 if llm.is_reasoning else 1_800_000

        prompt_cost: float | None = None
        completion_cost: float | None = None
        total_cost: float | None = None

        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=llm.litellm_model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            total_cost = prompt_cost + completion_cost
        except Exception as e:  # noqa
            logger.warning("Could not calculate cost for model %s: %s", llm.litellm_model_name, e)

        # Retrieve rate limit values from RateLimiter
        model_rl = rate_limiter.get_rate_limit_for_model(llm.litellm_model_name)
        tpm = model_rl.tpm
        itpm = model_rl.itpm
        otpm = model_rl.otpm
        rpm = model_rl.rpm

        # Determine colour & icon
        company = llm.company_name.lower()
        color = company_colors.get(company, default_color)
        company_icon = get_company_icon(company)

        # Remove trailing release-date numbers from display name (6+ digit sequences)
        sanitized_name = re.sub(r"(?:\s*\(?\d{4,}\)?)+$", "", llm.display_name).strip()

        model_entries.append(
            {
                "name": sanitized_name,
                "model_id": llm.litellm_model_name,
                "company": llm.company_name,
                "company_icon": company_icon,
                "release_date": llm.release_date.isoformat() if llm.release_date else None,
                "is_reasoning": llm.is_reasoning,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "tpm": tpm,
                "itpm": itpm,
                "otpm": otpm,
                "rpm": rpm,
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "total_cost": total_cost,
                "color": color,
            }
        )

    # Sort by total cost (descending where available)
    model_entries.sort(key=lambda x: x["total_cost"] if x["total_cost"] is not None else -1, reverse=True)
    return model_entries


@cache
def cheap_model_configs():
    """Return LLMConfig objects whose estimated total cost is less than $1."""
    entries = model_info()
    cheap_ids = {entry["model_id"] for entry in entries if entry["total_cost"] is not None and entry["total_cost"] < 1}
    if not cheap_ids:
        return []
    configs = create_model_configs()
    return [config for config in configs if config.litellm_model_name in cheap_ids]


def model_resolver(model_slugs: list[str]) -> list[LLMConfig]:
    if not model_slugs:
        return cheap_model_configs()

    configs = create_model_configs()
    model_lookup = {config.slug: [config] for config in configs}
    model_group_lookup = {
        "cheap": cheap_model_configs,
        "default": cheap_model_configs,
        "all": configs,
        "reasoning": [config for config in configs if config.is_reasoning],
    }
    found_configs = []
    for slug in model_slugs:
        if slug in model_lookup:
            found_configs.extend(model_lookup[slug])
        elif slug in model_group_lookup:
            found_configs.extend(model_group_lookup[slug])
        else:
            msg = f"Unknown model config: {slug}"
            raise ValueError(msg)
    return found_configs
