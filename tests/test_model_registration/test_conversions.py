from bulkllm.model_registration.anthropic import convert_anthropic_to_litellm
from bulkllm.model_registration.gemini import convert_gemini_to_litellm
from bulkllm.model_registration.openai import convert_openai_to_litellm


def test_convert_openai():
    sample = {
        "id": "gpt-4o",
        "object": "model",
        "owned_by": "openai",
        "root": "gpt-4o",
    }
    result = convert_openai_to_litellm(sample)
    assert result == {
        "model_name": "openai/gpt-4o",
        "model_info": {
            "litellm_provider": "openai",
            "mode": "chat",
            "object": "model",
            "owned_by": "openai",
            "root": "gpt-4o",
        },
    }


def test_convert_anthropic():
    sample = {
        "id": "claude-3-7-sonnet-20250219",
        "context_window": 1000,
        "deprecation_date": "2026-02-01",
    }
    result = convert_anthropic_to_litellm(sample)
    assert result == {
        "model_name": "anthropic/claude-3-7-sonnet-20250219",
        "model_info": {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "max_input_tokens": 1000,
            "deprecation_date": "2026-02-01",
        },
    }


def test_convert_gemini():
    sample = {
        "name": "models/gemini-2.5-pro-exp-03-25",
        "inputTokenLimit": 2048,
        "outputTokenLimit": 4096,
        "supported_generation_methods": ["generateContent", "countTokens"],
    }
    result = convert_gemini_to_litellm(sample)
    assert result == {
        "model_name": "gemini/gemini-2.5-pro-exp-03-25",
        "model_info": {
            "litellm_provider": "gemini",
            "mode": "chat",
            "max_input_tokens": 2048,
            "max_output_tokens": 4096,
            "supports_prompt_caching": True,
        },
    }
