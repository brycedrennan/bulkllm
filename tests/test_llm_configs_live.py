"""Live smoke test that sends a minimal prompt to every configured model."""

import os
from collections.abc import Iterable

import pytest

from bulkllm.llm import completion, initialize_litellm
from bulkllm.llm_configs import model_resolver

RUN_ENV_VAR = "BULKLLM_RUN_LIVE_LLM_TESTS"
MAX_COMPLETION_TOKENS = 8
PROMPT = "Reply with the single lowercase word 'pong'. Do not add any other text."


if not os.getenv(RUN_ENV_VAR):
    pytest.skip(f"Set {RUN_ENV_VAR}=1 to run live LLM ping tests.", allow_module_level=True)

initialize_litellm()


def _normalize_content(content) -> str:
    """Convert LiteLLM response message content into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, Iterable):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif item.get("type") == "output_text" and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts).strip()

    return str(content).strip()


@pytest.mark.parametrize("llm_config", model_resolver(["all"]), ids=lambda cfg: cfg.slug)
def test_llm_config_ping(llm_config):
    kwargs = llm_config.completion_kwargs()

    # Cap completion tokens to keep the smoke test inexpensive.
    if "max_completion_tokens" in kwargs and kwargs["max_completion_tokens"] is not None:
        kwargs["max_completion_tokens"] = max(1, min(kwargs["max_completion_tokens"], MAX_COMPLETION_TOKENS))
    else:
        max_tokens = kwargs.get("max_tokens", MAX_COMPLETION_TOKENS)
        kwargs["max_tokens"] = max(1, min(max_tokens, MAX_COMPLETION_TOKENS))

    messages = []
    if llm_config.system_prompt:
        messages.append({"role": "system", "content": llm_config.system_prompt})
    messages.append({"role": "user", "content": PROMPT})
    kwargs["messages"] = messages

    response = completion(**kwargs)
    content = response.choices[0].message.get("content")  # type: ignore[attr-defined]
    text = _normalize_content(content)

    assert "pong" in text.lower(), f"{llm_config.slug} replied {text!r}"
