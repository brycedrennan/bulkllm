"""Live smoke test that sends a minimal prompt to every configured model."""

import os
from collections.abc import Iterable

import pytest

from bulkllm.llm import completion
from bulkllm.llm_configs import model_resolver

MAX_COMPLETION_TOKENS = 100


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


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="Live ping test runs only outside GitHub Actions")
@pytest.mark.parametrize("llm_config", model_resolver(["all"]), ids=lambda cfg: cfg.slug)
def test_llm_config_ping(llm_config):
    kwargs = llm_config.completion_kwargs()
    PROMPT = "Reply with the single lowercase word 'pong'. Do not add any other text. PING"

    messages = []
    if llm_config.system_prompt:
        messages.append({"role": "system", "content": llm_config.system_prompt})
    messages.append({"role": "user", "content": PROMPT})
    kwargs["messages"] = messages

    response = completion(**kwargs)
    content = response.choices[0].message.get("content")  # type: ignore[attr-defined]
    text = _normalize_content(content)

    assert "pong" in text.lower(), f"{llm_config.slug} replied {text!r}"
