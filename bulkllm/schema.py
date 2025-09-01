from __future__ import annotations

import hashlib as _hashlib
import logging
from collections.abc import Iterable
from datetime import date
from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    computed_field,
    model_validator,
)

logger = logging.getLogger(__name__)


class CiBaseModel(BaseModel):
    @staticmethod
    def _md5(*parts: Iterable[str]) -> str:
        """Return an MD5 hex digest for the joined string parts."""
        parts = [str(part) for part in parts]
        return _hashlib.md5("|".join(parts).encode()).hexdigest()


class LLMConfig(CiBaseModel):
    slug: str
    display_name: str
    company_name: str
    litellm_model_name: str
    llm_family: str
    temperature: float
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    thinking_config: dict | None = None
    system_prompt: str | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    # Optional, new OpenAI GPT-5 parameter. Keep flexible until providers standardize.
    # Use str | int to allow either enum-like strings or numeric levels.
    verbosity: Literal["low", "medium", "high"] | None = None
    release_date: date | None = None
    is_reasoning: bool = False
    is_deprecated: bool | date = False
    timeout: int = 120

    @computed_field
    @property
    def md5_hash(self) -> str:  # noqa: D401
        parts: list[str] = [
            self.litellm_model_name,
            str(self.temperature),
            str(self.max_tokens),
            str(self.thinking_config),
            str(self.system_prompt),
        ]
        if self.reasoning_effort is not None:
            parts.append(str(self.reasoning_effort))
        if self.max_completion_tokens is not None:
            parts.append(str(self.max_completion_tokens))
        # Include verbosity in hash only when set to preserve existing hashes
        if self.verbosity is not None:
            parts.append(f"verbosity={self.verbosity}")
        return self._md5(*parts)

    @model_validator(mode="after")
    def _populate_token_defaults(self):
        """Infer reasonable token defaults from LiteLLM if missing."""
        if self.max_tokens is None:
            import litellm

            from bulkllm.model_registration.main import register_models

            register_models()
            self.max_tokens = min(litellm.get_max_tokens(self.litellm_model_name), 8_000)

        return self

    def completion_kwargs(self) -> dict[str, Any]:
        """
        Return the base keyword-arguments for a litellm completion call that
        depend *solely* on this config.  The caller is still responsible for
        adding a ``messages`` list and final token-window parameters.
        """
        completion_kwargs: dict[str, Any] = {
            "model": self.litellm_model_name,
            "temperature": self.temperature,
            "stream": False,
            "timeout": self.timeout,
        }

        if self.reasoning_effort:
            completion_kwargs["reasoning_effort"] = self.reasoning_effort
            # Ensure LiteLLM forwards this param for OpenAI when supported
            if self.litellm_model_name.startswith("openai/"):
                allowed = completion_kwargs.get("allowed_openai_params") or []
                if "reasoning_effort" not in allowed:
                    allowed = [*allowed, "reasoning_effort"]
                completion_kwargs["allowed_openai_params"] = allowed

        if self.thinking_config:
            completion_kwargs["thinking"] = self.thinking_config

        # Forward verbosity when set. We use extra_body universally and additionally
        # allow a top-level param for OpenAI to support early provider handling.
        if self.verbosity is not None:
            # Ensure extra_body exists and attach verbosity for passthrough
            extra_body = completion_kwargs.get("extra_body") or {}
            if "verbosity" not in extra_body:
                extra_body["verbosity"] = self.verbosity
            completion_kwargs["extra_body"] = extra_body

            # For OpenAI providers, also add top-level param and allow-list it
            if self.litellm_model_name.startswith("openai/"):
                completion_kwargs["verbosity"] = self.verbosity
                allowed = completion_kwargs.get("allowed_openai_params") or []
                if "verbosity" not in allowed:
                    allowed = [*allowed, "verbosity"]
                completion_kwargs["allowed_openai_params"] = allowed

        if self.max_completion_tokens is not None:
            completion_kwargs["max_completion_tokens"] = self.max_completion_tokens
        else:
            completion_kwargs["max_tokens"] = self.max_tokens

        return completion_kwargs
