from __future__ import annotations

import litellm
import typer

from bulkllm.llm_configs import create_model_configs
from bulkllm.model_registration.main import register_models
from bulkllm.rate_limiter import RateLimiter


def _canonical_model_name(name: str) -> str:
    """Return canonical name for ``name`` dropping provider wrappers."""
    if name.startswith("openrouter/"):
        parts = name.split("/", 2)
        if len(parts) == 3 and parts[1] in {"anthropic", "openai", "gemini"}:
            return f"{parts[1]}/{parts[2]}"
    if name.startswith("bedrock/"):
        after = name[len("bedrock/") :]
        if after.startswith("anthropic."):
            return "anthropic/" + after[len("anthropic.") :]
    return name


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback(invoke_without_command=True)
def main_callback() -> None:
    """BulkLLM command line interface."""


@app.command("list-models")
def list_models() -> None:
    """List all models registered with LiteLLM."""
    register_models()
    for model in sorted(litellm.model_cost):
        typer.echo(model)


@app.command("list-unique-models")
def list_unique_models() -> None:
    """List unique models, collapsing provider duplicates."""
    register_models()
    unique: set[str] = set()
    for model in sorted(litellm.model_cost):
        canonical = _canonical_model_name(model)
        unique.add(canonical)
    for name in sorted(unique):
        typer.echo(name)


@app.command("list-missing-rate-limits")
def list_missing_rate_limits() -> None:
    """List models without a configured rate limit."""
    register_models()
    limiter = RateLimiter()
    for model in sorted(litellm.model_cost):
        if limiter.get_rate_limit_for_model(model) is limiter.default_rate_limit:
            typer.echo(model)


@app.command("list-missing-model-configs")
def list_missing_model_configs() -> None:
    """List models without a corresponding LLMConfig."""
    register_models()
    known = {cfg.litellm_model_name for cfg in create_model_configs()}
    for model in sorted(litellm.model_cost):
        if model not in known:
            typer.echo(model)


def main() -> None:  # pragma: no cover - CLI entry point
    app()


if __name__ == "__main__":  # pragma: no cover - CLI runner
    main()
