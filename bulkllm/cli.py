from __future__ import annotations

import litellm
import typer

from bulkllm.llm_configs import create_model_configs
from bulkllm.model_registration.canonical import _canonical_model_name
from bulkllm.model_registration.main import register_models
from bulkllm.rate_limiter import RateLimiter

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback(invoke_without_command=True)
def main_callback() -> None:
    """BulkLLM command line interface."""


@app.command("list-models")
def list_models() -> None:
    """List all models registered with LiteLLM."""
    register_models()
    for model, model_info in sorted(litellm.model_cost.items()):
        typer.echo(model)


@app.command("list-unique-models")
def list_unique_models() -> None:
    """List unique models, collapsing provider duplicates."""
    register_models()
    unique: set[str] = set()
    for model, model_info in litellm.model_cost.items():
        canonical = _canonical_model_name(model, model_info)
        if canonical is None:
            continue
        unique.add(canonical)
    for name in sorted(unique):
        typer.echo(name)

    print(f"Total unique models: {len(unique)}")


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
