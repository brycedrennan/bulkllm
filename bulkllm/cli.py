from __future__ import annotations

import litellm
import typer

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
    for model in sorted(litellm.model_cost):
        typer.echo(model)


@app.command("list-missing-rate-limits")
def list_missing_rate_limits() -> None:
    """List models without a configured rate limit."""
    register_models()
    limiter = RateLimiter()
    for model in sorted(litellm.model_cost):
        if limiter.get_rate_limit_for_model(model) is limiter.default_rate_limit:
            typer.echo(model)


def main() -> None:  # pragma: no cover - CLI entry point
    app()


if __name__ == "__main__":  # pragma: no cover - CLI runner
    main()
