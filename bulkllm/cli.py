from __future__ import annotations

from datetime import date

import litellm
import typer

from bulkllm.llm_configs import create_model_configs
from bulkllm.model_registration.canonical import _canonical_model_name, get_canonical_models
from bulkllm.model_registration.main import register_models
from bulkllm.rate_limiter import RateLimiter


def _tabulate(rows: list[list[str]], headers: list[str]) -> str:
    """Return a simple table for CLI output."""
    columns = list(zip(*([headers, *rows]))) if rows else [headers]
    widths = [max(len(str(c)) for c in col) for col in columns]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    divider = "-+-".join("-" * w for w in widths)
    lines = [fmt.format(*headers), divider]
    for row in rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)


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


@app.command("list-canonical-models")
def list_canonical_models() -> None:
    """List canonical chat models with release dates."""
    rows = get_canonical_models()
    table = _tabulate(rows, headers=["model", "mode", "release_date", "scraped_date"])
    typer.echo(table)


@app.command("list-configs")
def list_configs(
    sort_by: str = typer.Option(
        "slug",
        "--sort-by",
        "-s",
        help="Sort by slug, company or release-date",
        case_sensitive=False,
    ),
) -> None:
    """List LLM configurations."""
    sort_key = sort_by.replace("-", "_").lower()
    key_funcs = {
        "slug": lambda c: c.slug,
        "company": lambda c: c.company_name.lower(),
        "release_date": lambda c: c.release_date or date.min,
    }
    if sort_key not in key_funcs:
        raise typer.BadParameter("Invalid sort option")

    configs = sorted(create_model_configs(), key=key_funcs[sort_key])

    rows = [
        [
            cfg.slug,
            cfg.company_name,
            cfg.display_name,
            cfg.release_date.isoformat() if cfg.release_date else "",
            cfg.is_deprecated.isoformat() if isinstance(cfg.is_deprecated, date) else "",
        ]
        for cfg in configs
    ]

    table = _tabulate(
        rows,
        headers=["slug", "company", "display_name", "release_date", "deprecation_date"],
    )
    typer.echo(table)


@app.command("list-missing-rate-limits")
def list_missing_rate_limits() -> None:
    """List models without a configured rate limit."""
    register_models()
    limiter = RateLimiter()
    for model in sorted(litellm.model_cost):
        if limiter.get_rate_limit_for_model(model) is limiter.default_rate_limit:
            typer.echo(model)


@app.command("missing-rate-limits")
def missing_rate_limits() -> None:
    """List config slugs without a configured rate limit."""
    register_models()
    limiter = RateLimiter()
    for cfg in create_model_configs():
        if limiter.get_rate_limit_for_model(cfg.litellm_model_name) is limiter.default_rate_limit:
            typer.echo(cfg.slug)


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
