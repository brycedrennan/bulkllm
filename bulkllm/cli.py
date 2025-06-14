from __future__ import annotations

import datetime

import litellm
import typer

from bulkllm.llm_configs import create_model_configs
from bulkllm.model_registration import (
    anthropic,
    gemini,
    mistral,
    openai,
    xai,
)
from bulkllm.model_registration.canonical import _canonical_model_name
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
    register_models()

    scraped_models: dict[str, dict] = {}
    providers = [
        openai.get_openai_models,
        anthropic.get_anthropic_models,
        gemini.get_gemini_models,
        mistral.get_mistral_models,
        xai.get_xai_models,
        # openrouter.get_openrouter_models,
    ]
    for get_models in providers:
        scraped_models.update(get_models())

    def _is_xai_fast(name: str) -> bool:
        return name.startswith("xai/") and "fast" in name

    # Keep only chat models
    scraped_models = {name: info for name, info in scraped_models.items() if info.get("mode") == "chat"}

    alias_names = {
        c
        for a in openai.get_openai_aliases()
        if (c := _canonical_model_name(a, {"litellm_provider": "openai", "mode": "chat"}))
    }
    alias_names |= {
        c
        for a in mistral.get_mistral_aliases()
        if (c := _canonical_model_name(a, {"litellm_provider": "mistral", "mode": "chat"}))
    }

    canonical_scraped: dict[str, dict] = {}
    for model, model_info in scraped_models.items():
        canonical = _canonical_model_name(model, model_info)
        if canonical is None or canonical in alias_names or _is_xai_fast(canonical):
            continue
        canonical_scraped.setdefault(canonical, model_info)

    def _dedupe_gemini_by_version(models: dict[str, dict]) -> dict[str, dict]:
        seen: set[str] = set()
        deduped: dict[str, dict] = {}
        for name in sorted(models):
            info = models[name]
            if info.get("litellm_provider") == "gemini":
                version = info.get("version")
                if version and version in seen:
                    continue
                if version:
                    seen.add(version)
            deduped[name] = info
        return deduped

    canonical_scraped = _dedupe_gemini_by_version(canonical_scraped)

    canonical_registered: dict[str, dict] = {}
    for model, model_info in litellm.model_cost.items():
        if model_info.get("mode") != "chat":
            continue
        canonical = _canonical_model_name(model, model_info)
        if canonical is None or canonical in alias_names or _is_xai_fast(canonical):
            continue
        canonical_registered.setdefault(canonical, model_info)

    # Map canonical model name to release date from LLMConfig
    release_dates = {}
    for cfg in create_model_configs():
        if not cfg.release_date:
            continue
        provider = cfg.litellm_model_name.split("/", 1)[0]
        canonical = _canonical_model_name(
            cfg.litellm_model_name,
            {"mode": "chat", "litellm_provider": provider},
        )
        if canonical and canonical not in release_dates:
            release_dates[canonical] = cfg.release_date.isoformat()

    created_dates = {}
    for name, info in canonical_scraped.items():
        created = info.get("created_at", info.get("created"))
        if created is None:
            continue
        try:
            if isinstance(created, str):
                dt = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
            else:
                dt = datetime.datetime.fromtimestamp(
                    int(created), tz=datetime.timezone.utc
                )
            created_dates[name] = dt.date().isoformat()
        except (ValueError, OSError, OverflowError):
            created_dates[name] = str(created)

    rows = []
    for name in sorted(canonical_scraped):
        info = canonical_registered.get(name, canonical_scraped[name])
        release_date = release_dates.get(name, "")
        created = created_dates.get(name, "")
        rows.append([name, str(info.get("mode", "")), release_date, created])

    table = _tabulate(rows, headers=["model", "mode", "release_date", "scraped_date"])
    typer.echo(table)


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
