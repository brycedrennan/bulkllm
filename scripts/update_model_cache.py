from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import requests
import typer

from bulkllm.model_registration.utils import get_data_file

if TYPE_CHECKING:
    from pathlib import Path

app = typer.Typer(add_completion=False, no_args_is_help=True)


CACHE_MAX_AGE = timedelta(hours=0)


def sort_openai_data(data: dict[str, Any]) -> dict[str, Any]:
    """Return ``data`` with models sorted by creation time when available."""
    models = data.get("data", [])
    data["data"] = sorted(models, key=lambda m: m.get("created", 0))
    return data


def sort_xai_data(data: dict[str, Any]) -> dict[str, Any]:
    """Return ``data`` with models sorted by creation time."""
    return sort_openai_data(data)


def sort_anthropic_data(data: dict[str, Any]) -> dict[str, Any]:
    """Return ``data`` with models sorted by creation time when available."""
    models = data.get("data", [])
    data["data"] = sorted(models, key=lambda m: m.get("created_at", ""))
    return data


def sort_gemini_data(data: dict[str, Any]) -> dict[str, Any]:
    """Return ``data`` with models sorted by name."""
    models = data.get("models", [])
    data["models"] = sorted(models, key=lambda m: m.get("name", ""))
    return data


def sort_openrouter_data(data: dict[str, Any]) -> dict[str, Any]:
    """Return ``data`` with models sorted by creation time."""
    models = data.get("data", [])
    data["data"] = sorted(models, key=lambda m: m.get("created", 0))
    return data


SORTERS: dict[str, callable[[dict[str, Any]], dict[str, Any]]] = {
    "openai": sort_openai_data,
    "xai": sort_xai_data,
    "anthropic": sort_anthropic_data,
    "gemini": sort_gemini_data,
    "openrouter": sort_openrouter_data,
}


def needs_update(path: Path) -> bool:
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return datetime.now(tz=UTC) - mtime > CACHE_MAX_AGE


def fetch(url: str, *, headers: dict[str, str] | None = None, params: dict[str, str] | None = None) -> dict:
    resp = requests.get(url, headers=headers or {}, params=params)
    resp.raise_for_status()
    return resp.json()


def write_json(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` after applying provider-specific sorting."""
    provider = path.stem
    sorter = SORTERS.get(provider)
    if sorter:
        data = sorter(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    typer.echo(f"Updated {path}")


@app.command()
def main(force: bool = False) -> None:
    """Update cached provider responses if older than one hour."""
    # OpenAI
    openai_path = get_data_file("openai")
    if force or needs_update(openai_path):
        headers = {}
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = fetch("https://api.openai.com/v1/models", headers=headers)
        write_json(openai_path, data)

    # XAI
    xai_path = get_data_file("xai")
    if force or needs_update(xai_path):
        headers = {}
        api_key = os.getenv("XAI_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            typer.echo("XAI_API_KEY is not set, skipping update")
        data = fetch("https://api.x.ai/v1/language-models", headers=headers)
        write_json(xai_path, data)

    # Anthropic
    anthropic_path = get_data_file("anthropic")
    if force or needs_update(anthropic_path):
        headers = {
            "x-api-key": os.getenv("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        data = fetch("https://api.anthropic.com/v1/models", headers=headers)
        write_json(anthropic_path, data)

    # Gemini
    gemini_path = get_data_file("gemini")
    if force or needs_update(gemini_path):
        api_key = os.getenv("GEMINI_API_KEY", "")
        params = {"key": api_key} if api_key else None
        data = fetch("https://generativelanguage.googleapis.com/v1beta/models", params=params)
        write_json(gemini_path, data)

    # OpenRouter
    openrouter_path = get_data_file("openrouter")
    if force or needs_update(openrouter_path):
        data = fetch("https://openrouter.ai/api/v1/models")
        write_json(openrouter_path, data)


if __name__ == "__main__":  # pragma: no cover - manual script
    typer.run(main)
