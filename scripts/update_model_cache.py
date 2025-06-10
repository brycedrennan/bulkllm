from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from bulkllm.model_registration import anthropic, gemini, openai, openrouter
from bulkllm.model_registration.utils import get_data_file

app = typer.Typer(add_completion=False, no_args_is_help=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    typer.echo(f"Updated {path}")


@app.command()
def main() -> None:
    providers: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("openai", openai.fetch_openai_data),
        ("anthropic", anthropic.fetch_anthropic_data),
        ("gemini", gemini.fetch_gemini_data),
        ("openrouter", openrouter.fetch_openrouter_data),
    ]
    for name, fetcher in providers:
        data = fetcher()
        write_json(get_data_file(name), data)


if __name__ == "__main__":  # pragma: no cover - manual script
    typer.run(main)
