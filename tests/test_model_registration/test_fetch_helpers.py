import json
from pathlib import Path
from typing import Any

import requests

from bulkllm.model_registration import anthropic, gemini, openai, openrouter, utils


class DummyResponse:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._data


def _patch_get(monkeypatch, data: dict[str, Any]) -> None:
    def fake_get(*args, **kwargs):
        return DummyResponse(data)

    monkeypatch.setattr(requests, "get", fake_get)


def _patch_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(utils, "USER_CACHE_DIR", tmp_path)


def _load(provider: str, cache_dir: Path) -> dict[str, Any]:
    with open(cache_dir / f"{provider}.json") as f:
        return json.load(f)


def test_openai_fetch_sort(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path)
    sample = {"data": [{"id": "b", "created": 2}, {"id": "a", "created": 1}]}
    _patch_get(monkeypatch, sample)
    data = openai.fetch_openai_data()
    assert [m["id"] for m in data["data"]] == ["a", "b"]
    saved = _load("openai", tmp_path)
    assert [m["id"] for m in saved["data"]] == ["a", "b"]


def test_anthropic_fetch_sort(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path)
    sample = {
        "data": [
            {"id": "b", "created_at": "2021"},
            {"id": "a", "created_at": "2020"},
        ]
    }
    _patch_get(monkeypatch, sample)
    data = anthropic.fetch_anthropic_data()
    assert [m["id"] for m in data["data"]] == ["a", "b"]
    saved = _load("anthropic", tmp_path)
    assert [m["id"] for m in saved["data"]] == ["a", "b"]


def test_gemini_fetch_sort(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path)
    sample = {"models": [{"name": "b"}, {"name": "a"}]}
    _patch_get(monkeypatch, sample)
    data = gemini.fetch_gemini_data()
    assert [m["name"] for m in data["models"]] == ["a", "b"]
    saved = _load("gemini", tmp_path)
    assert [m["name"] for m in saved["models"]] == ["a", "b"]


def test_openrouter_fetch_sort(monkeypatch, tmp_path):
    _patch_cache(monkeypatch, tmp_path)
    sample = {"data": [{"id": "b", "created": 2}, {"id": "a", "created": 1}]}
    _patch_get(monkeypatch, sample)
    data = openrouter.fetch_openrouter_data()
    assert [m["id"] for m in data["data"]] == ["a", "b"]
    saved = _load("openrouter", tmp_path)
    assert [m["id"] for m in saved["data"]] == ["a", "b"]
