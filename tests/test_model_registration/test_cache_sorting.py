import importlib.util
import json
from pathlib import Path
from typing import Any, cast

spec = importlib.util.spec_from_file_location(
    "update_model_cache",
    Path(__file__).resolve().parents[2] / "scripts" / "update_model_cache.py",
)
assert spec
assert spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
um = cast("Any", module)


def test_sort_openai(tmp_path):
    data = {"data": [{"id": "b", "created": 2}, {"id": "a", "created": 1}]}
    path = tmp_path / "openai.json"
    um.write_json(path, data)
    saved = json.loads(path.read_text())
    assert [m["id"] for m in saved["data"]] == ["a", "b"]


def test_sort_anthropic(tmp_path):
    data = {"data": [{"id": "b", "created_at": "2021"}, {"id": "a", "created_at": "2020"}]}
    path = tmp_path / "anthropic.json"
    um.write_json(path, data)
    saved = json.loads(path.read_text())
    assert [m["id"] for m in saved["data"]] == ["a", "b"]


def test_sort_gemini(tmp_path):
    data = {"models": [{"name": "b"}, {"name": "a"}]}
    path = tmp_path / "gemini.json"
    um.write_json(path, data)
    saved = json.loads(path.read_text())
    assert [m["name"] for m in saved["models"]] == ["a", "b"]


def test_sort_openrouter(tmp_path):
    data = {"data": [{"id": "b", "created": 2}, {"id": "a", "created": 1}]}
    path = tmp_path / "openrouter.json"
    um.write_json(path, data)
    saved = json.loads(path.read_text())
    assert [m["id"] for m in saved["data"]] == ["a", "b"]
