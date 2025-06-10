import importlib.util
import json
from pathlib import Path
from typing import Any, cast

from bulkllm.model_registration import anthropic, gemini, openai, openrouter

spec = importlib.util.spec_from_file_location(
    "update_model_cache",
    Path(__file__).resolve().parents[2] / "scripts" / "update_model_cache.py",
)
assert spec
assert spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
um = cast("Any", module)


def test_update_script_uses_helpers(monkeypatch, tmp_path):
    monkeypatch.setattr(um, "get_data_file", lambda name: tmp_path / f"{name}.json")

    called = {}

    def make_fetch(name: str):
        def _fetch() -> dict[str, Any]:
            called[name] = True
            return {"name": name}

        return _fetch

    monkeypatch.setattr(openai, "fetch_openai_data", make_fetch("openai"))
    monkeypatch.setattr(anthropic, "fetch_anthropic_data", make_fetch("anthropic"))
    monkeypatch.setattr(gemini, "fetch_gemini_data", make_fetch("gemini"))
    monkeypatch.setattr(openrouter, "fetch_openrouter_data", make_fetch("openrouter"))

    um.main()

    for prov in ["openai", "anthropic", "gemini", "openrouter"]:
        assert called.get(prov)
        with open(tmp_path / f"{prov}.json") as f:
            assert json.load(f) == {"name": prov}
