import importlib
import sys
import types

import pytest

from bulkllm.schema import LLMConfig


def _import_llm_configs(monkeypatch: pytest.MonkeyPatch):
    dummy_litellm = types.SimpleNamespace(
        get_max_tokens=lambda model: 8192,
        cost_per_token=lambda model, prompt_tokens, completion_tokens: (0.0, 0.0),
        register_model=lambda *a, **k: None,
        get_model_info=lambda *a, **k: None,
        model_cost={},
    )
    monkeypatch.setitem(sys.modules, "litellm", dummy_litellm)
    if "bulkllm.llm_configs" in sys.modules:
        del sys.modules["bulkllm.llm_configs"]
    return importlib.import_module("bulkllm.llm_configs")


@pytest.fixture
def stub_configs(monkeypatch: pytest.MonkeyPatch):
    llm_configs = _import_llm_configs(monkeypatch)
    cfg1 = LLMConfig(
        slug="cfg1",
        display_name="Cfg1",
        company_name="ACME",
        litellm_model_name="acme/cfg1",
        llm_family="cfg1",
        temperature=0.0,
        max_tokens=100,
    )
    cfg2 = LLMConfig(
        slug="cfg2",
        display_name="Cfg2",
        company_name="ACME",
        litellm_model_name="acme/cfg2",
        llm_family="cfg2",
        temperature=0.0,
        max_tokens=100,
    )
    all_cfgs = [cfg1, cfg2]
    cheap_cfgs = [cfg1]
    monkeypatch.setattr(
        llm_configs, "create_model_configs", lambda system_prompt="You are a helpful AI assistant.": all_cfgs
    )
    monkeypatch.setattr(llm_configs, "cheap_model_configs", lambda: cheap_cfgs)
    return llm_configs, cheap_cfgs, all_cfgs


def test_model_resolver_cheap(stub_configs):
    llm_configs, cheap_cfgs, _ = stub_configs
    result = llm_configs.model_resolver(["cheap"])
    assert result == cheap_cfgs
    assert all(isinstance(c, LLMConfig) for c in result)


def test_model_resolver_default(stub_configs):
    llm_configs, cheap_cfgs, _ = stub_configs
    result = llm_configs.model_resolver(["default"])
    assert result == cheap_cfgs


def test_model_resolver_all(stub_configs):
    llm_configs, _, all_cfgs = stub_configs
    result = llm_configs.model_resolver(["all"])
    assert result == all_cfgs
