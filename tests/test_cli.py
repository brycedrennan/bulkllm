import datetime

from typer.testing import CliRunner

from bulkllm.schema import LLMConfig

LLMConfig.model_rebuild(_types_namespace={"datetime": datetime})

from bulkllm.cli import app  # noqa: E402


def test_list_models(monkeypatch):
    import litellm

    # Ensure a clean slate
    monkeypatch.setattr(litellm, "model_cost", {})

    def fake_register_models() -> None:
        litellm.model_cost["fake/model"] = {
            "litellm_provider": "openai",
            "mode": "chat",
        }

    monkeypatch.setattr("bulkllm.cli.register_models", fake_register_models)
    monkeypatch.setattr("bulkllm.model_registration.main.register_models", fake_register_models)

    runner = CliRunner()
    result = runner.invoke(app, ["list-models"])
    assert result.exit_code == 0
    assert "fake/model" in result.output


def test_list_missing_rate_limits(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "model_cost", {})

    def fake_register_models() -> None:
        litellm.model_cost["limited/model"] = {
            "litellm_provider": "openai",
            "mode": "chat",
        }
        litellm.model_cost["unlimited/model"] = {
            "litellm_provider": "openai",
            "mode": "chat",
        }

    class DummyRateLimiter:
        def __init__(self) -> None:
            self.default_rate_limit = object()

        def get_rate_limit_for_model(self, model: str) -> object:
            if model == "limited/model":
                return object()
            return self.default_rate_limit

    monkeypatch.setattr("bulkllm.cli.register_models", fake_register_models)
    monkeypatch.setattr("bulkllm.model_registration.main.register_models", fake_register_models)
    monkeypatch.setattr("bulkllm.cli.RateLimiter", DummyRateLimiter)

    runner = CliRunner()
    result = runner.invoke(app, ["list-missing-rate-limits"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert "unlimited/model" in lines
    assert "limited/model" not in lines


def test_list_missing_model_configs(monkeypatch):
    from types import SimpleNamespace

    import litellm

    monkeypatch.setattr(litellm, "model_cost", {})

    def fake_register_models() -> None:
        litellm.model_cost["configured/model"] = {
            "litellm_provider": "openai",
            "mode": "chat",
        }
        litellm.model_cost["unconfigured/model"] = {
            "litellm_provider": "openai",
            "mode": "chat",
        }

    monkeypatch.setattr("bulkllm.cli.register_models", fake_register_models)
    monkeypatch.setattr(
        "bulkllm.model_registration.main.register_models",
        fake_register_models,
    )
    monkeypatch.setattr(
        "bulkllm.cli.create_model_configs",
        lambda: [SimpleNamespace(litellm_model_name="configured/model")],
    )

    runner = CliRunner()
    result = runner.invoke(app, ["list-missing-model-configs"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert "unconfigured/model" in lines
    assert "configured/model" not in lines


def test_list_unique_models(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "model_cost", {})

    def fake_register_models() -> None:
        litellm.model_cost["anthropic/claude-3"] = {
            "litellm_provider": "anthropic",
            "mode": "chat",
        }
        litellm.model_cost["openrouter/anthropic/claude-3"] = {
            "litellm_provider": "openrouter",
            "mode": "chat",
        }
        litellm.model_cost["bedrock/anthropic.claude-3"] = {
            "litellm_provider": "bedrock",
            "mode": "chat",
        }

    monkeypatch.setattr("bulkllm.cli.register_models", fake_register_models)
    monkeypatch.setattr("bulkllm.model_registration.main.register_models", fake_register_models)

    runner = CliRunner()
    result = runner.invoke(app, ["list-unique-models"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert lines.count("anthropic/claude-3") == 1
    assert "openrouter/anthropic/claude-3" not in lines
    assert "bedrock/anthropic.claude-3" not in lines


def test_list_canonical_models(monkeypatch):
    monkeypatch.setattr(
        "bulkllm.cli.openai.get_openai_models",
        lambda: {"openai/gpt": {"litellm_provider": "openai", "mode": "chat"}},
    )
    monkeypatch.setattr(
        "bulkllm.cli.anthropic.get_anthropic_models",
        lambda: {"anthropic/claude": {"litellm_provider": "anthropic", "mode": "chat"}},
    )
    monkeypatch.setattr(
        "bulkllm.cli.gemini.get_gemini_models",
        lambda: {"gemini/flash": {"litellm_provider": "gemini", "mode": "chat"}},
    )
    monkeypatch.setattr(
        "bulkllm.cli.mistral.get_mistral_models",
        lambda: {"mistral/small": {"litellm_provider": "mistral", "mode": "chat"}},
    )
    monkeypatch.setattr(
        "bulkllm.cli.openrouter.get_openrouter_models",
        lambda: {"openrouter/google/gamma": {"litellm_provider": "openrouter", "mode": "chat"}},
    )

    runner = CliRunner()
    result = runner.invoke(app, ["list-canonical-models"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert "openai/gpt" in lines
    assert "anthropic/claude" in lines
    assert "gemini/flash" in lines
    assert "mistral/small" in lines
    # openrouter canonicalisation drops the prefix
    assert "google/gamma" in lines
    assert "openrouter/google/gamma" not in lines
