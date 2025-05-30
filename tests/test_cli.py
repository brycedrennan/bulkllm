from typer.testing import CliRunner

from bulkllm.cli import app


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


def test_list_missing_configs(monkeypatch):
    import datetime

    import litellm

    from bulkllm.schema import LLMConfig

    LLMConfig.model_rebuild(_types_namespace={"datetime": datetime})

    monkeypatch.setattr(litellm, "model_cost", {})

    def fake_register_models() -> None:
        litellm.model_cost["has/config"] = {"litellm_provider": "openai", "mode": "chat"}
        litellm.model_cost["no/config"] = {"litellm_provider": "openai", "mode": "chat"}

    monkeypatch.setattr("bulkllm.cli.register_models", fake_register_models)
    monkeypatch.setattr("bulkllm.model_registration.main.register_models", fake_register_models)

    def fake_create_model_configs(system_prompt: str | None = "You are a helpful AI assistant."):
        return [
            LLMConfig(
                slug="cfg",
                display_name="Cfg",
                company_name="ACME",
                litellm_model_name="has/config",
                temperature=0.0,
                max_tokens=100,
            )
        ]

    monkeypatch.setattr("bulkllm.llm_configs.create_model_configs", fake_create_model_configs)

    runner = CliRunner()
    result = runner.invoke(app, ["list-missing-configs"])
    assert result.exit_code == 0
    assert "no/config" in result.output
    assert "has/config" not in result.output

    
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

