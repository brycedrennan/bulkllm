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
