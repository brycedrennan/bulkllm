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
    import litellm

    monkeypatch.setattr(litellm, "model_cost", {})

    monkeypatch.setattr(
        "bulkllm.cli.openai.get_openai_models",
        lambda: {
            "openai/gpt": {"litellm_provider": "openai", "mode": "chat"},
            "openai/text": {"litellm_provider": "openai", "mode": "completion"},
        },
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
        lambda: {
            "openrouter/google/gamma": {
                "litellm_provider": "openrouter",
                "mode": "chat",
            }
        },
    )

    def fake_register_models() -> None:
        litellm.model_cost["openai/gpt"] = {"litellm_provider": "openai", "mode": "chat"}
        litellm.model_cost["openai/text"] = {
            "litellm_provider": "openai",
            "mode": "completion",
        }
        litellm.model_cost["anthropic/claude"] = {"litellm_provider": "anthropic", "mode": "chat"}
        litellm.model_cost["gemini/flash"] = {"litellm_provider": "gemini", "mode": "chat"}
        litellm.model_cost["mistral/small"] = {"litellm_provider": "mistral", "mode": "chat"}
        litellm.model_cost["openrouter/google/gamma"] = {
            "litellm_provider": "openrouter",
            "mode": "chat",
        }

    monkeypatch.setattr("bulkllm.cli.register_models", fake_register_models)
    monkeypatch.setattr("bulkllm.model_registration.main.register_models", fake_register_models)
    monkeypatch.setattr(
        "bulkllm.cli.create_model_configs",
        lambda: [
            LLMConfig(
                slug="gpt",
                display_name="GPT",
                company_name="openai",
                litellm_model_name="openai/gpt",
                llm_family="gpt",
                temperature=1,
                max_tokens=1,
                release_date=datetime.date(2025, 1, 1),
            ),
            LLMConfig(
                slug="claude",
                display_name="Claude",
                company_name="anthropic",
                litellm_model_name="anthropic/claude",
                llm_family="claude",
                temperature=1,
                max_tokens=1,
                release_date=datetime.date(2025, 2, 2),
            ),
            LLMConfig(
                slug="flash",
                display_name="Flash",
                company_name="gemini",
                litellm_model_name="gemini/flash",
                llm_family="flash",
                temperature=1,
                max_tokens=1,
                release_date=datetime.date(2025, 3, 3),
            ),
            LLMConfig(
                slug="small",
                display_name="Small",
                company_name="mistral",
                litellm_model_name="mistral/small",
                llm_family="small",
                temperature=1,
                max_tokens=1,
                release_date=datetime.date(2025, 4, 4),
            ),
            LLMConfig(
                slug="gamma",
                display_name="Gamma",
                company_name="google",
                litellm_model_name="openrouter/google/gamma",
                llm_family="gamma",
                temperature=1,
                max_tokens=1,
                release_date=datetime.date(2025, 5, 5),
            ),
        ],
    )

    runner = CliRunner()
    result = runner.invoke(app, ["list-canonical-models"])

    assert result.exit_code == 0
    lines = [line.strip() for line in result.output.splitlines() if line.strip()]
    rows = [line.split("|") for line in lines[2:]]  # skip header and divider
    table = {cells[0].strip(): (cells[1].strip(), cells[2].strip()) for cells in rows}

    assert table["openai/gpt"] == ("chat", "2025-01-01")
    assert table["anthropic/claude"] == ("chat", "2025-02-02")
    assert table["gemini/flash"] == ("chat", "2025-03-03")
    assert table["mistral/small"] == ("chat", "2025-04-04")
    # openrouter canonicalisation drops the prefix
    assert table["google/gamma"] == ("chat", "2025-05-05")
    assert "openrouter/google/gamma" not in table
    assert "openai/text" not in table


def test_list_canonical_models_drops_aliases(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "model_cost", {})

    monkeypatch.setattr(
        "bulkllm.cli.openai.get_openai_models",
        lambda: {
            "openai/base": {"litellm_provider": "openai", "mode": "chat"},
            "openai/alias": {"litellm_provider": "openai", "mode": "chat"},
        },
    )
    monkeypatch.setattr(
        "bulkllm.cli.anthropic.get_anthropic_models",
        dict,
    )
    monkeypatch.setattr(
        "bulkllm.cli.gemini.get_gemini_models",
        dict,
    )
    monkeypatch.setattr(
        "bulkllm.cli.mistral.get_mistral_models",
        dict,
    )
    monkeypatch.setattr(
        "bulkllm.cli.openrouter.get_openrouter_models",
        dict,
    )
    monkeypatch.setattr(
        "bulkllm.cli.openai.get_openai_aliases",
        lambda: {"openai/alias"},
    )

    def fake_register_models() -> None:
        litellm.model_cost["openai/base"] = {"litellm_provider": "openai", "mode": "chat"}
        litellm.model_cost["openai/alias"] = {"litellm_provider": "openai", "mode": "chat"}

    monkeypatch.setattr("bulkllm.cli.register_models", fake_register_models)
    monkeypatch.setattr("bulkllm.model_registration.main.register_models", fake_register_models)
    monkeypatch.setattr(
        "bulkllm.cli.create_model_configs",
        lambda: [
            LLMConfig(
                slug="base",
                display_name="Base",
                company_name="openai",
                litellm_model_name="openai/base",
                llm_family="base",
                temperature=1,
                max_tokens=1,
                release_date=datetime.date(2025, 1, 1),
            ),
            LLMConfig(
                slug="alias",
                display_name="Alias",
                company_name="openai",
                litellm_model_name="openai/alias",
                llm_family="alias",
                temperature=1,
                max_tokens=1,
                release_date=datetime.date(2025, 2, 2),
            ),
        ],
    )

    runner = CliRunner()
    result = runner.invoke(app, ["list-canonical-models"])

    assert result.exit_code == 0
    lines = [line.strip() for line in result.output.splitlines() if line.strip()]
    rows = [line.split("|") for line in lines[2:]]  # skip header and divider
    table = {cells[0].strip(): cells[1].strip() for cells in rows}

    assert "openai/base" in table
    assert "openai/alias" not in table
