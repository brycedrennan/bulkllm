import datetime

from bulkllm.schema import LLMConfig

LLMConfig.model_rebuild(_types_namespace={"datetime": datetime})


def test_llmconfig_md5_and_kwargs():
    cfg1 = LLMConfig(
        slug="s1",
        display_name="S1",
        company_name="ACME",
        litellm_model_name="model-1",
        llm_family="s1",
        temperature=0.1,
        max_tokens=100,
        system_prompt="hello",
    )
    cfg2 = cfg1.model_copy(update={"temperature": 0.2})

    assert cfg1.md5_hash != cfg2.md5_hash

    kwargs = cfg1.completion_kwargs()
    assert kwargs["model"] == "model-1"
    assert kwargs["temperature"] == 0.1
    assert kwargs["max_tokens"] == 100
    assert kwargs["stream"] is False


def test_llmconfig_verbosity_hash_and_kwargs_passthrough():
    base = LLMConfig(
        slug="s2",
        display_name="S2",
        company_name="ACME",
        litellm_model_name="model-2",
        llm_family="s2",
        temperature=0.0,
        max_tokens=10,
    )

    # Adding verbosity should change hash; absence should preserve existing hashes
    with_verbosity = base.model_copy(update={"verbosity": "high"})
    assert base.md5_hash != with_verbosity.md5_hash

    # When not set, verbosity should not appear in kwargs
    no_verb_kwargs = base.completion_kwargs()
    assert "verbosity" not in no_verb_kwargs
    assert "extra_body" not in no_verb_kwargs or "verbosity" not in no_verb_kwargs.get("extra_body", {})

    # When set, verbosity should be included via extra_body for generic models
    verb_kwargs = with_verbosity.completion_kwargs()
    assert verb_kwargs.get("extra_body", {}).get("verbosity") == "high"
    # Not an openai/ model, should not add top-level param or allow list
    assert "verbosity" not in verb_kwargs
    assert "allowed_openai_params" not in verb_kwargs

    # For openai/ models, it should add top-level and allow-list
    openai_cfg = base.model_copy(update={"litellm_model_name": "openai/gpt-5", "verbosity": 2})
    ok = openai_cfg.completion_kwargs()
    assert ok.get("verbosity") == 2
    assert "verbosity" in ok.get("allowed_openai_params", [])
    assert ok.get("extra_body", {}).get("verbosity") == 2
