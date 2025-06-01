import litellm

from bulkllm.model_registration.main import register_models

primary_providers = {"openai", "anthropic", "gemini", "xai", "qwen", "deepseek", "mistral"}


def _canonical_model_name(name: str, model_info) -> str | None:
    """Return canonical name for ``name`` dropping provider wrappers."""
    name = name.replace("/x-ai/", "/xai/")
    if name.startswith("x-ai/"):
        name = name.replace("x-ai/", "xai/")

    if model_info.get("mode") in ("image_generation", "embedding"):
        return None

    provider = model_info.get("litellm_provider")
    if provider == "text-completion-openai":
        return None

    if provider in primary_providers:
        if "ft:" in name:
            return None
        if name.startswith(f"{provider}/"):
            return name
        if "/" not in name:
            return f"{provider}/{name}"
        msg = f"Invalid model name: {name}"
        raise ValueError(msg)

    for p in primary_providers:
        if p in name:
            return None

    if name.startswith("openrouter/"):
        name = name[len("openrouter/") :]
        return name
    if name.startswith("bedrock/"):
        after = name[len("bedrock/") :]
        if "nova" not in after:
            return None
        return after

    return name


def canonical_models():
    """Return list of canonical model names."""
    unique: set[str] = set()
    for model, model_info in text_models().items():
        canonical = _canonical_model_name(model, model_info)
        if canonical is None:
            continue
        unique.add(canonical)

    return sorted(unique)


def model_modes():
    register_models()
    modes: set[str] = set()
    for model, model_info in litellm.model_cost.items():
        modes.add(str(model_info.get("mode")))

    return sorted(modes)


def model_providers():
    register_models()
    providers: set[str] = set()
    for model, model_info in litellm.model_cost.items():
        providers.add(str(model_info.get("litellm_provider")))

    return sorted(providers)


def text_models():
    register_models()
    models = {}
    for model, model_info in litellm.model_cost.items():
        if model == "sample_spec":
            continue
        if "audio" in model:
            continue
        if model_info.get("mode") in ("chat", "completion"):
            if "/" not in model:
                model = f"{model_info.get('litellm_provider')}/{model}"
            models[model] = model_info

    return models


def _primary_provider_model_names():
    model_names = set()

    for model, model_info in text_models().items():
        if model_info.get("litellm_provider") in primary_providers:
            provider_name, model_name = model.split("/", 1)
            if model_name in model_names:
                print(f"ERROR: Duplicate model name: {model_name}")
            model_names.add(model_name)
    return sorted(model_names)


if __name__ == "__main__":
    print(model_modes())
    print(model_providers())
    print(text_models().keys())
    print(_primary_provider_model_names())
    print(f"Total models: {len(litellm.model_cost)}")
    print(f"Total text models: {len(text_models())}")
    print(f"Total primary provider models: {len(_primary_provider_model_names())}")
