from collections import defaultdict

import litellm
from litellm.caching.caching import Cache

from bulkllm.llm import initialize_litellm
from bulkllm.llm_configs import model_resolver


def test_litellm_cache_bug(monkeypatch):
    """there was a bug in litellm where the cache key was not being hashed correctly and was not sufficinetly unique"""
    initialize_litellm()

    assert litellm.cache is not None

    monkeypatch.setattr(Cache, "_get_hashed_cache_key", lambda k: k)
    key_counts = defaultdict(int)

    for llm_config in model_resolver(["all"]):
        key = litellm.cache.get_cache_key(**llm_config.completion_kwargs())
        key_counts[key] += 1

    too_many_keys = [key for key, count in key_counts.items() if count > 1]
    assert not too_many_keys, f"Duplicate configs found: {too_many_keys}"


if __name__ == "__main__":
    test_litellm_cache_bug()
