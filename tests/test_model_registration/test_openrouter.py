from unittest.mock import patch

from bulkllm.model_registration.openrouter import get_openrouter_models


def test_filter_free_versions_when_non_suffixed_exists():
    """Test that :free versions are filtered out when non-suffixed versions exist."""

    # Clear the cache to ensure fresh data
    get_openrouter_models.cache_clear()

    # Mock OpenRouter API response with both :free and non-suffixed versions
    mock_data = {
        "data": [
            {
                "id": "model-a",  # Non-suffixed version
                "pricing": {"prompt": 0.0001, "completion": 0.0002},
                "context_length": 4096,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            },
            {
                "id": "model-a:free",  # Free version - should be filtered out
                "pricing": {"prompt": 0.0, "completion": 0.0},
                "context_length": 4096,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            },
            {
                "id": "model-b:free",  # Only free version exists - should be kept
                "pricing": {"prompt": 0.0, "completion": 0.0},
                "context_length": 4096,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            },
            {
                "id": "model-c",  # Only non-suffixed version exists - should be kept
                "pricing": {"prompt": 0.0001, "completion": 0.0002},
                "context_length": 4096,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            },
        ]
    }

    with patch("bulkllm.model_registration.openrouter.load_cached_provider_data") as mock_load:
        mock_load.return_value = mock_data

        models = get_openrouter_models(use_cached=True)

        # Should have model-a (not model-a:free), model-b:free (only version), and model-c
        expected_models = {"openrouter/model-a", "openrouter/model-b:free", "openrouter/model-c"}

        actual_models = set(models.keys())

        # Check that we have the expected models
        assert expected_models.issubset(actual_models), (
            f"Expected models {expected_models} not found in {actual_models}"
        )

        # Check that the :free version of model-a was filtered out
        assert "openrouter/model-a:free" not in actual_models, "model-a:free should have been filtered out"

        # Check that we kept the non-suffixed version of model-a
        assert "openrouter/model-a" in actual_models, "model-a should have been kept"

        # Check that model-b:free was kept (no non-suffixed version exists)
        assert "openrouter/model-b:free" in actual_models, "model-b:free should have been kept"


def test_only_free_versions_are_kept():
    """Test that :free versions are kept when no non-suffixed versions exist."""

    # Clear the cache to ensure fresh data
    get_openrouter_models.cache_clear()

    mock_data = {
        "data": [
            {
                "id": "model-only-free:free",
                "pricing": {"prompt": 0.0, "completion": 0.0},
                "context_length": 4096,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            },
        ]
    }

    with patch("bulkllm.model_registration.openrouter.load_cached_provider_data") as mock_load:
        mock_load.return_value = mock_data

        models = get_openrouter_models(use_cached=True)

        # Should include the :free version since no non-suffixed version exists
        assert "openrouter/model-only-free:free" in models
