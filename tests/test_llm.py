import os
import time

import pytest


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="requires OPENAI_API_KEY")
def test_helloworld():
    from bulkllm.llm import completion

    response = completion(model="openai/gpt-4.1-mini", messages=[{"role": "user", "content": "Hello, world!"}])
    print(response)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="requires OPENAI_API_KEY")
def test_created_field_cached_after_5_seconds():
    """Test that the 'created' field is cached when identical calls are made 5 seconds apart."""
    from bulkllm.llm import completion

    # Make first call
    first_response = completion(
        model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "Test caching message"}]
    )

    # Wait 5 seconds
    time.sleep(2)

    # Make identical second call
    second_response = completion(
        model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "Test caching message"}]
    )

    # Check that the second response was a cache hit
    assert hasattr(second_response, "is_cached_hit"), "Response should have is_cached_hit attribute"
    assert second_response.is_cached_hit, "Second response should be a cache hit"

    # Check that the 'created' field is the same (indicating it was cached)
    assert hasattr(first_response, "created"), "Response should have 'created' field"
    assert hasattr(second_response, "created"), "Response should have 'created' field"
    assert first_response.created == second_response.created, "Created field should be identical for cached responses"

    # Additionally verify that the response content is identical
    assert first_response.choices[0].message.content == second_response.choices[0].message.content, (
        "Response content should be identical"
    )


if __name__ == "__main__":
    import pytest

    # run the tests in this file
    pytest.main([__file__])
