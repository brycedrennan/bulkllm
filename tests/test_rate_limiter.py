import pytest

from bulkllm.rate_limiter import ModelRateLimit, RateLimiter


def test_has_capacity_exceeds_itpm():
    limit = ModelRateLimit(model_names=["m"], itpm=10)
    with pytest.raises(ValueError, match="input tokens per minute limit"):
        limit.has_capacity(15, 0)


def test_reserve_record_usage_sync():
    limit = ModelRateLimit(model_names=["m"], rpm=2, tpm=50, itpm=50, otpm=50)
    with limit.reserve_capacity_sync(10, 5) as ctx:
        ctx.record_usage_sync(10, 5)
    assert limit.current_requests_in_window == 1
    assert not limit._pending_requests
    assert len(limit._completed_requests) == 1


def test_exit_without_record_raises():
    limit = ModelRateLimit(model_names=["m"], rpm=1)
    with pytest.raises(RuntimeError, match="Usage must be recorded"):
        with limit.reserve_capacity_sync(1, 1):
            pass
    assert not limit._pending_requests
    assert limit.current_requests_in_window == 0


def test_exit_with_exception_propagates_and_cancels():
    limit = ModelRateLimit(model_names=["m"], rpm=1)
    with pytest.raises(ValueError, match="boom"):
        with limit.reserve_capacity_sync(1, 1):
            raise ValueError("boom")
    assert not limit._pending_requests
    assert not limit._completed_requests


def test_get_rate_limit_for_model_with_regex():
    rl = RateLimiter([])
    pattern_limit = ModelRateLimit(model_names=["^foo-.*$"], rpm=5, is_regex=True)
    rl.add_rate_limit(pattern_limit)
    assert rl.get_rate_limit_for_model("foo-bar") is pattern_limit
    assert rl.get_rate_limit_for_model("other") is rl.default_rate_limit

