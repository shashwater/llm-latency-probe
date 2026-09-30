import pytest

from llm_latency_probe.stats import percentile, mean, summarize


class R:
    def __init__(self, ok=True, ttft_ms=100.0, total_ms=500.0, completion_tokens=10,
                 tokens_per_sec=20.0, error=None):
        self.ok = ok
        self.ttft_ms = ttft_ms
        self.total_ms = total_ms
        self.completion_tokens = completion_tokens
        self.tokens_per_sec = tokens_per_sec
        self.error = error


def test_percentile_basic():
    vals = [10, 20, 30, 40, 50]
    assert percentile(vals, 50) == 30
    assert percentile(vals, 0) == 10
    assert percentile(vals, 100) == 50


def test_percentile_interpolates():
    assert percentile([0, 100], 25) == 25


def test_percentile_edge_cases():
    assert percentile([], 50) == 0.0
    assert percentile([7], 99) == 7
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_mean_empty():
    assert mean([]) == 0.0


def test_summarize_counts_failures():
    results = [R(), R(), R(ok=False, error="HTTP 500: boom"), R(ok=False, error="HTTP 500: boom")]
    s = summarize("ep", results, wall_seconds=2.0)
    assert s.total == 4 and s.ok == 2 and s.failed == 2
    assert s.errors == ["HTTP 500: boom"]
    assert s.total_output_tokens == 20
    assert s.agg_tps == pytest.approx(10.0)  # 20 tokens / 2s wall


def test_summarize_percentiles():
    results = [R(ttft_ms=float(t), total_ms=float(t * 5)) for t in range(1, 11)]
    s = summarize("ep", results, wall_seconds=1.0)
    assert s.ttft_p50_ms == pytest.approx(percentile([float(t) for t in range(1, 11)], 50))
    assert s.latency_p99_ms >= s.latency_p50_ms
