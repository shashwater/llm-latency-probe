"""Aggregation helpers for probe results."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile. values need not be sorted."""
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class EndpointSummary:
    label: str
    total: int = 0
    ok: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    text_responses: int = 0
    tool_responses: int = 0
    tool_calls_completed: int = 0
    first_tool_argument_p50_ms: float | None = None
    first_tool_argument_p90_ms: float | None = None
    token_counts_unavailable: bool = False
    ttft_p50_ms: float | None = None
    ttft_p90_ms: float | None = None
    ttft_p99_ms: float | None = None
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0
    mean_tps: float | None = None          # mean per-request output tokens/sec
    agg_tps: float | None = None           # total output tokens / wall time (throughput under load)
    total_output_tokens: int | None = None
    token_counts_estimated: bool = False  # True if any success lacked server usage stats


def summarize(label: str, results, wall_seconds: float) -> EndpointSummary:
    """results: iterable of RequestResult (has ok, ttft_ms, total_ms,
    completion_tokens, token_count_estimated, error)."""
    results = list(results)
    good = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    ttfts = [r.ttft_ms for r in good if r.ttft_ms is not None]
    tool_times = [r.first_tool_argument_ms for r in good
                  if getattr(r, "first_tool_argument_ms", None) is not None]
    lats = [r.total_ms for r in good]
    tps = [r.tokens_per_sec for r in good if r.tokens_per_sec is not None and r.tokens_per_sec > 0]
    unknown = any(r.completion_tokens is None for r in good)
    total_tokens = None if unknown else sum(r.completion_tokens for r in good)
    return EndpointSummary(
        label=label,
        total=len(results),
        ok=len(good),
        failed=len(bad),
        errors=sorted({(r.error or "unknown") for r in bad}),
        text_responses=len(ttfts),
        tool_responses=sum(getattr(r, "output_kind", "text") in ("tool_only", "mixed") for r in good),
        tool_calls_completed=sum(getattr(r, "tool_calls_completed", 0) for r in good),
        first_tool_argument_p50_ms=percentile(tool_times, 50) if tool_times else None,
        first_tool_argument_p90_ms=percentile(tool_times, 90) if tool_times else None,
        token_counts_unavailable=unknown,
        ttft_p50_ms=percentile(ttfts, 50) if ttfts else None,
        ttft_p90_ms=percentile(ttfts, 90) if ttfts else None,
        ttft_p99_ms=percentile(ttfts, 99) if ttfts else None,
        latency_p50_ms=percentile(lats, 50),
        latency_p90_ms=percentile(lats, 90),
        latency_p99_ms=percentile(lats, 99),
        mean_tps=None if unknown else mean(tps),
        agg_tps=None if unknown else ((total_tokens / wall_seconds) if wall_seconds > 0 else 0.0),
        total_output_tokens=total_tokens,
        token_counts_estimated=any(
            getattr(r, "token_count_estimated", False) for r in good),
    )
