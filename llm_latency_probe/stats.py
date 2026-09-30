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
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0
    mean_tps: float = 0.0          # mean per-request output tokens/sec
    agg_tps: float = 0.0           # total output tokens / wall time (throughput under load)
    total_output_tokens: int = 0
    token_counts_estimated: bool = False  # True if any success lacked server usage stats


def summarize(label: str, results, wall_seconds: float) -> EndpointSummary:
    """results: iterable of RequestResult (has ok, ttft_ms, total_ms,
    completion_tokens, token_count_estimated, error)."""
    results = list(results)
    good = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    ttfts = [r.ttft_ms for r in good]
    lats = [r.total_ms for r in good]
    tps = [r.tokens_per_sec for r in good if r.tokens_per_sec > 0]
    total_tokens = sum(r.completion_tokens for r in good)
    return EndpointSummary(
        label=label,
        total=len(results),
        ok=len(good),
        failed=len(bad),
        errors=sorted({(r.error or "unknown") for r in bad}),
        ttft_p50_ms=percentile(ttfts, 50),
        ttft_p90_ms=percentile(ttfts, 90),
        ttft_p99_ms=percentile(ttfts, 99),
        latency_p50_ms=percentile(lats, 50),
        latency_p90_ms=percentile(lats, 90),
        latency_p99_ms=percentile(lats, 99),
        mean_tps=mean(tps),
        agg_tps=(total_tokens / wall_seconds) if wall_seconds > 0 else 0.0,
        total_output_tokens=total_tokens,
        token_counts_estimated=any(
            getattr(r, "token_count_estimated", False) for r in good),
    )
