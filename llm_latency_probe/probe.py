"""Async streaming probe against OpenAI-compatible chat-completions endpoints."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import httpx


@dataclass
class Endpoint:
    label: str
    base_url: str          # e.g. http://localhost:8000/v1
    model: str
    api_key: str = ""


@dataclass
class RequestResult:
    ok: bool
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    completion_tokens: int = 0
    tokens_per_sec: float = 0.0
    token_count_estimated: bool = False   # True when server omitted usage stats
    error: str | None = None


def _payload(endpoint: Endpoint, prompt: str, max_tokens: int) -> dict:
    return {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def probe_once(client: httpx.AsyncClient, endpoint: Endpoint,
                     prompt: str, max_tokens: int) -> RequestResult:
    """One streaming chat completion. TTFT = time to first content token."""
    url = endpoint.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"

    start = time.perf_counter()
    ttft_ms = 0.0
    usage_tokens = 0
    counted_deltas = 0
    try:
        async with client.stream("POST", url, headers=headers,
                                 json=_payload(endpoint, prompt, max_tokens)) as resp:
            if resp.status_code != 200:
                body = (await resp.aread())[:300].decode("utf-8", "replace")
                return RequestResult(ok=False, error=f"HTTP {resp.status_code}: {body}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens"):
                    usage_tokens = usage["completion_tokens"]
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        counted_deltas += 1
                        if ttft_ms == 0.0:
                            ttft_ms = (time.perf_counter() - start) * 1000
    except httpx.HTTPError as e:
        return RequestResult(ok=False, error=f"{type(e).__name__}: {e}")

    total_ms = (time.perf_counter() - start) * 1000

    # An empty stream - no usage report and no content tokens - is a failure,
    # not a zero-token success.
    if usage_tokens == 0 and counted_deltas == 0:
        return RequestResult(ok=False, total_ms=total_ms,
                             error="empty stream: server returned no completion tokens")

    # Prefer the server's usage report. When absent, streamed chunks are only
    # an approximation of token count (servers may batch multiple tokens per
    # chunk), so mark the count as estimated.
    estimated = usage_tokens == 0
    tokens = usage_tokens or counted_deltas
    gen_ms = total_ms - ttft_ms
    tps = (tokens - 1) / (gen_ms / 1000) if gen_ms > 0 and tokens > 1 else 0.0
    return RequestResult(
        ok=True, ttft_ms=ttft_ms, total_ms=total_ms,
        completion_tokens=tokens, tokens_per_sec=tps,
        token_count_estimated=estimated,
    )


async def run_probe(endpoint: Endpoint, prompt: str, max_tokens: int,
                    requests: int, concurrency: int) -> tuple[list[RequestResult], float]:
    """Fire `requests` probes at `endpoint` with at most `concurrency` in flight.
    Returns (results, wall_seconds)."""
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)
    results: list[RequestResult] = []

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        async def one() -> None:
            async with sem:
                results.append(await probe_once(client, endpoint, prompt, max_tokens))

        start = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(requests)))
        wall = time.perf_counter() - start
    return results, wall
