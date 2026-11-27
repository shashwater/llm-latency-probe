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
    ttft_ms: float | None = None
    total_ms: float = 0.0
    completion_tokens: int | None = 0
    tokens_per_sec: float | None = 0.0
    token_count_estimated: bool = False   # True when server omitted usage stats
    error: str | None = None
    output_kind: str = "empty"
    first_tool_argument_ms: float | None = None
    tool_calls_completed: int = 0


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
    ttft_ms = None
    first_tool_argument_ms = None
    first_output_ms = None
    tools = {}
    finished_choices = set()
    saw_done = False
    invalid_stream = False
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
                    saw_done = True
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    invalid_stream = True
                    continue
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens"):
                    usage_tokens = usage["completion_tokens"]
                for choice in chunk.get("choices") or []:
                    choice_index = choice.get("index", 0)
                    if choice.get("finish_reason") in ("tool_calls", "function_call"):
                        finished_choices.add(choice_index)
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    elapsed = (time.perf_counter() - start) * 1000
                    if text:
                        counted_deltas += 1
                        if ttft_ms is None:
                            ttft_ms = elapsed
                        if first_output_ms is None:
                            first_output_ms = elapsed
                    calls = delta.get("tool_calls") or []
                    # Legacy OpenAI function_call streams have one unnamed-index call.
                    if delta.get("function_call"):
                        calls = [{"index": 0, "function": delta["function_call"]}]
                    for call in calls:
                        key = (choice_index, call.get("index", 0))
                        item = tools.setdefault(key, {"name": "", "arguments": ""})
                        function = call.get("function") or {}
                        item["name"] += function.get("name") or ""
                        arguments = function.get("arguments") or ""
                        item["arguments"] += arguments
                        if arguments and first_tool_argument_ms is None:
                            first_tool_argument_ms = elapsed
                        if first_output_ms is None:
                            first_output_ms = elapsed
    except httpx.HTTPError as e:
        return RequestResult(ok=False, error=f"{type(e).__name__}: {e}")

    total_ms = (time.perf_counter() - start) * 1000

    kind = "mixed" if tools and counted_deltas else (
        "tool_only" if tools else "text" if counted_deltas else "empty")
    common = dict(total_ms=total_ms, ttft_ms=ttft_ms, output_kind=kind,
                  first_tool_argument_ms=first_tool_argument_ms)
    if invalid_stream:
        return RequestResult(ok=False, error="malformed SSE JSON", **common)
    if kind == "empty":
        return RequestResult(ok=False, error="empty stream: no text or tool calls", **common)
    for (choice_index, _), item in tools.items():
        if not saw_done or choice_index not in finished_choices:
            return RequestResult(ok=False, error="incomplete tool-call stream", **common)
        try:
            arguments = json.loads(item["arguments"])
        except (ValueError, TypeError):
            return RequestResult(ok=False, error="invalid tool argument JSON", **common)
        if not item["name"] or not isinstance(arguments, dict):
            return RequestResult(ok=False, error="tool requires name and object arguments", **common)

    # Tool argument chunks are not tokens. Without server usage we cannot infer
    # either total tokens or decode throughput for tool-only/mixed responses.
    estimated = usage_tokens == 0 and not tools
    tokens = usage_tokens or (counted_deltas if not tools else None)
    gen_ms = total_ms - first_output_ms if first_output_ms is not None else 0
    tps = ((tokens - 1) / (gen_ms / 1000) if gen_ms > 0 and tokens > 1 else 0.0
           ) if tokens is not None else None
    return RequestResult(
        ok=True, completion_tokens=tokens, tokens_per_sec=tps,
        token_count_estimated=estimated, tool_calls_completed=len(tools), **common,
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
