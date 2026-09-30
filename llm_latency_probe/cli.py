"""Command-line interface: python -m llm_latency_probe ..."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .probe import Endpoint, run_probe
from .report import print_table, write_csv, write_json
from .stats import summarize

DEFAULT_PROMPT = (
    "Write a short, factual paragraph explaining how a token bucket rate limiter works."
)

_PRESETS = {
    # label -> (base_url, env var holding the key)
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "local": ("http://localhost:8000/v1", ""),  # vLLM / llama.cpp server default
}


def _parse_endpoint(spec: str, args) -> Endpoint:
    """Accepts 'label=url', a bare url, or a preset name."""
    if "=" in spec:
        label, url = spec.split("=", 1)
    elif spec in _PRESETS:
        label, url = spec, _PRESETS[spec][0]
    else:
        label, url = spec, spec
    key = args.api_key or ""
    if not key:
        env = _PRESETS.get(label, (None, "OPENAI_API_KEY"))[1]
        key = os.environ.get(env, "") if env else ""
    return Endpoint(label=label, base_url=url, model=args.model, api_key=key)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-latency-probe",
        description="Benchmark TTFT and tokens/sec across OpenAI-compatible LLM endpoints.",
    )
    p.add_argument("--endpoint", action="append", required=True,
                   help="label=base_url, a bare base_url, or preset: " + ", ".join(_PRESETS)
                        + " (repeatable)")
    p.add_argument("--model", required=True, help="model name sent to every endpoint")
    p.add_argument("--requests", type=int, default=16, help="probes per endpoint (default 16)")
    p.add_argument("--concurrency", type=int, default=4, help="max in-flight (default 4)")
    p.add_argument("--max-tokens", type=int, default=128, help="completion cap (default 128)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt sent to every probe")
    p.add_argument("--api-key", default="", help="override key for all endpoints "
                                                 "(else $OPENAI_API_KEY / preset env var)")
    p.add_argument("--output-csv", metavar="PATH", help="write summary rows to CSV")
    p.add_argument("--output-json", metavar="PATH", help="write full summaries to JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.concurrency < 1 or args.requests < 1:
        print("requests and concurrency must be >= 1", file=sys.stderr)
        return 2

    endpoints = [_parse_endpoint(spec, args) for spec in args.endpoint]
    summaries = []
    for ep in endpoints:
        print(f"probing {ep.label} ({ep.base_url}) x{args.requests} "
              f"@ concurrency {args.concurrency} ...", file=sys.stderr)
        results, wall = asyncio.run(
            run_probe(ep, args.prompt, args.max_tokens, args.requests, args.concurrency))
        summaries.append(summarize(ep.label, results, wall))

    print_table(summaries)
    if args.output_csv:
        write_csv(summaries, args.output_csv)
    if args.output_json:
        write_json(summaries, args.output_json)
    return 1 if all(s.ok == 0 for s in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
