"""Terminal table and file exports for probe summaries."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict

from rich.console import Console
from rich.table import Table

from .stats import EndpointSummary

_COLUMNS = [
    ("endpoint", "label", str),
    ("ok", "ok", int),
    ("fail", "failed", int),
    ("ttft p50", "ttft_p50_ms", "ms"),
    ("ttft p90", "ttft_p90_ms", "ms"),
    ("tool arg p50", "first_tool_argument_p50_ms", "ms"),
    ("lat p50", "latency_p50_ms", "ms"),
    ("lat p99", "latency_p99_ms", "ms"),
    ("tok/s (req)", "mean_tps", "tps"),
    ("tok/s (agg)", "agg_tps", "tps"),
]


def print_table(summaries: list[EndpointSummary], console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="llm-latency-probe results", title_justify="left")
    for header, _, _ in _COLUMNS:
        table.add_column(header, justify="right" if header != "endpoint" else "left")
    for s in summaries:
        est = "~" if s.token_counts_estimated else ""
        row = []
        for _, attr, kind in _COLUMNS:
            v = getattr(s, attr)
            if v is None:
                row.append("N/A")
            elif kind == "ms":
                row.append(f"{v:,.0f}")
            elif kind == "tps":
                row.append(f"{v:,.1f}{est}")
            else:
                row.append(str(v))
        table.add_row(*row)
    console.print(table)
    for s in summaries:
        if s.tool_responses:
            console.print(f"{s.label}: {s.text_responses} text-bearing, "
                          f"{s.tool_responses} tool-bearing responses; "
                          f"{s.tool_calls_completed} completed tool calls. "
                          "Text TTFT excludes tool-only responses; tool arg latency is separate.")
        if s.token_counts_unavailable:
            console.print(f"{s.label}: token throughput N/A: tool output lacks server usage.")
        if s.token_counts_estimated:
            console.print(
                f"[yellow]{s.label}: tok/s marked ~ is estimated from streamed chunks "
                f"- the server omitted usage stats, so chunk rate approximates token rate[/yellow]")
        if s.errors:
            console.print(f"[red]{s.label}: {s.failed} failed[/red] - {', '.join(s.errors)}")


def write_csv(summaries: list[EndpointSummary], path: str) -> None:
    fields = [attr for _, attr, _ in _COLUMNS] + [
        "total_output_tokens", "token_counts_estimated", "token_counts_unavailable",
        "text_responses", "tool_responses", "tool_calls_completed", "errors"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for s in summaries:
            d = asdict(s)
            w.writerow(["; ".join(d[k]) if k == "errors" else d[k] for k in fields])


def write_json(summaries: list[EndpointSummary], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2)
