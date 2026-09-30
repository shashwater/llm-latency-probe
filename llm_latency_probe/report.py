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
            if kind == "ms":
                row.append(f"{v:,.0f}")
            elif kind == "tps":
                row.append(f"{v:,.1f}{est}")
            else:
                row.append(str(v))
        table.add_row(*row)
    console.print(table)
    for s in summaries:
        if s.token_counts_estimated:
            console.print(
                f"[yellow]{s.label}: tok/s marked ~ is estimated from streamed chunks "
                f"- the server omitted usage stats, so chunk rate approximates token rate[/yellow]")
        if s.errors:
            console.print(f"[red]{s.label}: {s.failed} failed[/red] - {', '.join(s.errors)}")


def write_csv(summaries: list[EndpointSummary], path: str) -> None:
    fields = [attr for _, attr, _ in _COLUMNS] + [
        "total_output_tokens", "token_counts_estimated", "errors"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for s in summaries:
            d = asdict(s)
            w.writerow(["; ".join(d[k]) if k == "errors" else d[k] for k in fields])


def write_json(summaries: list[EndpointSummary], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2)
