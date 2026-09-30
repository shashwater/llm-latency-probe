# llm-latency-probe

Benchmark **time to first token (TTFT)** and **tokens/sec** across any
OpenAI-compatible LLM inference endpoint - hosted APIs, or your own
vLLM / llama.cpp server - under real concurrent load.

Built to answer one question honestly: *how fast does this endpoint actually
feel, and how does it hold up when more than one request is in flight?*

## Why

Every voice and agent product lives or dies on the latency budget of its
inference calls. A single curl tells you nothing - you want percentiles under
concurrency, split into the parts you can optimize:

- **TTFT (p50 / p90 / p99)** - what the user perceives before anything renders
- **Total latency (p50 / p99)** - full completion time
- **tok/s per request** - decode speed a single user experiences
- **tok/s aggregate** - real throughput of the endpoint under load

## Install

```bash
pip install -r requirements.txt   # httpx, rich - that's it
```

No SDK dependencies. Any endpoint that speaks the OpenAI
`/v1/chat/completions` streaming format works.

## Usage

```bash
# one endpoint
python -m llm_latency_probe --endpoint openai --model gpt-4o-mini \
    --requests 32 --concurrency 8

# head-to-head comparison (repeatable --endpoint)
python -m llm_latency_probe --model gpt-4o-mini \
    --endpoint openai \
    --endpoint groq=https://api.groq.com/openai/v1 \
    --endpoint local=http://localhost:8000/v1 \
    --requests 32 --concurrency 8 --output-csv results.csv
```

Presets (`openai`, `openrouter`, `groq`, `together`, `local`) fill in the
base URL and read the key from the matching env var
(`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`).
Anything else takes `label=base_url` plus `--api-key` or `$OPENAI_API_KEY`.

| flag | default | meaning |
|---|---|---|
| `--requests` | 16 | probes per endpoint |
| `--concurrency` | 4 | max requests in flight |
| `--max-tokens` | 128 | completion cap per probe |
| `--prompt` | built-in | same prompt for every probe, so numbers compare |
| `--output-csv` / `--output-json` | - | export results |

## Sample output

Real run (12 requests, concurrency 4, against a local mock with a simulated
150 ms TTFT):

```
probing mock (http://127.0.0.1:8765/v1) x12 @ concurrency 4 ...
        llm-latency-probe results
┏━━━━━━━━━━┳━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ endpoint ┃ ok ┃ fail ┃ ttft p50  ┃ ttft p90  ┃ lat p50 ┃ lat p99 ┃ tok/s (req) ┃ tok/s (agg) ┃
┡━━━━━━━━━━╇━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ mock     │ 12 │    0 │       155 │       156 │     369 │     401 │        32.7 │        83.9 │
└──────────┴────┴──────┴───────────┴───────────┴─────────┴─────────┴─────────────┴─────────────┘
```

## How it measures

- Streams every request (`stream: true`) and timestamps the **first content
  token** - that is TTFT. Token counts come from the server's `usage` report.
- When a server omits usage stats, streamed chunks are counted instead - but
  servers may batch several tokens per chunk, so those tok/s numbers are
  **marked with `~`** in the table and flagged as `token_counts_estimated` in
  CSV/JSON exports. Treat them as chunk-rate approximations, not true tok/s.
- Per-request tok/s excludes the first token (`(tokens - 1) / decode time`)
  so TTFT does not pollute the decode-rate number.
- Aggregate tok/s divides total output tokens by wall-clock time for the whole
  run - the number that matters for capacity planning.
- Failures (non-200s, timeouts, connection errors) are counted and reported,
  never silently dropped. A stream that ends with no usage report and no
  content tokens counts as a failure ("empty stream"), not a zero-token pass.

## Tests

```bash
python -m pytest tests/ -q
```

Includes an end-to-end test against a mock streaming server, so correctness
is validated without any API key or spend.

## License

MIT
