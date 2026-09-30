import asyncio
import threading

from llm_latency_probe.probe import Endpoint, run_probe
from llm_latency_probe.stats import summarize
from tests.mock_server import serve, TOKENS, FIRST_TOKEN_DELAY_S


def _run(port, prompt, requests=6):
    ep = Endpoint(label="mock", base_url=f"http://127.0.0.1:{port}/v1", model="mock")
    return asyncio.run(run_probe(ep, prompt, max_tokens=16,
                                 requests=requests, concurrency=3))


def test_probe_against_mock_server():
    srv, port = serve()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        results, wall = _run(port, "hi")
        assert all(r.ok for r in results), [r.error for r in results]
        assert all(r.completion_tokens == len(TOKENS) for r in results)
        # server-reported usage means counts are exact, not estimated
        assert all(not r.token_count_estimated for r in results)
        # mock sleeps 150ms before first token; TTFT >= that, well under total
        assert all(r.ttft_ms >= FIRST_TOKEN_DELAY_S * 1000 * 0.9 for r in results)
        assert all(r.total_ms > r.ttft_ms for r in results)
        s = summarize("mock", results, wall)
        assert s.ok == 6 and s.failed == 0
        assert s.total_output_tokens == 6 * len(TOKENS)
        assert not s.token_counts_estimated
    finally:
        srv.shutdown()


def test_empty_stream_is_failure():
    srv, port = serve()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        results, _ = _run(port, "EMPTY please", requests=3)
        assert all(not r.ok for r in results)
        assert all("empty stream" in (r.error or "") for r in results)
        s = summarize("mock", results, 1.0)
        assert s.ok == 0 and s.failed == 3
        assert s.total_output_tokens == 0
    finally:
        srv.shutdown()


def test_missing_usage_marks_estimate():
    srv, port = serve()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        results, _ = _run(port, "NOUSAGE please", requests=3)
        assert all(r.ok for r in results)
        assert all(r.token_count_estimated for r in results)
        # falls back to counting streamed deltas
        assert all(r.completion_tokens == len(TOKENS) for r in results)
        s = summarize("mock", results, 1.0)
        assert s.token_counts_estimated
    finally:
        srv.shutdown()
