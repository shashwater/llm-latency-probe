"""Local SSE fixtures. No provider calls or credentials."""
import asyncio
import csv
import io
import json

import httpx
import pytest
from rich.console import Console

from llm_latency_probe.probe import Endpoint, probe_once
from llm_latency_probe.report import print_table, write_csv, write_json
from llm_latency_probe.stats import summarize


def run(chunks, done=True):
    body = ''.join('data: ' + (c if isinstance(c, str) else json.dumps(c)) + '\n\n'
                   for c in chunks)
    if done:
        body += 'data: [DONE]\n\n'

    async def go():
        transport = httpx.MockTransport(lambda req: httpx.Response(
            200, text=body, headers={'Content-Type': 'text/event-stream'}))
        async with httpx.AsyncClient(transport=transport) as client:
            return await probe_once(client, Endpoint('fixture', 'http://fixture/v1', 'fake'),
                                    'no remote inference', 32)
    return asyncio.run(go())


def delta(value, finish=None):
    return {'choices': [{'index': 0, 'delta': value, 'finish_reason': finish}]}


def call(arguments, name='lookup', index=0):
    return {'tool_calls': [{'index': index, 'function': {'name': name, 'arguments': arguments}}]}


def tool_chunks(usage=False):
    chunks = [delta(call('{"id":')), delta(call('7}', name='')),
              delta({}, 'tool_calls')]
    if usage:
        chunks += [{'choices': [], 'usage': {'completion_tokens': 12}}]
    return chunks


def test_tool_only_no_usage_is_success_not_empty():
    r = run(tool_chunks())
    assert r.ok and r.output_kind == 'tool_only'
    assert r.ttft_ms is None
    assert r.first_tool_argument_ms is not None
    assert r.tool_calls_completed == 1
    assert r.completion_tokens is None and r.tokens_per_sec is None
    assert not r.token_count_estimated


def test_tool_only_with_usage_has_no_text_ttft():
    r = run(tool_chunks(True))
    assert r.ok and r.completion_tokens == 12
    assert r.ttft_ms is None and r.first_tool_argument_ms is not None
    assert r.tokens_per_sec is not None


def test_mixed_text_and_tool_without_usage_is_not_chunk_tokens():
    r = run([delta({'content': 'Looking up.'})] + tool_chunks())
    assert r.ok and r.output_kind == 'mixed'
    assert r.ttft_ms is not None and r.first_tool_argument_ms is not None
    assert r.completion_tokens is None


@pytest.mark.parametrize('args', ['{"id":', 'not JSON', '[]', 'null'])
def test_malformed_or_nonobject_arguments_fail(args):
    r = run([delta(call(args)), delta({}, 'tool_calls')])
    assert not r.ok and 'argument' in r.error


@pytest.mark.parametrize('done,finish', [(False, True), (True, False)])
def test_truncated_tool_stream_fails(done, finish):
    chunks = [delta(call('{}'))]
    if finish:
        chunks += [delta({}, 'tool_calls')]
    r = run(chunks, done=done)
    assert not r.ok and 'incomplete' in r.error


def test_length_finish_does_not_claim_complete_call():
    r = run([delta(call('{}')), delta({}, 'length')])
    assert not r.ok and 'incomplete' in r.error


def test_parallel_calls_assemble_by_index():
    r = run([delta(call('{"x":', index=0)), delta(call('{}', name='second', index=1)),
             delta(call('1}', name='', index=0)), delta({}, 'tool_calls')])
    assert r.ok and r.tool_calls_completed == 2


def test_legacy_function_call():
    r = run([delta({'function_call': {'name': 'lookup', 'arguments': '{}'}}),
             delta({}, 'function_call')])
    assert r.ok and r.output_kind == 'tool_only'


def test_usage_only_is_empty_not_fake_success():
    r = run([{'choices': [], 'usage': {'completion_tokens': 4}}])
    assert not r.ok and 'empty' in r.error


def test_bad_sse_json_is_not_silently_ignored():
    r = run(['{bad', delta({'content': 'text'})])
    assert not r.ok and 'malformed' in r.error


def test_tool_summary_and_exports_use_unavailable_not_zero(tmp_path):
    r = run(tool_chunks())
    s = summarize('fixture', [r], 1)
    assert s.text_responses == 0 and s.tool_responses == 1
    assert s.ttft_p50_ms is None and s.first_tool_argument_p50_ms is not None
    assert s.total_output_tokens is None and s.agg_tps is None and s.mean_tps is None
    assert s.token_counts_unavailable
    j, c = tmp_path / 'out.json', tmp_path / 'out.csv'
    write_json([s], str(j)); write_csv([s], str(c))
    assert json.loads(j.read_text())[0]['ttft_p50_ms'] is None
    assert list(csv.DictReader(c.open()))[0]['ttft_p50_ms'] == ''
    console = Console(file=io.StringIO(), width=160, color_system=None)
    print_table([s], console)
    assert 'N/A' in console.file.getvalue()
    assert 'tool arg p50' in console.file.getvalue()


def test_mixed_summary_excludes_tool_only_from_text_percentile():
    text = run([delta({'content': 'text'})])
    tool = run(tool_chunks())
    s = summarize('fixture', [text, tool], 1)
    assert s.ttft_p50_ms == text.ttft_ms
    assert s.text_responses == 1 and s.tool_responses == 1
    assert s.token_counts_unavailable
