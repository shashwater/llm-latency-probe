"""A tiny OpenAI-compatible streaming server for end-to-end validation.
Delays before the first token (simulated TTFT) and between tokens.

Test hooks via the prompt string:
  prompt containing "EMPTY"   -> stream ends with [DONE] and no content/usage
  prompt containing "NOUSAGE" -> streams content but omits the usage report
"""
from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKENS = ["The", " token", " bucket", " refills", " steadily", " over", " time", "."]
FIRST_TOKEN_DELAY_S = 0.15
INTER_TOKEN_DELAY_S = 0.03


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send_chunk(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self.send_response(404); self.end_headers(); return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = ""
        for msg in body.get("messages", []):
            prompt += msg.get("content", "")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        if "EMPTY" in prompt:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        time.sleep(FIRST_TOKEN_DELAY_S)
        for i, tok in enumerate(TOKENS):
            self._send_chunk({
                "id": "mock", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
            })
            if i < len(TOKENS) - 1:
                time.sleep(INTER_TOKEN_DELAY_S)
        if "NOUSAGE" not in prompt:
            self._send_chunk({
                "id": "mock", "object": "chat.completion.chunk", "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": len(TOKENS),
                          "total_tokens": 20},
            })
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def serve(port: int = 0) -> tuple[ThreadingHTTPServer, int]:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return srv, srv.server_address[1]
