"""Tiny localhost listener that mirrors live draft state to a file.

This is the latency fix for agent use, and the reason is structural rather
than clever.

An agent driving a browser pays, on every single step: page snapshot ->
serialise the accessibility tree -> model reads it -> tool call -> repeat. That
is several seconds per round trip before the model has thought about anything,
and it scales with page complexity -- the ESPN draft room is a heavy React app.
No amount of tuning makes that fit inside a 90-second pick clock.

But almost none of that work is necessary. The draft state is small: a pick
number, a slot, and a list of names. If the page pushes that to a file, an
agent reads ONE tiny blob in a single round trip and skips the browser
entirely. The remaining cost is one inference, which is the irreducible part.

So: the draft page POSTs state here on every change, this writes
`output/draft_state.json`, and `draft_advise.py` turns that into a briefing.

usage:  draft_state_server.py [port]        default 8788
        (leave it running in a terminal during the draft)
"""

import json
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
STATE = OUT / "draft_state.json"
PORT = 8788


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        # the page is served from file:// or an extension origin, so the
        # browser will preflight the JSON POST
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        body = STATE.read_bytes() if STATE.exists() else b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self.send_response(400)
            self._cors()
            self.end_headers()
            return
        data["written_at"] = datetime.now().isoformat(timespec="seconds")
        STATE.write_text(json.dumps(data, indent=1))
        picks = data.get("picks", [])
        mine = sum(1 for p in picks if p.get("mine"))
        print(f"  pick {data.get('pick', '?'):>3}  "
              f"({len(picks)} gone, {mine} mine)", flush=True)
        self.send_response(204)
        self._cors()
        self.end_headers()

    def log_message(self, *a) -> None:      # silence the default access log
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    OUT.mkdir(exist_ok=True)
    print(f"draft state -> {STATE}")
    print(f"listening on http://127.0.0.1:{port}  (ctrl-c to stop)\n")
    try:
        HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
