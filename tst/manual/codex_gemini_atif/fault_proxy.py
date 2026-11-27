"""The model endpoint record.py's error turns go through, run by it in a container beside the agent. On each port it
forwards every request to ``UPSTREAM``, except POSTs, the model calls:

- refuse (``PORT_REFUSE``): every one is answered 400, so the CLI ends the turn with an API error;
- resume (``PORT_RESUME``): after the first, each is answered 503 until one holds the Codex bridge's resume prompt,
  so the CLI's attempt fails and the bridge's resume goes through;
- retry (``PORT_RETRY``): the second and third are answered 503, so the CLI retries them itself.

Standard library only; it never logs a header or a body.

    UPSTREAM=https://... python3 fault_proxy.py BIND_HOST PORT_REFUSE PORT_RESUME PORT_RETRY"""
import http.client
import http.server
import os
import sys
import threading
import urllib.parse

UPSTREAM = urllib.parse.urlsplit(os.environ["UPSTREAM"].strip())
CONNECTION = http.client.HTTPSConnection if UPSTREAM.scheme == "https" else http.client.HTTPConnection
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "host", "content-length",
       "accept-encoding"}
RESUME = b"interrupted by a transient API error"
UNAVAILABLE = b'{"error": {"message": "service unavailable", "code": 503}}'


def handler(mode):
    state = {"posts": 0, "resumed": False}
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            print(mode, self.command, self.path.split("?")[0][:80], flush=True)

        def answer(self, status, body):
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def forward(self, body=None):
            conn = CONNECTION(UPSTREAM.netloc, timeout=300)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            conn.request(self.command, UPSTREAM.path.rstrip("/") + self.path, body=body, headers=headers)
            response = conn.getresponse()
            data = response.read()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in HOP:
                    self.send_header(key, value)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
            if mode == "refuse":
                return self.answer(400, b'{"error": {"message": "refused by the recording proxy", "code": 400}}')
            with lock:
                state["posts"] += 1
                posts = state["posts"]
                state["resumed"] = state["resumed"] or RESUME in body
                resumed = state["resumed"]
            if mode == "resume" and posts > 1 and not resumed:
                return self.answer(503, UNAVAILABLE)
            if mode == "retry" and posts in (2, 3):
                return self.answer(503, UNAVAILABLE)
            return self.forward(body)

        def do_GET(self):
            self.forward()

    return Handler


for mode, port in zip(("refuse", "resume", "retry"), sys.argv[2:5]):
    server = http.server.ThreadingHTTPServer((sys.argv[1], int(port)), handler(mode))
    threading.Thread(target=server.serve_forever, daemon=True).start()
threading.Event().wait()
