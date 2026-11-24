"""The model endpoint record.py's error turns go through, run by it in a container beside the agent. On each port it
forwards every request to ``UPSTREAM``, streaming the answer back, except:

- refuse (``PORT_REFUSE``): every ``/v1/messages`` call is answered 400, so the CLI ends the turn with an API error;
- stall (``PORT_STALL``): after the first ``/v1/messages`` call, each call hangs until the client gives up, unless its
  body holds the bridge's resume prompt, so the CLI's attempt fails and the bridge's resume goes through.

Standard library only; it never logs a header or a body.

    UPSTREAM=https://... python3 fault_proxy.py BIND_HOST PORT_REFUSE PORT_STALL"""
import http.client
import http.server
import json
import os
import sys
import threading
import time
import urllib.parse

UPSTREAM = urllib.parse.urlsplit(os.environ["UPSTREAM"].strip())
# The endpoint's root, as the bridge's ``_api_root`` takes it: the CLI appends its own /v1/messages.
ROOT = UPSTREAM.path.rstrip("/").removesuffix("/v1").rstrip("/")
CONNECTION = http.client.HTTPSConnection if UPSTREAM.scheme == "https" else http.client.HTTPConnection
RESUME = b"interrupted by a transient API error"
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization", "host",
       "content-length", "accept-encoding"}


def handler(mode):
    calls = {"n": 0}
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.forward(b"")

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path.startswith("/v1/messages") and "count_tokens" not in self.path:
                with lock:
                    calls["n"] += 1
                    n = calls["n"]
                print(f"{mode} call {n} {self.path}", flush=True)
                if mode == "refuse":
                    payload = json.dumps({"type": "error", "error": {
                        "type": "invalid_request_error", "message": "refused by the recording proxy"}}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if mode == "stall" and n > 1 and RESUME not in body:
                    try:
                        time.sleep(600)
                    finally:
                        return
            self.forward(body)

        def forward(self, body):
            conn = CONNECTION(UPSTREAM.hostname, UPSTREAM.port, timeout=600)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            conn.request(self.command, ROOT + self.path, body=body or None, headers=headers)
            response = conn.getresponse()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in HOP:
                    self.send_header(key, value)
            self.end_headers()
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
            conn.close()

    return Handler


host, refuse, stall = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
servers = [http.server.ThreadingHTTPServer((host, refuse), handler("refuse")),
           http.server.ThreadingHTTPServer((host, stall), handler("stall"))]
for server in servers:
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
print("listening", flush=True)
threading.Event().wait()
