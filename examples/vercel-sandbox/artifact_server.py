"""Serve only disposable review fixture image archives, with short-lived signatures."""

import hashlib
import hmac
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path("/tmp/poc-artifacts")
KEY = b"agentenv-poc-public-image-fixture"
ALLOWED = {"manifest.json", *(name + ".tar.gz" for name in (
    "poc-gateway", "poc-db", "poc-db-web", "poc-db-mcp", "poc-items", "poc-agent"))}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        request = urlsplit(self.path)
        name = request.path.removeprefix("/")
        query = parse_qs(request.query)
        expires = query.get("expires", ["0"])[0]
        signature = query.get("signature", [""])[0]
        expected = hmac.new(KEY, (name + ":" + expires).encode(), hashlib.sha256).hexdigest()
        if name not in ALLOWED:
            self.send_error(404)
            return
        if not expires.isdigit() or int(expires) < time.time() or not hmac.compare_digest(signature, expected):
            self.send_error(403)
            return
        path = ROOT / name
        if not path.is_file():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as source:
            shutil.copyfileobj(source, self.wfile)


ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
