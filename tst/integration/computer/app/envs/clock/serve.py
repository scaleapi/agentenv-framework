import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from app import PORT, Clock

zone = Clock().settings.zone
Path.home().joinpath(".clock.json").write_text(json.dumps({"zone": zone}))


class Zone(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"zone": zone, "known": Path("/usr/share/zoneinfo", zone).exists()}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


HTTPServer(("127.0.0.1", PORT), Zone).serve_forever()
