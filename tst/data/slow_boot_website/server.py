"""A site that answers nothing until it has booted, as a third-party app's server can take minutes to: it sleeps the
seconds its settings' ``boot_s`` gives, then serves 200 at its health path and, at ``/shm``, the bytes its ``/dev/shm``
holds. It serves no card, so a deploy reads its readiness at its health path."""

import http.server
import json
import os
import time

time.sleep(float(json.loads(os.environ.get("AGENTENV_WEBSITE_SETTINGS", "{}")).get("boot_s", 0)))


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health":
            body = b"ok"
        elif self.path == "/shm":
            shm = os.statvfs("/dev/shm")
            body = str(shm.f_frsize * shm.f_blocks).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


http.server.ThreadingHTTPServer(("0.0.0.0", int(os.environ["PORT"])), Handler).serve_forever()
