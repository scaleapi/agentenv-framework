"""Pages a computer's start tabs open, served on this machine, which a computer reaches as ``host.docker.internal``: a
site's route, whose home page sends itself elsewhere before it loads, or answers an error; pages titled by name; an
event stream that sends one event and holds the response open; and one that never answers."""

import contextlib
import http.server
import threading
import time
from collections.abc import Iterator

import httpx

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.types import WELL_KNOWN_PATH
from agentenv_protocol.website_browser import BROWSER

#: How long the page that never answers holds its request, and the image that holds the site's home page from loading.
HANG_S, SLOW_S = 120, 30
#: The site's home page: it replaces itself with /landed by script as it's read, while an image keeps it from loading.
LEAVING = b"<title>leaving</title><script>location.replace('/landed')</script><img src='/slow'>"


@contextlib.contextmanager
def host_pages(site_status: int = 200) -> Iterator[int]:
    """The pages served on every address of this machine while the block runs, at the port it yields: under
    ``/website/shop/``, a site's route, its home page, which sends itself to ``landed`` before it loads, or with any
    other ``site_status`` answers it, and ``landed``; ``/alpha`` and ``/bravo``, and any other path, titled by its last
    part; an ``events`` path, an event stream that sends one event and holds the response open for ``SLOW_S``; and
    ``/hang``, which never answers."""

    class Page(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/hang", "/website/shop/slow"):
                time.sleep(HANG_S if self.path == "/hang" else SLOW_S)
                return
            if self.path.endswith("/events"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: first\n\n")
                self.wfile.flush()
                time.sleep(SLOW_S)
                return
            home = self.path == "/website/shop/"
            status = site_status if home else 200
            title = self.path.strip("/").rsplit("/", 1)[-1]
            body = LEAVING if home and status == 200 else f"<title>{title}</title>".encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), Page)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


async def tabs_of(deployed, app: str = "chromium") -> list[dict]:
    """The open tabs of the browser ``app`` in ``deployed``, as its ``browser/v1`` ``tabs/list`` gives them."""
    async with httpx.AsyncClient(timeout=30, headers=deployed.control_headers) as http:
        card = (await http.get(f"{deployed.environment_url}/svc/{app}{WELL_KNOWN_PATH}")).json()
        endpoint = protocol_v1.find_extension_method(card, BROWSER, "tabs/list")["endpoint"]
        answer = await http.get(f"{deployed.environment_url}/svc/{app}{endpoint}")
    answer.raise_for_status()
    return answer.json()["tabs"]
