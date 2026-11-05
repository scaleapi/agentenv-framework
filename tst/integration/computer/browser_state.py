"""browser/v1's typed operations on a deployed chromium, inside a computer or alone: cookies set with a max_age, which
the site gets and the browser expires an hour from the computer's clock, read for the site's URL and deleted by name;
local storage written through a page agentenv serves, so the site sees no request, read for every origin and cleared;
a download with its file; and an operation refused without the control token."""

import asyncio
import base64
import json
import time
from typing import Callable

import httpx

from agent_env.env.envs.computer.utils import inside

SITE = "http://127.0.0.1:8779/"
ORIGIN = SITE.rstrip("/")
#: How long the site serves, an effect may take to show, how often it's asked meanwhile, and how far the cookie's
#: expiry may be from an hour after the computer's clock read before it was set.
SERVE_S, WAIT_S, POLL_S, EXPIRY_SLACK_S = 600, 60.0, 0.5, 120
SERVED = "/tmp/browser-state-served.json"
# Serves, on loopback for the seconds it's given, a page whose title is the Cookie header it got, its document.cookie
# and its local storage, and report.csv as a download; each path it's asked for written to SERVED.
SERVE_STATE = f"""import http.server, json, sys, threading, time
served = []
class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        served.append(self.path)
        with open({SERVED!r}, "w") as file:
            json.dump(served, file)
        if self.path == "/report.csv":
            body, kind = b"a,b\\n1,2\\n", "text/csv"
        else:
            body = ("<title>loading</title><script>document.title = JSON.stringify({{header: %s, "
                    "cookie: document.cookie, storage: Object.assign({{}}, localStorage)}})</script>"
                    % json.dumps(self.headers.get("Cookie", ""))).encode()
            kind = "text/html"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        if self.path == "/report.csv":
            self.send_header("Content-Disposition", "attachment; filename=report.csv")
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 8779), Page)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[1]))
"""


async def check_browser_state(deployed, endpoint: Callable[[str], str]) -> None:
    """Every cookie, local storage and download operation of ``browser/v1`` on the deployed chromium, each at
    ``endpoint(op)``, as the module says."""
    async def call(op: str, params: dict | None = None, *, method: str = "POST", headers=None) -> httpx.Response:
        headers = deployed.control_headers if headers is None else headers
        async with httpx.AsyncClient(timeout=60) as http:
            if method == "GET":
                return await http.get(endpoint(op), headers=headers)
            return await http.post(endpoint(op), json=params or {}, headers=headers)

    async def ok(op: str, params: dict | None = None, *, method: str = "POST") -> dict:
        response = await call(op, params, method=method)
        assert response.status_code == 200, f"{op}: {response.status_code} {response.text}"
        return response.json()

    async def page_at(url: str) -> dict:
        await ok("navigate", {"url": url})
        last, deadline = None, time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            tabs = (await ok("tabs/list", method="GET"))["tabs"]
            active = next((tab for tab in tabs if tab["active"]), None)
            if active and active["url"] == url and active["title"].startswith("{"):
                return json.loads(active["title"])
            last = tabs
            await asyncio.sleep(POLL_S)
        raise AssertionError(f"the page at {url} didn't load within {WAIT_S:g} s; the tabs were {last!r}")

    async def served() -> list[str]:
        out = await inside.run(deployed, ["sh", "-c", f"cat {SERVED} 2>/dev/null || echo []"], timeout_s=30)
        return json.loads(out["stdout"])

    serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_STATE, str(SERVE_S)],
                                               timeout_s=SERVE_S + 60))
    try:
        await asyncio.sleep(3)
        refused = await call("cookies/get", {}, headers={})
        now = float((await inside.run(deployed, ["date", "+%s"], timeout_s=30))["stdout"])
        await ok("cookies/set", {"cookies": [{"name": "sid", "value": "v1", "url": SITE, "max_age": 3600},
                                             {"name": "pref", "value": "dark", "domain": "127.0.0.1",
                                              "httpOnly": True}]})
        cookies = {cookie["name"]: cookie for cookie in (await ok("cookies/get", {"urls": [SITE]}))["cookies"]}
        await ok("local_storage/set", {"origins": [{"origin": ORIGIN, "localStorage": [{"name": "cart", "value": "3"}]}]})
        before_visit = await served()
        signed_in = await page_at(SITE)
        every_origin = (await ok("local_storage/get"))["origins"]
        deleted = (await ok("cookies/delete", {"name": "sid"}))["deleted"]
        left = [cookie["name"] for cookie in (await ok("cookies/get", {"urls": [SITE]}))["cookies"]]
        cleared = (await ok("local_storage/clear", {"origins": [ORIGIN]}))["origins"]
        after_clear = (await ok("local_storage/get", {"origins": [ORIGIN]}))["origins"]
        signed_out = await page_at(f"{SITE}?again")
        await ok("navigate", {"url": f"{SITE}report.csv"})
        download, deadline = None, time.monotonic() + WAIT_S
        while time.monotonic() < deadline and not (download and download["sha256"]):
            download = next((d for d in (await ok("downloads/list", method="GET"))["downloads"]
                             if d["filename"] == "report.csv"), None)
            await asyncio.sleep(POLL_S)
        got = await ok("downloads/get", {"guid": download["guid"]}) if download else None
    finally:
        serving.cancel()

    assert refused.status_code in (401, 403), (refused.status_code, refused.text)
    assert abs(cookies["sid"]["expires"] - (now + 3600)) < EXPIRY_SLACK_S, (cookies["sid"], now)
    assert (cookies["pref"]["domain"], cookies["pref"]["httpOnly"], cookies["pref"]["expires"]) == ("127.0.0.1", True, -1)
    assert before_visit == [], before_visit
    assert "sid=v1" in signed_in["header"] and "pref=dark" in signed_in["header"], signed_in
    assert signed_in["cookie"] == "sid=v1" and signed_in["storage"] == {"cart": "3"}, signed_in
    assert {"origin": ORIGIN, "localStorage": [{"name": "cart", "value": "3"}]} in every_origin, every_origin
    assert (deleted, left, cleared) == (1, ["pref"], [ORIGIN])
    assert after_clear == [{"origin": ORIGIN, "localStorage": []}], after_clear
    assert "sid" not in signed_out["header"] and signed_out["storage"] == {}, signed_out
    assert got is not None and got["download"]["state"] == "completed", download
    assert base64.b64decode(got["parts"][0]["file"]["bytes"]) == b"a,b\n1,2\n", got
