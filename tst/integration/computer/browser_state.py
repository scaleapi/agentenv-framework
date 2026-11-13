"""browser/v1's typed operations on a deployed chromium, inside a computer or alone: cookies set with a max_age, which
the site gets and the browser expires an hour from the computer's clock, read for the site's URL and deleted by name;
local storage written through a page agentenv serves, so the site sees no request, read for every origin and cleared;
a partitioned cookie read with its key and deleted beside an unpartitioned one of its name; a site's service worker, which answers its pages, kept from agentenv's;
a download with its file; and an operation refused without the control token. Under chaos: calls at once each get
their answer; the pages agentenv serves itself never show in the record; with the browser's server killed and back,
the operations work again; and with the browser killed, they fail at once rather than wait."""

import asyncio
import base64
import json
import time
from typing import Awaitable, Callable

import httpx

from agent_env.env.envs.computer.utils import inside

SITE = "http://127.0.0.1:8779/"
ORIGIN = SITE.rstrip("/")
#: How long the site serves, an effect may take to show, how often it's asked meanwhile, and how far the cookie's
#: expiry may be from an hour after the computer's clock read before it was set.
SERVE_S, WAIT_S, POLL_S, EXPIRY_SLACK_S = 600, 60.0, 0.5, 120
SERVED = "/tmp/browser-state-served.json"
#: A site on an origin of its own whose service worker answers every navigation with a page that writes to local
#: storage.
SW_SITE = "http://127.0.0.1:8781/"
SW_ORIGIN = SW_SITE.rstrip("/")
# Serves, on loopback for the seconds it's given: on 8779 a page whose title is the Cookie header it got, its
# document.cookie and its local storage, and /download, a page that goes to report.csv, a download; on 8781 a page that
# registers the service worker. Each port and path it's asked for is written to SERVED.
SERVE_STATE = r"""import http.server, json, sys, threading, time
served = []
WORKER = b"self.addEventListener('install', e => self.skipWaiting()); self.addEventListener('activate', e => e.waitUntil(clients.claim())); self.addEventListener('fetch', e => { if (e.request.mode === 'navigate') e.respondWith(new Response('<title>from-sw</title><script>localStorage.setItem(\"sw_ran\", \"yes\")</script>', {headers: {'Content-Type': 'text/html'}})); });"
class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        port = self.server.server_address[1]
        served.append(f"{port}{self.path}")
        with open("SERVED_PATH", "w") as file:
            json.dump(served, file)
        kind = "text/html"
        if port == 8781:
            body, kind = (WORKER, "text/javascript") if self.path == "/sw.js" else (
                b"<title>registering</title><script>navigator.serviceWorker.register('/sw.js').then(() => "
                b"navigator.serviceWorker.ready).then(() => document.title = 'registered')</script>", kind)
        elif self.path == "/report.csv":
            body, kind = b"a,b\n1,2\n", "text/csv"
        elif self.path == "/download":
            body = b"<title>downloading</title><script>location.href = '/report.csv'</script>"
        else:
            body = ("<title>loading</title><script>document.title = JSON.stringify({header: %s, "
                    "cookie: document.cookie, storage: Object.assign({}, localStorage)})</script>"
                    % json.dumps(self.headers.get("Cookie", ""))).encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        if self.path == "/report.csv":
            self.send_header("Content-Disposition", "attachment; filename=report.csv")
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
for port in (8779, 8781):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[1]))
""".replace("SERVED_PATH", SERVED)


def _client(deployed, endpoint: Callable[[str], str]):
    """``call``, an operation's response, and ``ok``, its result once it answered 200."""
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

    return call, ok


def _serve(deployed) -> asyncio.Future:
    return asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_STATE, str(SERVE_S)],
                                            timeout_s=SERVE_S + 60))


async def check_browser_state(deployed, endpoint: Callable[[str], str]) -> None:
    """Every cookie, local storage and download operation of ``browser/v1`` on the deployed chromium, each at
    ``endpoint(op)``, as the module says."""
    call, ok = _client(deployed, endpoint)

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

    async def titled(url: str, title: str) -> dict:
        await ok("navigate", {"url": url})
        last, deadline = None, time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            tabs = (await ok("tabs/list", method="GET"))["tabs"]
            if (active := next((tab for tab in tabs if tab["active"]), None)) and active["title"] == title:
                return active
            last = tabs
            await asyncio.sleep(POLL_S)
        raise AssertionError(f"the page at {url} wasn't {title!r} within {WAIT_S:g} s; the tabs were {last!r}")

    async def served() -> list[str]:
        out = await inside.run(deployed, ["sh", "-c", f"cat {SERVED} 2>/dev/null || echo []"], timeout_s=30)
        return json.loads(out["stdout"])

    serving = _serve(deployed)
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
        chip = {"name": "chip", "value": "p", "domain": "embed.test", "path": "/", "secure": True, "sameSite": "None"}
        key = {"topLevelSite": "https://top.test", "hasCrossSiteAncestor": True}
        await ok("cookies/set", {"cookies": [{**chip, "partitionKey": key}, {**chip, "value": "u"}]})
        chips = [c for c in (await ok("cookies/get"))["cookies"] if c["name"] == "chip"]
        chips_deleted = (await ok("cookies/delete", {"name": "chip"}))["deleted"]
        chips_left = [c for c in (await ok("cookies/get"))["cookies"] if c["name"] == "chip"]
        await titled(f"{SW_SITE}register", "registered")
        worker_site_before = [path for path in await served() if path.startswith("8781")]
        worker_bypassed = (await ok("local_storage/get", {"origins": [SW_ORIGIN]}))["origins"]
        worker_site_after = [path for path in await served() if path.startswith("8781")]
        await titled(f"{SW_SITE}any", "from-sw")
        worker_answered = (await ok("local_storage/get", {"origins": [SW_ORIGIN]}))["origins"]
        await ok("navigate", {"url": f"{SITE}download"})
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
    assert sorted((c["value"], c.get("partitionKey")) for c in chips) == sorted([("p", key), ("u", None)]), chips
    assert (chips_deleted, chips_left) == (2, []), chips_left
    assert worker_bypassed == [{"origin": SW_ORIGIN, "localStorage": []}], worker_bypassed
    assert worker_site_after == worker_site_before, (worker_site_before, worker_site_after)
    assert {"name": "sw_ran", "value": "yes"} in worker_answered[0]["localStorage"], worker_answered
    assert got is not None and got["download"]["state"] == "completed", download
    assert base64.b64decode(got["parts"][0]["file"]["bytes"]) == b"a,b\n1,2\n", got


async def check_browser_state_under_chaos(deployed, endpoint: Callable[[str], str], kill: Callable[[str], Awaitable],
                                          ready: Callable[[], Awaitable]) -> dict:
    """The chaos the module names, ``kill(pattern)`` killing the newest process matching it, or the oldest for the
    browser, and ``ready()`` waiting for the computer to be ready. Returns whether the cookie and the item set before
    the server was killed were still there after, which depends on what the browser had written to its profile."""
    call, ok = _client(deployed, endpoint)
    item = {"origin": ORIGIN, "localStorage": [{"name": "k", "value": "1"}]}
    serving = _serve(deployed)
    try:
        await asyncio.sleep(3)
        await ok("cookies/set", {"cookies": [{"name": "kept", "value": "1", "url": SITE, "max_age": 3600}]})
        await ok("local_storage/set", {"origins": [item]})
        at_once = await asyncio.gather(*[ok("local_storage/get", {"origins": [ORIGIN]}) for _ in range(4)],
                                       ok("cookies/get", {"urls": [SITE]}), ok("local_storage/set", {"origins": [
                                           {"origin": ORIGIN, "localStorage": [{"name": "k2", "value": "2"}]}]}),
                                       ok("local_storage/clear", {"origins": ["http://127.0.0.1:8780"]}))
        after = (await ok("local_storage/get", {"origins": [ORIGIN]}))["origins"]
        recorded = await ok("record", method="GET")
        await kill("[s]erve_app")
        await ready()
        cookies_back = [c["name"] for c in (await ok("cookies/get", {"urls": [SITE]}))["cookies"]]
        storage_back = (await ok("local_storage/get", {"origins": [ORIGIN]}))["origins"][0]["localStorage"]
        recorded_back = await ok("record", method="GET")
        await kill("[-]-remote-debugging-pipe")
        started = time.monotonic()
        down = [await call("cookies/get", {"urls": [SITE]}), await call("local_storage/get", {"origins": [ORIGIN]})]
        down_s = time.monotonic() - started
    finally:
        serving.cancel()

    for answer in at_once[:4]:
        assert {"name": "k", "value": "1"} in answer["origins"][0]["localStorage"], at_once
    assert "kept" in [c["name"] for c in at_once[4]["cookies"]], at_once[4]
    assert {"name": "k2", "value": "2"} in after[0]["localStorage"], after
    for record in (recorded, recorded_back):
        assert [tab["url"] for tab in record["tabs"]] == ["about:blank"] and record["closed_tabs"] == [], record
    assert all(answer.status_code >= 400 for answer in down) and down_s < 30, ([a.text for a in down], down_s)
    return {"cookie_kept": "kept" in cookies_back, "item_kept": {"name": "k", "value": "1"} in storage_back}
