"""CDP addresses scoped to leases on a deployed chromium: steps lease buyer for ada and seller for bob, each signed in
by its own cookie with a tab open; ada's address scoped to buyer sees only buyer's targets, the cross-origin iframe
of a page there among them as its own target, and can't attach to seller's; bob isn't given an address to buyer;
through the real clients, Playwright on buyer's address finds the lease as its default context, its cookie and tabs,
and clicks inside the iframe and downloads a file from it, which the lease lists, Puppeteer on seller's finds seller
alone, and Playwright on the default address none of the leases' tabs; both leases' agents click in their iframes at
once, both getting there; and the lease released closes its connections, saying so, and refuses its address after."""

import asyncio
import itertools
import json
import subprocess
import time
from pathlib import Path
from typing import Callable

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from agent_env.env.envs.computer.utils import inside
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from tst.integration.computer.browser_contexts import _caller

SITE = "http://127.0.0.1:8791/"
FRAME = "http://127.0.0.2:8792/frame"
SERVE_S, WAIT_S, POLL_S = 900, 90.0, 0.5
# Serves, on two loopback addresses for the seconds it's given, a page titled "frames" whose iframe is the other
# address's /frame: a button that counts its presses in its body's data-clicked, and a link to /file, a download.
SERVE_FRAMES = r"""import http.server, sys, threading, time
PAGE = b"<title>frames</title><iframe src='http://127.0.0.2:8792/frame' width='400' height='200'></iframe>"
FRAME = (b"<title>frame</title><body data-clicked='0'><button id='press' onclick=\"document.body.dataset.clicked = "
         b"String(Number(document.body.dataset.clicked) + 1)\">press</button><a id='get' href='/file'>get</a></body>")
class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/file"):
            body = b"a receipt from the iframe\n"
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", "attachment; filename=receipt.txt")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = FRAME if self.path.startswith("/frame") else PAGE
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
for host, port in (("127.0.0.1", 8791), ("127.0.0.2", 8792)):
    server = http.server.ThreadingHTTPServer((host, port), Page)
    threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[1]))
"""


async def _raw(ws_url: str, commands: list[tuple[str, dict]]) -> list[dict]:
    """Each of ``commands`` sent in order over one CDP connection at ``ws_url``; each answer."""
    answers = []
    async with connect(ws_url, max_size=None, open_timeout=60) as ws:
        ids = itertools.count(1)
        for method, params in commands:
            message_id = next(ids)
            await ws.send(json.dumps({"id": message_id, "method": method, "params": params}))
            while (reply := json.loads(await asyncio.wait_for(ws.recv(), 60))).get("id") != message_id:
                pass
            answers.append(reply)
    return answers


async def _until(read, holds, what: str):
    last, deadline = None, time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        last = await read()
        if holds(last):
            return last
        await asyncio.sleep(POLL_S)
    raise AssertionError(f"{what} didn't happen within {WAIT_S:g} s: {last!r}")


async def check_scoped_addresses(deployed, env_id: str, op_url: Callable[[str], str], clients: Path) -> None:
    """The scoped addresses, as the module says, through browser/v1 at ``op_url(op)`` and the Node clients installed
    in ``clients``."""
    call = _caller(deployed, op_url)
    context = TaskStepContext(deployed_envs=[deployed])

    def step(type: str, suffix: str, **fields):
        return get_task_step_registry()[type].from_dict({"id": f"{type}-{suffix}", "version": None, "type": type,
                                                          "env_id": env_id, **fields})

    async def cdp(user: str, label: str | None, status: int = 200) -> dict:
        return await call("cdp", {"user": user, **({"context": label} if label else {})}, status=status)

    serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_FRAMES, str(SERVE_S)],
                                               timeout_s=SERVE_S + 60))
    try:
        await asyncio.sleep(3)
        for label, user in (("buyer", "ada"), ("seller", "bob")):
            context = await step("browser_lease_context", label, label=label, grants=[user]).execute(context)
            context = await step("browser_set_cookies", label, context=label, cookies=[
                {"name": "sid", "value": label, "url": SITE, "max_age": 3600}]).execute(context)
            context = await step("browser_open_tab", label, context=label, url=SITE).execute(context)
        leases = {lease["context"]: lease["id"] for lease in (await call("contexts/list", method="GET"))["contexts"]}
        seller_tab = (await call("tabs/list", {"context": "seller"}, method="GET"))["tabs"][0]["id"]
        buyer, seller, default = await cdp("ada", "buyer"), await cdp("bob", "seller"), await cdp("carol", None)
        refused = await cdp("bob", "buyer", status=403)

        async def buyer_targets() -> list[dict]:
            [listed] = await _raw(buyer["ws_url"], [("Target.getTargets", {})])
            return listed["result"]["targetInfos"]

        seen = await _until(buyer_targets, lambda infos: any(info["type"] == "iframe" for info in infos),
                            "the buyer's cross-origin iframe")
        [attached] = await _raw(buyer["ws_url"], [("Target.attachToTarget", {"targetId": seller_tab,
                                                                             "flatten": True})])
        ran = subprocess.run(["node", "scoped.mjs", buyer["ws_url"], seller["ws_url"], default["ws_url"], SITE],
                             cwd=clients, capture_output=True, text=True, timeout=600)
        assert ran.returncode == 0, ran.stderr[-2000:]
        clients_saw = json.loads(ran.stdout.strip().splitlines()[-1])
        buyer_downloads = await _until(lambda: call("downloads/list", {"context": "buyer"}, method="GET"),
                                       lambda found: any(d["filename"] == "receipt.txt" and d["state"] == "completed"
                                                         for d in found["downloads"]), "the iframe's download")

        async with connect(buyer["ws_url"], max_size=None, open_timeout=60) as held:
            await call("contexts/release", {"context": "buyer"})
            with pytest.raises(ConnectionClosed) as closed:
                await asyncio.wait_for(held.recv(), 30)
        with pytest.raises(InvalidStatus):
            await _raw(buyer["ws_url"], [("Browser.getVersion", {})])
    finally:
        serving.cancel()

    assert "permission_denied" in json.dumps(refused), refused
    assert {info["browserContextId"] for info in seen} == {leases["buyer"]}, seen
    assert attached["error"]["message"] == "No target with given id found", attached
    by_buyer, by_seller, by_default = (clients_saw[key].get("ok") for key in ("buyer", "seller", "default"))
    assert by_buyer and by_seller and by_default, clients_saw
    assert by_buyer["contexts"] == 1 and by_buyer["cookies"] == ["sid=buyer"], by_buyer
    assert set(by_buyer["pages"]) == {SITE} and by_buyer["clicked"] == "1" and by_buyer["frame_url"] == FRAME
    assert by_buyer["download"] == "receipt.txt", by_buyer
    assert all(d["context"] == "buyer" for d in buyer_downloads["downloads"]), buyer_downloads
    assert by_seller["contexts"] == 1 and by_seller["pages"] == [SITE] and by_seller["cookies"] == ["sid=seller"]
    assert SITE not in by_default["pages"], by_default
    at_once = clients_saw["at_once"]
    assert at_once["buyer"].get("ok", {}).get("clicked") == "10", at_once
    assert at_once["seller"].get("ok", {}).get("clicked") == "10", at_once
    assert (closed.value.rcvd.code, closed.value.rcvd.reason) == (1001, "the lease ended: ask for an address again")
