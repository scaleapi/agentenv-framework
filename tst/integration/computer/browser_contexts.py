"""browser/v1's contexts on a deployed chromium, inside a computer or alone: two leases made by task steps each sign in
to a loopback site as its own user, by cookies and local storage of its own, in tabs of its own the default context
doesn't list; five leases of one label at once are one; a lease's download lands in its own folder, listed in it alone;
a context an agent makes over CDP is leased to it as agent-<user>-<n>, another agent and the task's lease can't be
disposed by it, its own can; data/get carries each lease, and once agents have acted a load still leases a missing one
back with its cookies; and a release, then a reset, end them and delete their folders."""

import asyncio
import base64
import itertools
import json
import time
from typing import Callable

import httpx
from websockets.asyncio.client import connect

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.types import DataPart, uploaded_object_path

from agent_env.env.envs.computer.utils import inside
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep
from tst.integration.computer.browser_data import ORIGIN, POLL_S, SERVE_S, SERVE_SITE, SITE, WAIT_S

#: A loopback server, for the seconds it's given, whose /file is a download.
FILE_SITE = "http://127.0.0.1:8786/file"
SERVE_FILE = r"""import http.server, sys, threading, time
class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"invoice for the buyer\n"
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", "attachment; filename=invoice.txt")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 8786), Page)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[1]))
"""


def _caller(deployed, op_url: Callable[[str], str]):
    async def call(op: str, params: dict | None = None, *, method: str = "POST", status: int = 200) -> dict:
        async with httpx.AsyncClient(timeout=120, headers=deployed.control_headers) as http:
            answer = await (http.get(op_url(op), params=params or {}) if method == "GET"
                            else http.post(op_url(op), json=params or {}))
        assert answer.status_code == status, f"{op}: {answer.status_code} {answer.text}"
        return answer.json()

    return call


async def _until(read, holds, what: str):
    last, deadline = None, time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        last = await read()
        if holds(last):
            return last
        await asyncio.sleep(POLL_S)
    raise AssertionError(f"{what} didn't happen within {WAIT_S:g} s: {last!r}")


async def _cdp_commands(ws_url: str, commands: list[tuple[str, dict]]) -> list[dict]:
    """Each of ``commands`` sent in order over one CDP connection at ``ws_url``, closed after; each answer."""
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


async def check_browser_contexts(deployed, env_id: str, op_url: Callable[[str], str], data_url: str,
                                 cdp: Callable[[str], object]) -> None:
    """The contexts, as the module says, read back through browser/v1 at ``op_url(op)``, its data plane at
    ``data_url``, and CDP addresses from ``cdp(user)``."""
    call = _caller(deployed, op_url)
    context = TaskStepContext(deployed_envs=[deployed])

    def step(type: str, **fields) -> TaskStep:
        made = get_task_step_registry()[type].from_dict({"id": type, "version": None, "type": type, "env_id": env_id,
                                                          **fields})
        assert made.config_problems() + made.preflight() == [], type
        return made

    async def tabs(label: str | None = None) -> list[dict]:
        return (await call("tabs/list", {"context": label} if label else None, method="GET"))["tabs"]

    async def titled(label: str, title: str) -> dict:
        found = await _until(lambda: tabs(label), lambda found: any(tab["title"] == title for tab in found),
                             f"a tab of {label} titled {title!r}")
        return next(tab for tab in found if tab["title"] == title)

    serving = [asyncio.ensure_future(inside.run(deployed, ["python3", "-c", script, str(SERVE_S)],
                                                timeout_s=SERVE_S + 60)) for script in (SERVE_SITE, SERVE_FILE)]
    try:
        await asyncio.sleep(3)
        default_before = [tab["id"] for tab in await tabs()]
        for label, user, theme in (("buyer", "ada", "dark"), ("seller", "bob", None)):
            context = await step("browser_lease_context", label=label, grants=[user]).execute(context)
            context = await step("browser_set_cookies", context=label, cookies=[
                {"name": "sid", "value": label, "url": SITE, "max_age": 3600}]).execute(context)
            if theme:
                context = await step("browser_set_local_storage", context=label, origin=ORIGIN,
                                     items={"theme": theme}).execute(context)
            context = await step("browser_open_tab", context=label, url=SITE).execute(context)
        buyer = await titled("buyer", "signed in as buyer theme dark")
        seller = await titled("seller", "signed in as seller theme none")
        default_after = [tab["id"] for tab in await tabs()]
        default_cookies = (await call("cookies/get", {"urls": [SITE]}))["cookies"]
        raced = await asyncio.gather(*(call("contexts/lease", {"label": "race"}) for _ in range(5)))

        await call("tabs/open", {"url": FILE_SITE, "context": "buyer"})
        landed = await _until(lambda: call("downloads/list", {"context": "buyer"}, method="GET"),
                              lambda found: any(d["state"] == "completed" and d.get("size") for d in found["downloads"]),
                              "the buyer's download")
        default_downloads = (await call("downloads/list", method="GET"))["downloads"]
        on_disk = (await inside.run(deployed, ["sh", "-c", "cat ~/Contexts/buyer/Downloads/invoice.txt; ls ~/Downloads"],
                                    timeout_s=30))["stdout"]

        ada, bob = (await cdp("ada"))["ws_url"], (await cdp("bob"))["ws_url"]
        [made] = await _cdp_commands(ada, [("Target.createBrowserContext", {})])
        agents = made["result"]["browserContextId"]
        listed = (await call("contexts/list", method="GET"))["contexts"]
        buyer_id = next(lease["id"] for lease in listed if lease["context"] == "buyer")
        [theirs] = await _cdp_commands(bob, [("Target.disposeBrowserContext", {"browserContextId": agents})])
        tasks, own = await _cdp_commands(ada, [("Target.disposeBrowserContext", {"browserContextId": buyer_id}),
                                               ("Target.disposeBrowserContext", {"browserContextId": agents})])
        after_dispose = [lease["context"] for lease in (await call("contexts/list", method="GET"))["contexts"]]

        state = await _state(deployed, data_url)
        context = await step("browser_release_context", label="seller").execute(context)
        released = context.metadata["browser_steps"]["browser_release_context"]["released"]
        again = await step("browser_release_context", label="seller").execute(context)
        seller_folder = (await inside.run(deployed, ["sh", "-c", "test -e ~/Contexts/seller && echo kept || echo gone"],
                                          timeout_s=30))["stdout"].strip()
        await protocol_v1.add_data(data_url, [DataPart(data={"contexts": {"seller": {
            "cookies": [{"name": "sid", "value": "seller", "url": SITE}]}}})], timeout=120,
            headers=deployed.control_headers)
        await call("tabs/open", {"url": SITE, "context": "seller"})
        reloaded = await titled("seller", "signed in as seller theme none")
    finally:
        for task in serving:
            task.cancel()
    await protocol_v1.reset_data(data_url, timeout=300, headers=deployed.control_headers)
    after_reset = (await call("contexts/list", method="GET"))["contexts"]
    folders_after_reset = (await inside.run(deployed, ["sh", "-c", "ls ~/Contexts 2>/dev/null | wc -l"],
                                            timeout_s=30))["stdout"].strip()
    refused = await call("tabs/list", {"context": "buyer"}, method="GET", status=404)

    assert buyer["context"] == "buyer" and seller["context"] == "seller", (buyer, seller)
    assert default_after == default_before and default_cookies == [], (default_before, default_after, default_cookies)
    assert len({answer["id"] for answer in raced}) == 1, raced
    [download] = [d for d in landed["downloads"] if d["state"] == "completed"]
    assert download["path"].endswith("/Contexts/buyer/Downloads/invoice.txt") and download["context"] == "buyer"
    assert default_downloads == [] and on_disk.startswith("invoice for the buyer"), (default_downloads, on_disk)
    assert {"context": "agent-ada-1", "id": agents, "made_by": "ada", "grants": ["ada"]} in listed, listed
    assert {"context": "buyer", "id": buyer_id, "made_by": "task", "grants": ["ada"]} in listed, listed
    assert "is another agent's" in theirs["error"]["message"] and "is the task's" in tasks["error"]["message"]
    assert "result" in own and sorted(after_dispose) == ["buyer", "race", "seller"], (own, after_dispose)
    assert state["contexts"]["buyer"]["cookies"][0]["value"] == "buyer", state["contexts"]
    assert any(path.startswith("Contexts/buyer/Downloads/") for path in state["files"]), state["files"]
    assert released is True and again.metadata["browser_steps"]["browser_release_context"]["released"] is False
    assert seller_folder == "gone" and reloaded["context"] == "seller", (seller_folder, reloaded)
    assert after_reset == [] and folders_after_reset == "0", (after_reset, folders_after_reset)
    assert "target_unavailable" in json.dumps(refused), refused


async def _state(deployed, data_url: str) -> dict:
    """The browser's data/get state, with the paths of the files it returned beside it."""
    answer = await protocol_v1.get_data(data_url, timeout=300, headers=deployed.control_headers)
    paths = [uploaded_object_path(part) for part in answer.parts]
    [state] = [json.loads(base64.b64decode(part.file.bytes)) for part, path in zip(answer.parts, paths)
               if path == "browser-state.json"]
    return {**state, "files": [path for path in paths if path]}
