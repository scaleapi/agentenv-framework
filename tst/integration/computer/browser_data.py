"""The website browser's data plane on a deployed chromium, inside a computer or alone, through agentenv's own steps:
load_artifact signs it in to a loopback site, its start tab loaded again signed in, and a load's tabs take the open
ones' place; snapshot_env keeps its state as browser-state.json, a tab closed during the run and its profile among its
files; once an agent has acted, a load can't change its tabs but sets its cookies; data/reset puts it back signed out at
its start tab with its addresses ended; the snapshot taken mid-run loads back whole, its cookies, local storage,
tabs and history; and reset_env, a reboot, brings back its start tab with no cookies or local storage. Under chaos: data calls at once each get their answer, its server killed and back takes a load, and
with chromium killed a snapshot fails at once rather than wait."""

import asyncio
import base64
import json
import time
import uuid
from typing import Awaitable, Callable

import httpx

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.types import DataPart, uploaded_object_path

from agent_env.artifact import EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.env.envs.computer.utils import inside
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from agent_env.task_step.task_steps.snapshot_env import SnapshotEnvTaskStep

SITE = "http://127.0.0.1:8783/"
ORIGIN = SITE.rstrip("/")
PROFILE = ".config/chromium"
#: How long the site serves, an effect may take to show and how often it's asked meanwhile.
SERVE_S, WAIT_S, POLL_S = 900, 90.0, 0.5
# Serves, on loopback for the seconds it's given: / as "signed in as <sid> theme <its local storage's theme>" for a
# request with the sid cookie, else a redirect to /login, titled "login"; and /other, titled "other".
SERVE_SITE = r"""import http.cookies, http.server, json, sys, threading, time
class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        sid = http.cookies.SimpleCookie(self.headers.get("Cookie", "")).get("sid")
        if self.path == "/" and sid is None:
            self.send_response(302)
            self.send_header("Location", "/login")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/":
            body = ("<title>loading</title><script>document.title = 'signed in as ' + %s + ' theme ' + "
                    "(localStorage.getItem('theme') || 'none')</script>" % json.dumps(sid.value)).encode()
        else:
            body = ("<title>%s</title>" % self.path.strip("/")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 8783), Page)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[1]))
"""
# The URLs in the browser's history, read from a copy of its History database and its write-ahead log as they are on
# disk, which the running browser writes some seconds after each visit.
HISTORY = (f"rm -rf /tmp/history && mkdir /tmp/history && cp ~/{PROFILE}/Default/History ~/{PROFILE}/Default/History-wal "
           "/tmp/history/ 2>/dev/null; python3 -c \"import sqlite3; print([row[0] for row in "
           "sqlite3.connect('/tmp/history/History').execute('select url from urls')])\" 2>&1")


def _client(deployed, op_url: Callable[[str], str]):
    async def call(op: str, params: dict | None = None, *, method: str = "POST") -> dict:
        async with httpx.AsyncClient(timeout=120, headers=deployed.control_headers) as http:
            answer = await (http.get(op_url(op)) if method == "GET" else http.post(op_url(op), json=params or {}))
        assert answer.status_code == 200, f"{op}: {answer.status_code} {answer.text}"
        return answer.json()

    async def titled(title: str) -> dict:
        last, deadline = None, time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            tabs = (await call("tabs/list", method="GET"))["tabs"]
            if (active := next((tab for tab in tabs if tab["active"]), None)) and active["title"] == title:
                return active
            last = tabs
            await asyncio.sleep(POLL_S)
        raise AssertionError(f"the active tab wasn't {title!r} within {WAIT_S:g} s; the tabs were {last!r}")

    return call, titled


def _universe(env_id: str, files: dict[str, bytes]) -> EnvironmentUniverseArtifact:
    """An environment universe whose chromium environment holds ``files``, by their path."""
    run = uuid.uuid4().hex[:8]
    environment = EnvironmentArtifact.put(id=f"{env_id}-browser-{run}", environment_name="chromium", file_artifacts={
        path: FileArtifact.put_bytes(id=f"{env_id}-browser-{run}-{index}", description="browser data e2e",
                                     filename=path.rsplit("/", 1)[-1], content=content,
                                     content_type="application/octet-stream")
        for index, (path, content) in enumerate(files.items())})
    return EnvironmentUniverseArtifact.put(id=f"{env_id}-universe-{run}", environment_artifacts=[environment])


def _state(**fields) -> bytes:
    return json.dumps(fields).encode()


def _snapshot_files(context: TaskStepContext, step: str) -> dict[str, bytes]:
    kept = context.metadata["env_snapshotted_universes"][step]
    universe = EnvironmentUniverseArtifact.get(kept["id"], version=kept["version"])
    (browser,) = [environment for environment in universe.get_environment_artifacts()
                  if environment.environment_name == "chromium"]
    return {path: file.load() for path, file in browser.get_file_artifacts().items()}


async def _load(context: TaskStepContext, env_id: str, step: str, **source) -> TaskStepContext:
    return await LoadArtifactTaskStep(id=step, version=None, env_id=env_id, **source).execute(context)


async def check_browser_data(deployed, env_id: str, op_url: Callable[[str], str],
                             act: Callable[[], Awaitable]) -> None:
    """The data plane through agentenv's own steps, as the module says; ``act`` is an agent's action in the
    computer."""
    call, titled = _client(deployed, op_url)
    context = TaskStepContext(deployed_envs=[deployed])
    serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_SITE, str(SERVE_S)],
                                               timeout_s=SERVE_S + 60))
    try:
        await asyncio.sleep(3)
        await call("navigate", {"url": SITE})
        signed_out = await titled("login")
        seed = _universe(env_id, {"browser-state.json": _state(
            cookies=[{"name": "sid", "value": "ada", "url": SITE, "max_age": 3600}],
            local_storage=[{"origin": ORIGIN, "localStorage": [{"name": "theme", "value": "dark"}]}]),
            "Uploads/cv.txt": b"cv"})
        context = await _load(context, env_id, "seed", artifact_id=seed.id)
        signed_in = await titled("signed in as ada theme dark")
        uploaded = (await inside.run(deployed, ["cat", "Uploads/cv.txt"], timeout_s=30))["stdout"]
        other = (await call("tabs/open", {"url": f"{SITE}other"}))["tab"]
        await call("tabs/close", {"tab": other})
        await _until_in_history(deployed, f"{SITE}other")
        context = await SnapshotEnvTaskStep(id="mid", version=None, env_id=env_id,
                                            snapshot_id=f"{env_id}-mid-{uuid.uuid4().hex[:8]}").execute(context)
        mid = _snapshot_files(context, "mid")
        tabs = _universe(env_id, {"browser-state.json": _state(tabs=[{"url": f"{SITE}other"},
                                                                     {"url": SITE, "active": True}])})
        context = await _load(context, env_id, "tabs", artifact_id=tabs.id)
        await titled("signed in as ada theme dark")
        replaced = [(tab["url"], tab["active"]) for tab in (await call("tabs/list", method="GET"))["tabs"]]
        await act()
        try:
            await _load(context, env_id, "tabs-after-agent", artifact_id=tabs.id)
            refused = None
        except Exception as e:
            refused = str(e)
        bob = _universe(env_id, {"browser-state.json": _state(
            cookies=[{"name": "sid", "value": "bob", "url": SITE, "max_age": 3600}])})
        context = await _load(context, env_id, "cookies-after-agent", artifact_id=bob.id)
        kept_page = (await titled("signed in as ada theme dark"))["title"]
        cookies_after_agent = [c["value"] for c in (await call("cookies/get", {"urls": [SITE]}))["cookies"]]
        address = await call("cdp", {"user": "agent"})
        await protocol_v1.reset_data(f"{deployed.environment_url}/svc/chromium", headers=deployed.control_headers)
        reset_page = await titled("login")
        reset_cookies = (await call("cookies/get", {"urls": [SITE]}))["cookies"]
        async with httpx.AsyncClient(timeout=30) as http:
            ended = (await http.get(f"{address['http_url']}json/version")).status_code
        context = await _load(context, env_id, "restore", artifact_from_step_id="mid")
        restored_page = await titled("signed in as ada theme dark")
        restored_tabs = [tab["url"] for tab in (await call("tabs/list", method="GET"))["tabs"]]
        restored_storage = (await call("local_storage/get", {"origins": [ORIGIN]}))["origins"]
        history = await _history(deployed)
        context = await ResetEnvTaskStep(id="reset", version=None, env_id=env_id,
                                         fail_task_on_error=True).execute(context)
        after_reset_tabs, deadline = None, time.monotonic() + WAIT_S
        while after_reset_tabs != [SITE] and time.monotonic() < deadline:
            after_reset_tabs = [tab["url"] for tab in (await call("tabs/list", method="GET"))["tabs"]]
            await asyncio.sleep(POLL_S)
        after_reset_cookies = (await call("cookies/get", {"urls": [SITE]}))["cookies"]
        after_reset_storage = (await call("local_storage/get", {"origins": [ORIGIN]}))["origins"]
    finally:
        serving.cancel()

    assert signed_out["url"].endswith("/login") and signed_in["url"] == SITE, (signed_out, signed_in)
    assert uploaded == "cv"
    state = json.loads(mid["browser-state.json"])
    assert [c["value"] for c in state["cookies"] if c["name"] == "sid"] == ["ada"], state["cookies"]
    assert {"origin": ORIGIN, "localStorage": [{"name": "theme", "value": "dark"}]} in state["local_storage"]
    assert [t["url"] for t in state["tabs"]] == [SITE] and f"{SITE}other" in [t["url"] for t in state["closed_tabs"]]
    assert state["clock"]["mode"] == "real" and f"{PROFILE}/Default/Preferences" in mid, sorted(mid)[:20]
    assert not [path for path in mid if "/Cache/" in path or path.endswith("SingletonLock")]
    assert replaced == [(f"{SITE}other", False), (SITE, True)], replaced
    assert refused is not None and "an agent has acted" in refused, refused
    assert kept_page == "signed in as ada theme dark" and cookies_after_agent == ["bob"]
    assert reset_page["url"].endswith("/login") and reset_cookies == [] and ended == 403
    assert restored_page["url"] == SITE and restored_tabs == [SITE], restored_tabs
    assert restored_storage == [{"origin": ORIGIN, "localStorage": [{"name": "theme", "value": "dark"}]}]
    assert f"{SITE}other" in history, history
    assert after_reset_tabs == [SITE] and after_reset_cookies == [], (after_reset_tabs, after_reset_cookies)
    assert after_reset_storage == [{"origin": ORIGIN, "localStorage": []}], after_reset_storage


async def _history(deployed) -> str:
    return (await inside.run(deployed, ["sh", "-c", HISTORY], timeout_s=60))["stdout"]


async def _until_in_history(deployed, url: str) -> None:
    history, deadline = "", time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        if url in (history := await _history(deployed)):
            return
        await asyncio.sleep(2)
    raise AssertionError(f"{url} wasn't in the browser's history within {WAIT_S:g} s: {history}")


async def check_browser_data_under_chaos(deployed, kill: Callable[[str], Awaitable],
                                         ready: Callable[[], Awaitable]) -> None:
    """The chaos the module names; ``kill(pattern)`` kills a process of the computer matching it, the oldest for the
    browser, and ``ready()`` waits for the computer to be ready."""
    base, headers = f"{deployed.environment_url}/svc/chromium", deployed.control_headers

    def cookie(value: str) -> list:
        return [DataPart(data={"cookies": [{"name": "c", "value": value, "url": SITE, "max_age": 600}]})]

    at_once = await asyncio.gather(protocol_v1.get_data(base, timeout=120, headers=headers),
                                   protocol_v1.add_data(base, cookie("1"), headers=headers),
                                   protocol_v1.get_data(base, timeout=120, headers=headers),
                                   protocol_v1.add_data(base, cookie("2"), headers=headers))
    await kill("[s]erve_app")
    await ready()
    await protocol_v1.add_data(base, cookie("3"), headers=headers)
    after = await protocol_v1.get_data(base, timeout=120, headers=headers)
    await kill("[-]-remote-debugging-pipe")
    started = time.monotonic()
    try:
        await protocol_v1.get_data(base, timeout=120, headers=headers)
        failed = None
    except Exception as e:
        failed = type(e).__name__
    failed_s = time.monotonic() - started

    for answer in (at_once[0], at_once[2], after):
        assert "browser-state.json" in [uploaded_object_path(part) for part in answer.parts]
    state = next(json.loads(base64.b64decode(part.file.bytes)) for part in after.parts
                 if uploaded_object_path(part) == "browser-state.json")
    assert [c["value"] for c in state["cookies"] if c["name"] == "c"] == ["3"], state["cookies"]
    assert failed is not None and failed_s < 60, (failed, failed_s)
