"""Each context of a deployed chromium recorded on its own: leases buyer, for ada, and seller, for bob, each with a tab
of its own colour, recorded with the default context, whose tab is green, all at once, while their agents act through
addresses scoped to them, real Playwright and Puppeteer. Buyer's agent opens an orange popup and closes it, and
screencasts its own tab meanwhile; seller's makes a context of its own and opens a yellow page there; the default's
clicks its tab. Then buyer's lease is released. Every frame of each video is its own context's colour, buyer's follows
the popup and back, seller's recording holds the context its agent made in its colour, the HAR has every context's
loads, the default's frames are taken, and buyer's recording ends with its lease."""

import asyncio
import io
import json
import subprocess
import uuid
from pathlib import Path
from typing import Callable

import av
from websockets.asyncio.client import connect

from agent_env.artifact import Artifact
from agent_env.config import get_config
from agent_env.env.envs.computer.utils import inside
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.browser.recording import (RECORDINGS_KEY, EndBrowserRecordingTaskStep,
                                                              StartBrowserRecordingTaskStep)
from tst.integration.computer.browser_contexts import _caller

SITE = "http://127.0.0.1:8794/"
COLORS = {"green": (0, 255, 0), "red": (255, 0, 0), "orange": (255, 136, 0), "blue": (0, 0, 255),
          "yellow": (255, 255, 0), "white": (255, 255, 255)}
SERVE_S = 900
# Serves, for the seconds it's given, a page per colour at /<colour>, the whole page that colour, with a link that
# opens /orange in a new tab and a button that only retitles the page.
SERVE_COLORS = r"""import http.server, sys, threading, time
COLORS = {"green": "#00ff00", "red": "#ff0000", "orange": "#ff8800", "blue": "#0000ff", "yellow": "#ffff00"}
class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        name = self.path.strip("/")
        body = (f"<title>{name}</title><body style='margin:0;height:100vh;background:{COLORS.get(name, '#ffffff')}'>"
                "<a id='pop' href='/orange' target='_blank' style='position:fixed;top:4px;left:4px'>pop</a>"
                "<button id='here' style='position:fixed;top:4px;right:4px' "
                "onclick='document.title = String(Date.now())'>here</button></body>").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 8794), Page)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[1]))
"""


async def _screencast_own_tab(ws_url: str, seconds: float) -> int:
    """An agent's own screencast of its context's red tab, over its scoped address, for ``seconds``: its frames."""
    frames, ids = 0, iter(range(1, 1 << 30))
    async with connect(ws_url, max_size=None, open_timeout=60) as ws:
        async def call(method: str, params: dict, session: str | None = None) -> dict:
            message_id = next(ids)
            await ws.send(json.dumps({"id": message_id, "method": method, "params": params,
                                      **({"sessionId": session} if session else {})}))
            while (reply := json.loads(await asyncio.wait_for(ws.recv(), 60))).get("id") != message_id:
                pass
            return reply

        infos = (await call("Target.getTargets", {}))["result"]["targetInfos"]
        target = next(info["targetId"] for info in infos if info["url"].endswith("/red"))
        session = (await call("Target.attachToTarget", {"targetId": target, "flatten": True}))["result"]["sessionId"]
        await call("Page.startScreencast", {"format": "jpeg", "quality": 50}, session)
        deadline = asyncio.get_running_loop().time() + seconds
        while (left := deadline - asyncio.get_running_loop().time()) > 0:
            try:
                message = json.loads(await asyncio.wait_for(ws.recv(), left))
            except asyncio.TimeoutError:
                break
            if message.get("method") == "Page.screencastFrame":
                frames += 1
                await ws.send(json.dumps({"id": next(ids), "method": "Page.screencastFrameAck",
                                          "params": {"sessionId": message["params"]["sessionId"]},
                                          "sessionId": session}))
    return frames


def _colors(files: dict[str, bytes], prefix: str) -> list[str]:
    """The colour of every frame of the video whose segments are ``files`` named ``<prefix>video-<n>.mp4``."""
    found = []
    for name in sorted(n for n in files if n.startswith(f"{prefix}video-") and n.endswith(".mp4")):
        with av.open(io.BytesIO(files[name])) as container:
            for frame in container.decode(video=0):
                pixel = frame.to_image().getpixel((frame.width // 2, frame.height // 2))
                found.append(min(COLORS, key=lambda c: sum((a - b) ** 2 for a, b in zip(COLORS[c], pixel))))
    return found


def _registered(entry: dict) -> dict[str, bytes]:
    universe = Artifact.get(entry["artifact"]["id"], version=entry["artifact"]["version"])
    objects = get_config().get_object_store()
    return {name: objects.get(Artifact.get(artifact_id).object_url)
            for name, artifact_id in universe.file_artifact_ids.items()}


async def check_recorded_contexts(deployed, env_id: str, op_url: Callable[[str], str], clients: Path) -> None:
    """The recordings, as the module says, through browser/v1 at ``op_url(op)`` and the Node clients in ``clients``."""
    call = _caller(deployed, op_url)
    context = TaskStepContext(instance_id=f"run-{uuid.uuid4().hex[:8]}", deployed_envs=[deployed])

    def step(type: str, suffix: str, **fields):
        return get_task_step_registry()[type].from_dict({"id": f"{type}-{suffix}", "version": None, "type": type,
                                                          "env_id": env_id, **fields})

    async def cdp(user: str, label: str | None) -> dict:
        return await call("cdp", {"user": user, **({"context": label} if label else {})})

    serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_COLORS, str(SERVE_S)],
                                               timeout_s=SERVE_S + 60))
    try:
        await asyncio.sleep(3)
        for label, user, color in (("buyer", "ada", "red"), ("seller", "bob", "blue")):
            context = await step("browser_lease_context", label, label=label, grants=[user]).execute(context)
            context = await step("browser_open_tab", label, context=label, url=SITE + color).execute(context)
        context = await step("browser_open_tab", "default", url=SITE + "green").execute(context)
        for name, label in (("buyer", "buyer"), ("seller", "seller"), ("default", None)):
            context = await StartBrowserRecordingTaskStep(
                id=f"start-{name}", version=None, env_id=env_id, name=name, frames=name == "default",
                context=label).execute(context)
        buyer, seller, default = await cdp("ada", "buyer"), await cdp("bob", "seller"), await cdp("carol", None)
        casting = asyncio.ensure_future(_screencast_own_tab(buyer["ws_url"], 12))
        ran = await asyncio.to_thread(subprocess.run, ["node", "recorded.mjs", buyer["ws_url"], seller["ws_url"],
                                                       default["ws_url"], SITE],
                                      cwd=clients, capture_output=True, text=True, timeout=600)
        heard = await casting
        for name in ("seller", "default"):
            context = await EndBrowserRecordingTaskStep(id=f"end-{name}", version=None, name=name).execute(context)
        context = await step("browser_release_context", "buyer", label="buyer").execute(context)
        await asyncio.sleep(3)
        context = await EndBrowserRecordingTaskStep(id="end-buyer", version=None, name="buyer").execute(context)
    finally:
        serving.cancel()

    assert ran.returncode == 0, ran.stderr[-2000:]
    agents = json.loads(ran.stdout.strip().splitlines()[-1])
    assert all("ok" in agents[name] for name in ("buyer", "seller", "default")), agents
    entries = context.metadata[RECORDINGS_KEY]
    files = {name: _registered(entries[name]) for name in ("buyer", "seller", "default")}
    manifests = {name: json.loads(files[name]["manifest.json"]) for name in files}
    colors = {name: _colors(files[name], "") for name in files}
    child = manifests["seller"]["children"]
    child_prefix = child[0]["folder"].replace("/", "-") if child else ""
    har = json.loads(files["buyer"]["network.har"])
    loaded = {entry["request"]["url"] for entry in har["log"]["entries"]}

    print("recorded", {name: {color: colors[name].count(color) for color in set(colors[name])} for name in colors},
          {"followed": len(manifests["buyer"]["video"]["followed"]), "screencast_frames_heard": heard,
           "har_entries": len(har["log"]["entries"]), "child": [c["context"] for c in child]})
    assert set(colors["buyer"]) == {"red", "orange"} and colors["buyer"][-1] == "red", colors["buyer"]
    assert set(colors["seller"]) == {"blue"} and set(colors["default"]) == {"green"}, colors
    tabs = [seen["tab"] for seen in manifests["buyer"]["video"]["followed"]]
    followed = [tab for n, tab in enumerate(tabs) if n == 0 or tab != tabs[n - 1]]
    assert len(followed) == 3 and followed[0] == followed[2] != followed[1], manifests["buyer"]["video"]["followed"]
    assert [c["made_by"] for c in child] == ["bob"] and child[0]["missing"] is None, child
    assert set(_colors(files["seller"], child_prefix)) <= {"yellow", "white"}, child_prefix
    assert "yellow" in _colors(files["seller"], child_prefix)
    assert {SITE + color for color in ("green", "red", "blue", "orange", "yellow")} <= loaded, loaded
    assert manifests["buyer"]["network"]["scope"] == "browser" and manifests["buyer"]["context"] == "buyer"
    assert entries["default"]["frames_taken"] >= 3 and entries["default"]["missing_frames"] == 0
    assert entries["buyer"]["ended_by"] == "context_ended" and heard > 0, (entries["buyer"], heard)
    assert all(entries[name]["state"] == "ended" and entries[name]["segments"] for name in entries)
