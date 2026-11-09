"""The notes app composed onto the deployed Windows computer: ready once its server says its window is shown and its
@ready_check passes; and, read inside the computer as the session user, on the app's own venv: what its service wrote on
the deploy's settings, its secret resolved at boot; its window, listed by its program and title; its Start Menu
launcher and its launch since the computer booted; what its install left in its folder: the download, its PowerShell
run script's output, run elevated, and its install step's file on the build setting its env fixes; its own folder,
which the session user, SYSTEM and Administrators alone reach; and its server, which answers its token alone."""
import json

from agentenv_protocol.computer.card import COMPUTER_READY
from agentenv_protocol.computer.conformance.client import Driver

PYTHON = "C:/agentenv/apps/notes/venv/Scripts/python.exe"
FACTS = r"""
import asyncio, json, os, subprocess, tempfile, time
from pathlib import Path
import httpx
from agentenv_protocol.computer import session
from agentenv_protocol.computer.app import ADDRESS
from agentenv_protocol.computer.served import served

prefix = Path("C:/agentenv/apps/notes")
written = Path.home() / ".notes.json"
deadline = time.monotonic() + 60
while not written.exists() and time.monotonic() < deadline:
    time.sleep(0.5)
reached = served(prefix)
tokenless = httpx.get(f"http://127.0.0.1:{reached.port}/.well-known/agent-env.json", trust_env=False, timeout=10)
with reached.sync_client(10) as http:
    answered = http.get("/.well-known/agent-env.json").status_code
acl = {}
for path in (prefix / ADDRESS.parent, prefix / ADDRESS):
    listed = subprocess.run(["icacls", str(path)], capture_output=True, text=True).stdout.replace(str(path), "", 1)
    acl[path.name] = sorted(line.strip().split(":")[0].rpartition("\\")[2] for line in listed.splitlines()
                            if ":(" in line)
print(json.dumps({
    "notes": json.loads(written.read_text()),
    "windows": [window for window in asyncio.run(session.windows()) if window["title"] == "Notes"],
    "launcher": (Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/notes.lnk").is_file(),
    "launched": sorted(path.name for path in (Path(tempfile.gettempdir()) / "agentenv" / "apps").iterdir()),
    "run": (prefix / "run.txt").read_text().strip(), "installed": (prefix / "installed.txt").read_text(),
    "license": (prefix / "LICENSE.txt").read_text().startswith("A. HISTORY OF THE SOFTWARE"),
    "acl": acl, "token": [tokenless.status_code, answered]}))
"""
EXPECTED = {
    "notes": {"signature": "-- Ada", "token": "t0ken", "edition": "pro"},
    "windows": [{"instance": "pythonw.exe", "class": "TkTopLevel", "title": "Notes"}], "launcher": True,
    "launched": ["notes"], "run": "elevated True in C:\\agentenv\\apps\\notes", "installed": "pro", "license": True,
    "acl": {name: ["Administrators", "SYSTEM", "user"] for name in (".agentenv", "server.json")},
    "token": [403, 200]}


async def verify(mcp_url, env_url):
    driver = await Driver.connect(env_url, 120)
    status = await driver.invoke(COMPUTER_READY, "status")
    app = [check for check in status.get("checks", []) if check.get("name") == "app notes"]
    results = [{"id": "ready", "result": bool(status.get("ready")) and [c.get("ok") for c in app] == [True],
                "description": json.dumps(app)}]
    out = await driver.run(PYTHON, "-c", FACTS, timeout_s=120)
    try:
        facts = json.loads(out.get("stdout") or "")
    except ValueError:
        return results + [{"id": "facts", "result": False, "description": json.dumps(out)[-2000:]}]
    return results + [{"id": name, "result": facts.get(name) == expected,
                       "description": json.dumps(facts.get(name))} for name, expected in EXPECTED.items()]
