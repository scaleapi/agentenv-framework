"""The notes app composed onto the deployed Omarchy computer: ready once its server says its window is shown and its
@ready_check passes; and, read inside the computer as the session user, on the app's own venv: what its service wrote on
the deploy's settings, its secret resolved at boot; its window, listed by its title; what its install left: the
download, its run script's output, run as root in its folder, its root install step's file on the build setting its env
fixes, and its session install step's, written as the session user in their home; and its own folder, the session
user's alone."""
import json

from agentenv_protocol.computer.card import COMPUTER_READY
from agentenv_protocol.computer.conformance.client import Driver

PYTHON = "/opt/agentenv/apps/notes/venv/bin/python"
FACTS = r"""
import asyncio, json, pwd, stat, time
from pathlib import Path
from agentenv_protocol.computer import session
from agentenv_protocol.computer.app import SOCKET

prefix = Path("/opt/agentenv/apps/notes")
written = Path.home() / ".notes.json"
deadline = time.monotonic() + 60
while not written.exists() and time.monotonic() < deadline:
    time.sleep(0.5)
own, home = (prefix / SOCKET).parent.stat(), Path.home() / ".notes-installed"
print(json.dumps({
    "notes": json.loads(written.read_text()),
    "windows": [window["title"] for window in asyncio.run(session.windows()) if window["title"] == "Notes"],
    "run": (prefix / "run.txt").read_text().strip(), "installed": (prefix / "installed.txt").read_text(),
    "home": [home.read_text(), pwd.getpwuid(home.stat().st_uid).pw_name],
    "license": (prefix / "LICENSE.txt").read_text().startswith("A. HISTORY OF THE SOFTWARE"),
    "own": [pwd.getpwuid(own.st_uid).pw_name, oct(stat.S_IMODE(own.st_mode))]}))
"""
EXPECTED = {
    "notes": {"signature": "-- Ada", "token": "t0ken", "edition": "pro"}, "windows": ["Notes"],
    "run": "root in /opt/agentenv/apps/notes", "installed": "pro", "home": ["omarchy", "omarchy"], "license": True,
    "own": ["omarchy", "0o700"]}


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
