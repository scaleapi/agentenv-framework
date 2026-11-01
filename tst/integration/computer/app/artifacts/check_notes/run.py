"""What the notes app's service wrote, which it can only once its settings reached it, the name the computer's card
gives and what its readiness says of the app, asked on loopback from inside the computer, the launcher its server added
to the session user's menu, the apps launched since the computer booted, and what its install left: its run script's
output of the apt packages it installed, its PPA's among them, its download, and its install step's file."""
import json
import time
import urllib.request
from pathlib import Path

PREFIX = Path("/opt/agentenv/apps/notes")


def run(input):
    written = Path.home() / ".notes.json"
    for _ in range(60):
        if written.exists():
            break
        time.sleep(0.5)
    with urllib.request.urlopen(f"{input['env']['url']}/.well-known/agent-env.json", timeout=10) as response:
        card = json.load(response)
    status = urllib.request.Request(f"{input['env']['url']}/agentenv/ext/computer-ready/status",
                                    headers=input["env"]["headers"])
    with urllib.request.urlopen(status, timeout=30) as response:
        ready = json.load(response)
    return {**json.loads(written.read_text()), "card": card["name"],
            "app_ready": [check for check in ready["checks"] if check["name"].startswith("app ")],
            "launchers": sorted(p.name for p in (Path.home() / ".local/share/applications").glob("agentenv-app-*")),
            "launched": sorted(p.name for p in Path("/run/agentenv/apps").iterdir()),
            "hello": (PREFIX / "hello.txt").read_text().strip(),
            "fastfetch": (PREFIX / "fastfetch.txt").read_text().splitlines(),
            "license": (PREFIX / "LICENSE.txt").read_text().startswith("A. HISTORY OF THE SOFTWARE"),
            "installed": (Path.home() / ".notes-installed").read_text()}
