"""What the notes app's service wrote, which it can only once its settings reached it, the name the computer's card
gives and what its readiness says of the app, asked on loopback from inside the computer, the launcher its server added
to the session user's menu and the apps launched since the computer booted."""
import json
import time
import urllib.request
from pathlib import Path


def run(input):
    written = Path.home() / ".notes.json"
    for _ in range(60):
        if written.exists():
            break
        time.sleep(0.5)
    with urllib.request.urlopen(f"{input['env']['url']}/.well-known/agent-env.json", timeout=10) as response:
        card = json.load(response)
    with urllib.request.urlopen(f"{input['env']['url']}/agentenv/ext/computer-ready/status", timeout=30) as response:
        ready = json.load(response)
    return {**json.loads(written.read_text()), "card": card["name"],
            "app_ready": [check for check in ready["checks"] if check["name"].startswith("app ")],
            "launchers": sorted(p.name for p in (Path.home() / ".local/share/applications").glob("agentenv-app-*")),
            "launched": sorted(p.name for p in Path("/run/agentenv/apps").iterdir())}
