"""What each of the two apps composed into the desk wrote on the settings its deploy gave it, the clock's answer on its
own port, the name the computer's card gives and what its readiness says of each app, asked on loopback from inside
the computer, the apps launched since it booted, and each app's install under its own folder, its run script's output
there."""
import json
import time
import urllib.request
from pathlib import Path

APPS = Path("/opt/agentenv/apps")


def run(input):
    written = [Path.home() / ".notes.json", Path.home() / ".clock.json"]
    for _ in range(60):
        if all(path.exists() for path in written):
            break
        time.sleep(0.5)
    with urllib.request.urlopen(f"{input['env']['url']}/.well-known/agent-env.json", timeout=10) as response:
        card = json.load(response)
    status = urllib.request.Request(f"{input['env']['url']}/agentenv/ext/computer-ready/status",
                                    headers=input["env"]["headers"])
    with urllib.request.urlopen(status, timeout=30) as response:
        ready = json.load(response)
    with urllib.request.urlopen("http://127.0.0.1:8137/", timeout=10) as response:
        clock = json.load(response)
    notes = json.loads(written[0].read_text())
    return {"notes": {"signature": notes["signature"], "token": notes["token"]},
            "clock": {**json.loads(written[1].read_text()), "served": clock}, "card": card["name"],
            "app_ready": sorted(check["name"] for check in ready["checks"]
                                if check["name"].startswith("app ") and check["ok"]),
            "launched": sorted(p.name for p in Path("/run/agentenv/apps").iterdir()),
            "folders": sorted(p.name for p in APPS.iterdir()), "hello": (APPS / "notes" / "hello.txt").exists(),
            "paris": "Paris" in (APPS / "clock" / "zones.txt").read_text().split()}
