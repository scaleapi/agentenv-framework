"""Each app's data/reset, sent from inside the computer through the driver's route to it, and what the apps keep
after: the notes in ~/Notes, and whether the clock's alarms are there."""
import json
import urllib.request
from pathlib import Path


def run(input):
    answers = {}
    for app in ("notes", "clock"):
        request = urllib.request.Request(f"{input['env']['url']}/svc/{app}/agentenv", data=json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "data/reset", "params": {}}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            answers[app] = "error" not in json.load(response)
    return {"reset": answers, "notes": sorted(p.name for p in (Path.home() / "Notes").iterdir()),
            "alarms": (Path.home() / ".clock-alarms.json").exists()}
