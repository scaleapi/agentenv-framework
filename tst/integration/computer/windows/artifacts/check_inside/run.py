"""A run_code script inside the Windows computer: it runs in the session user's home, and from there reaches the driver
on loopback, whose card says Windows, and its readiness with the computer's token."""
import json
import os
import urllib.request
from pathlib import Path


def run(input):
    with urllib.request.urlopen(f"{input['env']['url']}/.well-known/agent-env.json", timeout=10) as response:
        card = json.load(response)
    computer = next(ext["params"] for ext in card["capabilities"]["extensions"]
                    if ext["uri"] == "urn:agentenv:computer/v1")
    status = urllib.request.Request(f"{input['env']['url']}/agentenv/ext/computer-ready/status",
                                    headers=input["env"]["headers"])
    with urllib.request.urlopen(status, timeout=30) as response:
        ready = json.load(response)
    if computer["platform"]["os"] != "windows" or not Path.cwd().samefile(Path.home()) or not ready.get("ready"):
        raise AssertionError(f"os {computer['platform']['os']}, cwd {Path.cwd()}, ready {ready}")
    return {"user": os.environ.get("USERNAME"), "home": str(Path.home())}
