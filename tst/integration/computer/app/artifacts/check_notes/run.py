"""What the notes app's service wrote, which it can only once its settings reached it, and the name the computer's
card gives, asked on loopback from inside the computer."""
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
    return {**json.loads(written.read_text()), "card": card["name"]}
