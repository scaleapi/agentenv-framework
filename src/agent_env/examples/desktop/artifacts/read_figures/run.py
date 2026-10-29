"""What the computer holds where load_artifact put the figures, and the name its driver's card gives, asked on
loopback from inside the computer."""
import json
import urllib.request


def run(input):
    with open("Documents/figures.csv") as figures:
        header = figures.readline().strip()
    with urllib.request.urlopen(f"{input['env']['url']}/.well-known/agent-env.json", timeout=10) as response:
        card = json.load(response)
    return {"header": header, "card": card["name"]}
