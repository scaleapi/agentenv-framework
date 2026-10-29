import json
import time
from pathlib import Path

from app import Notes

notes = Notes()
Path.home().joinpath(".notes.json").write_text(json.dumps(
    {"signature": notes.settings.signature, "token": notes.settings.token.get_secret_value()}))
while True:
    time.sleep(3600)
