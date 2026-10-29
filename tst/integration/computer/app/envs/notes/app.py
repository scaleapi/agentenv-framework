"""A notes app for the computer app tests: its service writes the settings a deploy gave it to ~/.notes.json, and its
window opens with the desktop session; it's ready once both have happened."""
from pathlib import Path

from pydantic import BaseModel, SecretStr

from agentenv_protocol.computer import ComputerAppEnvironment, Launch, Linux, Ready, Service, computer_app, ready_check


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
class Notes(ComputerAppEnvironment):
    linux = Linux(python="requirements.txt", services=[Service(name="notes", command="venv/bin/python serve.py")])
    launch = Launch(at="boot", command=["venv/bin/python", "window.py"], window={"class": "Notes"},
                    ready=Ready(window=True))
    skill = "skill.md"

    class Settings(BaseModel):
        signature: str = "-- notes"
        token: SecretStr = SecretStr("")

    @ready_check
    def written(self) -> bool:
        return (Path.home() / ".notes.json").exists()
