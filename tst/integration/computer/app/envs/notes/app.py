"""A notes app for the computer app tests: its service writes the settings a deploy gave it to ~/.notes.json."""
from pydantic import BaseModel, SecretStr

from agentenv_protocol.computer import ComputerAppEnvironment, Launch, Linux, Service, computer_app


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
class Notes(ComputerAppEnvironment):
    linux = Linux(python="requirements.txt", services=[Service(name="notes", command="venv/bin/python serve.py")])
    launch = Launch(at="demand", command=["xdg-open", ".notes.json"])
    skill = "skill.md"

    class Settings(BaseModel):
        signature: str = "-- notes"
        token: SecretStr = SecretStr("")
