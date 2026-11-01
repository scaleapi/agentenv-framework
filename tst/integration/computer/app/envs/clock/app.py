"""A clock app for the computer env tests, composed beside notes: its service answers on a port of its own with the
time zone its settings name, and writes it to ~/.clock.json. It installs tzdata, which its run script lists the zones
of, and declares no window. Its data plane is its own hooks over its alarms, kept in ~/.clock-alarms.json."""
import json
from pathlib import Path

from pydantic import BaseModel

from agentenv_protocol import DataPart, add_data, get_data, reset_data
from agentenv_protocol.computer import (ComputerAppEnvironment, Packages, RunScript, Service, computer_app, linux,
                                        part_bytes)

PORT = 8137
ALARMS = ".clock-alarms.json"


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=["tzdata"]), run=[RunScript(script="setup.sh")],
       services=[Service(name="clock", command="venv/bin/python serve.py")], ports=[PORT])
class Clock(ComputerAppEnvironment):
    class Settings(BaseModel):
        zone: str = "UTC"

    @add_data
    async def load(self, parts: list) -> None:
        for part in parts:
            (Path.home() / ALARMS).write_bytes(await part_bytes(part))

    @get_data
    def alarms(self) -> list:
        path = Path.home() / ALARMS
        return [DataPart(data=json.loads(path.read_text()) if path.exists() else {"alarms": []})]

    @reset_data
    def clear(self) -> None:
        (Path.home() / ALARMS).unlink(missing_ok=True)
