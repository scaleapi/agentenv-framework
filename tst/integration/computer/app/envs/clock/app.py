"""A clock app for the computer env tests, composed beside notes: its service answers on a port of its own with the
time zone its settings name, and writes it to ~/.clock.json. It installs tzdata, and declares no window."""
from pydantic import BaseModel

from agentenv_protocol.computer import ComputerAppEnvironment, Packages, Service, computer_app, linux

PORT = 8137


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=["tzdata"]), services=[Service(name="clock", command="venv/bin/python serve.py")],
       ports=[PORT])
class Clock(ComputerAppEnvironment):
    class Settings(BaseModel):
        zone: str = "UTC"
