"""A clock app for the computer env tests, composed beside notes: its service answers on a port of its own with the
time zone its settings name, and writes it to ~/.clock.json. It installs tzdata, which its run script lists the zones
of, and declares no window."""
from pydantic import BaseModel

from agentenv_protocol.computer import ComputerAppEnvironment, Packages, RunScript, Service, computer_app, linux

PORT = 8137


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=["tzdata"]), run=[RunScript(script="setup.sh")],
       services=[Service(name="clock", command="venv/bin/python serve.py")], ports=[PORT])
class Clock(ComputerAppEnvironment):
    class Settings(BaseModel):
        zone: str = "UTC"
