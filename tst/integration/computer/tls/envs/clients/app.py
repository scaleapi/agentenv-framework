"""TLS clients for the CA trust tests: Node, a JDK and curl from Ubuntu's archive, and a copy of that JDK under /opt
with a store of its own, as a JDK not from apt has. Launched at boot, it loads its ``url`` setting through Node, each
JDK and curl, each through its own trust, and writes what each was answered to ~/.tls.json; it's ready once it has."""
from pathlib import Path

from pydantic import BaseModel

from agentenv_protocol.computer import (ComputerAppEnvironment, Launch, Packages, RunScript, computer_app, linux,
                                        ready_check)


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=["openjdk-21-jdk-headless", "nodejs", "curl"]), run=[RunScript(script="setup.sh")],
       launch=Launch(at="boot", command=["venv/bin/python", "fetch.py"]))
class Clients(ComputerAppEnvironment):

    class Settings(BaseModel):
        url: str = ""

    @ready_check(os="linux")
    def fetched(self) -> bool:
        return (Path.home() / ".tls.json").exists()
