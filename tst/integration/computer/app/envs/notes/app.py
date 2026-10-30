"""A notes app for the computer app tests: its service writes the settings a deploy gave it to ~/.notes.json, and its
own server launches its window once the desktop session is up; it's ready once both have happened. It requires no
Python of its own, so its service and window run in the venv every app gets for its server. Its install takes an apt
package, which its run script uses, a download checked by sha256, and an install step run as the session user."""
from pathlib import Path

from pydantic import BaseModel, SecretStr

from agentenv_protocol.computer import (ComputerAppEnvironment, Download, Launch, Linux, Packages, Ready, RunScript,
                                        Service, computer_app, install_step, ready_check)

LICENSE = "https://raw.githubusercontent.com/python/cpython/v3.12.0/LICENSE"


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
class Notes(ComputerAppEnvironment):
    linux = Linux(packages=Packages(apt=["hello"]),
                  downloads=[Download(url=LICENSE, to="LICENSE.txt",
                                      sha256="3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf")],
                  run=[RunScript(script="setup.sh")],
                  services=[Service(name="notes", command="venv/bin/python serve.py")])
    launch = Launch(at="boot", command=["venv/bin/python", "window.py"], window={"class": "Notes"},
                    ready=Ready(window=True))
    skill = "skill.md"

    class Settings(BaseModel):
        signature: str = "-- notes"
        token: SecretStr = SecretStr("")

    @install_step(scope="session")
    def remember(self) -> None:
        (Path.home() / ".notes-installed").write_text(self.settings.signature)

    @ready_check
    def written(self) -> bool:
        return (Path.home() / ".notes.json").exists()
