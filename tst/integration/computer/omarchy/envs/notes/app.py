"""A notes app for the Omarchy computer tests, composed onto a desktop as notes-desk: its service writes the settings a
deploy gave it to ~/.notes.json, and its own server launches its window once it serves; it's ready once both have
happened. Its install, Omarchy's alone, takes a download checked by sha256, a run script and an install step run as
root in its folder, the step on the build setting its env.toml fixes, and an install step run as the session user in
their home. Its window is GTK 3 on the system Python, which Omarchy ships with PyGObject, matched by its title."""
import os
import pwd
from pathlib import Path

from pydantic import BaseModel, SecretStr

from agentenv_protocol.computer import (ComputerAppEnvironment, Download, Launch, Ready, RunScript, Service,
                                        computer_app, install_step, linux, ready_check)

LICENSE = "https://raw.githubusercontent.com/python/cpython/v3.12.0/LICENSE"


@computer_app(supports=[{"os": {"type": "linux", "distribution": ["omarchy"], "display": ["wayland"]},
                         "arch": ["amd64"]}])
@linux(distribution="omarchy",
       downloads=[Download(url=LICENSE, to="LICENSE.txt",
                           sha256="3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf")],
       run=[RunScript(script="setup.sh")],
       services=[Service(name="notes", command="venv/bin/python serve.py")],
       launch=Launch(at="boot", command=["/usr/bin/python3", "window.py"], window={"title": "Notes"},
                     ready=Ready(window=True)))
class Notes(ComputerAppEnvironment):
    class Settings(BaseModel):
        signature: str = "-- notes"
        token: SecretStr = SecretStr("")

    class BuildSettings(BaseModel):
        edition: str = "home"

    @install_step()
    def stamp(self) -> None:
        Path("installed.txt").write_text(self.build_settings.edition)

    @install_step(scope="session")
    def home(self) -> None:
        (Path.home() / ".notes-installed").write_text(pwd.getpwuid(os.getuid()).pw_name)

    @ready_check()
    def written(self) -> bool:
        return (Path.home() / ".notes.json").exists()
