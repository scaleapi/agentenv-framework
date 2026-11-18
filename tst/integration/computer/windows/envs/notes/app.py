"""A notes app for the Windows computer tests, composed onto a desktop as notes-desk: its service writes the settings a
deploy gave it to ~/.notes.json, and its own server launches its window once it serves; it's ready once both have
happened. It requires no Python of its own, so its service and window run in the venv every app gets for its server.
Its install takes a download checked by sha256, and a PowerShell run script and an install step, both run elevated in
its folder, the step on the build setting its env.toml fixes; its window is matched by its title."""
from pathlib import Path

from pydantic import BaseModel, SecretStr

from agentenv_protocol.computer import (ComputerAppEnvironment, Download, Launch, Ready, RunScript, Service,
                                        computer_app, install_step, ready_check, windows)

LICENSE = "https://raw.githubusercontent.com/python/cpython/v3.12.0/LICENSE"


@computer_app(supports=[{"os": {"type": "windows", "display": ["dwm"]}, "arch": ["amd64"]}])
@windows(downloads=[Download(url=LICENSE, to="LICENSE.txt",
                             sha256="3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf")],
         run=[RunScript(script="setup.ps1")],
         services=[Service(name="notes", command="venv/Scripts/python.exe serve.py")],
         launch=Launch(at="boot", command=["venv/Scripts/pythonw.exe", "window.py"], window={"title": "Notes"},
                       ready=Ready(window=True)),
         paths=["Notes"])
class Notes(ComputerAppEnvironment):
    class Settings(BaseModel):
        signature: str = "-- notes"
        token: SecretStr = SecretStr("")

    class BuildSettings(BaseModel):
        edition: str = "home"

    @install_step()
    def stamp(self) -> None:
        Path("installed.txt").write_text(self.build_settings.edition)

    @ready_check()
    def written(self) -> bool:
        return (Path.home() / ".notes.json").exists()
