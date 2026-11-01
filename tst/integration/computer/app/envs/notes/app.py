"""A notes app for the computer app tests: its service writes the settings a deploy gave it to ~/.notes.json, and its
own server launches its window once the desktop session is up; it's ready once both have happened. It requires no
Python of its own, so its service and window run in the venv every app gets for its server. Its install takes an apt
package from Ubuntu's archive and one only a Launchpad PPA has, which its run script uses, a download checked by sha256,
and an install step run as the session user on the build setting its env.toml fixes, which also leaves a welcome note
in ~/Notes, where it keeps its data, so its server serves that folder as its data plane. On Linux its settings gain a
font, and its install step and ready check are Linux's."""
from pathlib import Path

from pydantic import BaseModel, SecretStr

from agentenv_protocol.computer import (AptSource, ComputerAppEnvironment, Download, Launch, Packages, Ready, RunScript,
                                        Service, computer_app, install_step, linux, ready_check)

LICENSE = "https://raw.githubusercontent.com/python/cpython/v3.12.0/LICENSE"
FASTFETCH = "https://ppa.launchpadcontent.net/zhangsongcui3371/fastfetch/ubuntu"


class LinuxSettings(BaseModel):
    font: str = "DejaVu Sans"


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=["hello", "fastfetch"]),
       sources=[AptSource(apt=FASTFETCH, suite="noble", components=["main"], key="fastfetch.asc")],
       downloads=[Download(url=LICENSE, to="LICENSE.txt",
                           sha256="3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf")],
       run=[RunScript(script="setup.sh")],
       services=[Service(name="notes", command="venv/bin/python serve.py")],
       launch=Launch(at="boot", command=["venv/bin/python", "window.py"], window={"class": "Notes"},
                     ready=Ready(window=True)),
       paths=["Notes"], settings=LinuxSettings)
class Notes(ComputerAppEnvironment):
    skill = "skill.md"

    class Settings(BaseModel):
        signature: str = "-- notes"
        token: SecretStr = SecretStr("")

    class BuildSettings(BaseModel):
        edition: str = "home"

    @install_step(scope="session", os="linux")
    def remember(self) -> None:
        (Path.home() / ".notes-installed").write_text(self.build_settings.edition)
        (Path.home() / "Notes").mkdir(exist_ok=True)
        (Path.home() / "Notes" / "welcome.md").write_text("# Welcome\n")

    @ready_check(os="linux")
    def written(self) -> bool:
        return (Path.home() / ".notes.json").exists()
