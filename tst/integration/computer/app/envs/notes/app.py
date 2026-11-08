"""A notes app for the computer app tests: its service writes the settings a deploy gave it to ~/.notes.json, and its
own server launches its window once the desktop session is up; it's ready once both have happened. It requires no
Python of its own, so its service and window run in the venv every app gets for its server. Its install takes an apt
package from Ubuntu's archive and one only a Launchpad PPA has, which its run script uses, a download checked by sha256,
and an install step run as the session user on the build setting its env.toml fixes, which also leaves a welcome note
in ~/Notes, where it keeps its data, so its server serves that folder as its data plane. On Linux its settings gain a
font, and its install step and ready check are Linux's. On Windows it takes the same download, a PowerShell run
script, run elevated, and an install step run elevated in its folder, as Windows runs none as the session user; its
window is matched by its title there, and its font is Windows'."""
from pathlib import Path

from pydantic import BaseModel, SecretStr

from agentenv_protocol.computer import (AptSource, ComputerAppEnvironment, Download, Launch, Packages, Ready, RunScript,
                                        Service, computer_app, install_step, linux, ready_check, windows)

LICENSE = "https://raw.githubusercontent.com/python/cpython/v3.12.0/LICENSE"
LICENSE_SHA256 = "3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf"
FASTFETCH = "https://ppa.launchpadcontent.net/zhangsongcui3371/fastfetch/ubuntu"


class LinuxSettings(BaseModel):
    font: str = "DejaVu Sans"


class WindowsSettings(BaseModel):
    font: str = "Segoe UI"


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]},
                        {"os": {"type": "windows", "display": ["dwm"]}, "arch": ["amd64"]}])
@linux(packages=Packages(apt=["hello", "fastfetch"]),
       sources=[AptSource(apt=FASTFETCH, suite="noble", components=["main"], key="fastfetch.asc")],
       downloads=[Download(url=LICENSE, to="LICENSE.txt", sha256=LICENSE_SHA256)],
       run=[RunScript(script="setup.sh")],
       services=[Service(name="notes", command="venv/bin/python serve.py")],
       launch=Launch(at="boot", command=["venv/bin/python", "window.py"], window={"class": "Notes"},
                     ready=Ready(window=True)),
       paths=["Notes"], settings=LinuxSettings)
@windows(downloads=[Download(url=LICENSE, to="LICENSE.txt", sha256=LICENSE_SHA256)],
         run=[RunScript(script="setup.ps1")],
         services=[Service(name="notes", command="venv/Scripts/python.exe serve.py")],
         launch=Launch(at="boot", command=["venv/Scripts/pythonw.exe", "window.py"], window={"title": "Notes"},
                       ready=Ready(window=True)),
         paths=["Notes"], settings=WindowsSettings)
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

    @install_step(os="windows")
    def stamp(self) -> None:
        Path("installed.txt").write_text(self.build_settings.edition)

    @ready_check(os=["linux", "windows"])
    def written(self) -> bool:
        return (Path.home() / ".notes.json").exists()
