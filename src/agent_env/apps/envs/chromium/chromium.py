"""agentenv's chromium app: Chromium as ``agentenv_protocol.website_browser`` pins and sets it up, opened at boot on
the deploy's ``start_urls``. Its data is its profile, which the image has none of, so every deploy and every reset
starts with a clean one; its HTTP cache is kept out of it. Run as a script, it is the app's launch."""
import os
from pathlib import Path

from pydantic import BaseModel

from agentenv_protocol.computer import ComputerAppEnvironment, Launch, Packages, Ready, computer_app, linux
from agentenv_protocol.website_browser import (CHROMIUM_DOWNLOADS, CHROMIUM_PACKAGES, WINDOW_CLASS, ChromiumInstall,
                                               chromium_command)

PROFILE = ".config/chromium"
CACHE = ".cache/chromium"


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=CHROMIUM_PACKAGES), downloads=CHROMIUM_DOWNLOADS, paths=[PROFILE],
       launch=Launch(at="boot", command=["venv/bin/python", "chromium.py"], window={"class": WINDOW_CLASS},
                     ready=Ready(window=True)))
class Chromium(ChromiumInstall, ComputerAppEnvironment):
    class Settings(BaseModel):
        #: Opened in tabs at launch, the first active; none opens a new tab.
        start_urls: list[str] = []


if __name__ == "__main__":
    command = chromium_command(Path(__file__).resolve().parent, profile=Path.home() / PROFILE,
                               urls=Chromium().settings.start_urls,
                               flags=[f"--disk-cache-dir={Path.home() / CACHE}"])
    os.execv(command[0], command)
