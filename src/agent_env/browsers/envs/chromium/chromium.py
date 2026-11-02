"""agentenv's chromium app: Chrome for Testing at the build Playwright 1.64.1 pins, for amd64 and arm64 (Google ships
no Linux arm64 Chrome), set up for unattended use by managed policies and opened at boot on the deploy's
``start_urls``. Its data is its profile, which the image has none of, so every deploy and every reset starts with a
clean one; its HTTP cache is kept out of it. Run as a script, it is the app's launch."""
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from agentenv_protocol.computer import (ComputerAppEnvironment, Download, Launch, Packages, Ready, computer_app,
                                        install_step, linux)

VERSION = "156.0.8078.4"
_ARCHIVES = {"amd64": ("linux64/chrome-linux64.zip",
                       "43fd31766c7533a02dda2ded5ad50c2f861cd5400be7509c0abad71129f79d79"),
             "arm64": ("linux-arm64/chrome-linux-arm64.zip",
                       "910a12fff6170202195f72ef3d353801c7063d301ab80b1f51b77d791fda60ce")}
DOWNLOADS = [Download(url=f"https://storage.googleapis.com/chrome-for-testing-public/{VERSION}/{path}", sha256=sha256,
                      to=f"chromium/chrome-{arch}.zip", arch=arch)
             for arch, (path, sha256) in _ARCHIVES.items()]
#: What it links against that Ubuntu 24.04 lacks, a font for the pages it shows, and unzip for its install step.
PACKAGES = ["fonts-liberation", "libasound2t64", "libatk-bridge2.0-0t64", "libatk1.0-0t64", "libatspi2.0-0t64",
            "libcairo2", "libcups2t64", "libdbus-1-3", "libdrm2", "libexpat1", "libgbm1", "libglib2.0-0t64",
            "libgtk-3-0t64", "libnspr4", "libnss3", "libpango-1.0-0", "libvulkan1", "libx11-6", "libxcb1",
            "libxcomposite1", "libxdamage1", "libxext6", "libxfixes3", "libxkbcommon0", "libxrandr2", "libxshmfence1",
            "unzip"]
WINDOW_CLASS = "Chromium"
#: Where Chrome for Testing reads managed policies; it reads neither Chromium's folder nor Chrome's.
POLICY_DIR = Path("/etc/opt/chrome_for_testing/policies/managed")
#: No sign-in, sync, welcome or promotional tabs, default-browser prompt, telemetry, password prompts, autofill on any
#: site, translation offers, updates, or location and notification prompts; and no warning bar for the flags a container
#: needs, which also takes away Chrome for Testing's own bar.
POLICIES = {
    "BrowserSignin": 0,
    "SyncDisabled": True,
    "PromotionalTabsEnabled": False,
    "DefaultBrowserSettingEnabled": False,
    "MetricsReportingEnabled": False,
    "PasswordManagerEnabled": False,
    "AutofillSettings": [{"url_pattern": "*", "blocked_types": ["all"]}],
    "TranslateEnabled": False,
    "ComponentUpdatesEnabled": False,
    "DefaultGeolocationSetting": 2,
    "DefaultNotificationsSetting": 2,
    "CommandLineFlagSecurityWarningsEnabled": False,
}
PROFILE = ".config/chromium"
CACHE = ".cache/chromium"


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=PACKAGES), downloads=DOWNLOADS, paths=[PROFILE],
       launch=Launch(at="boot", command=["venv/bin/python", "chromium.py"], window={"class": WINDOW_CLASS},
                     ready=Ready(window=True)))
class Chromium(ComputerAppEnvironment):
    class Settings(BaseModel):
        #: Opened in tabs at launch, the first active; none opens a new tab.
        start_urls: list[str] = []

    @install_step(scope="system", os="linux")
    def chromium(self) -> None:
        install_chromium(Path.cwd())


def install_chromium(folder: Path, policies: Path = POLICY_DIR) -> None:
    """Unpack the arch's archive under ``folder``'s ``chromium/`` to ``chromium/chrome/``, with unzip, which keeps its
    execute bits, in its place, and write ``POLICIES`` to ``policies``."""
    [archive] = (folder / "chromium").glob("chrome-*.zip")
    unpacked = folder / "chromium" / "unpacked"
    subprocess.run(["unzip", "-q", str(archive), "-d", str(unpacked)], check=True)
    [top] = unpacked.iterdir()
    top.rename(folder / "chromium" / "chrome")
    unpacked.rmdir()
    archive.unlink()
    policies.mkdir(parents=True, exist_ok=True)
    (policies / "agentenv.json").write_text(json.dumps(POLICIES, indent=1) + "\n", encoding="utf-8")


def chromium_command(folder: Path, *, home: Path, urls: Iterable[str]) -> list[str]:
    """Chromium's command: no sandbox, which a container can't give it, and no /dev/shm, which no computer sizes; no
    first run or default-browser check; ``WINDOW_CLASS`` for its windows; its profile and, outside it, its cache, under
    ``home``; and ``urls`` in tabs, the first active. It takes no proxy: a computer has no site proxy, and its sites on
    loopback must stay reachable."""
    return [str(folder / "chromium" / "chrome" / "chrome"), "--no-sandbox", "--disable-dev-shm-usage", "--no-first-run",
            "--no-default-browser-check", f"--class={WINDOW_CLASS}", f"--user-data-dir={home / PROFILE}",
            f"--disk-cache-dir={home / CACHE}", *urls]


if __name__ == "__main__":
    command = chromium_command(Path(__file__).resolve().parent, home=Path.home(), urls=Chromium().settings.start_urls)
    os.execv(command[0], command)
