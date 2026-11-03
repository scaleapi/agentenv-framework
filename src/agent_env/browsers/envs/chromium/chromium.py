"""agentenv's chromium app: Chrome for Testing at the build Playwright 1.64.1 pins, for amd64 and arm64 (Google ships
no Linux arm64 Chrome), a website browser whose flags, policies and settings agentenv's base class gives it. Its data is
its profile, which the image has none of, so every deploy and every reset starts with a clean one."""
import subprocess
from pathlib import Path

from agentenv_protocol.computer import Download, Launch, Packages, Ready, computer_app, install_step, linux
from agentenv_protocol.website_browser import WebsiteBrowserEnvironment

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
PROFILE = ".config/chromium"


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=PACKAGES), downloads=DOWNLOADS, paths=[PROFILE],
       launch=Launch(at="boot", command=["chromium/chrome/chrome", "--no-sandbox", f"--class={WINDOW_CLASS}"],
                     window={"class": WINDOW_CLASS}, ready=Ready(window=True)))
class Chromium(WebsiteBrowserEnvironment):
    """Chrome for Testing without its sandbox, which a container can't give it."""

    version = VERSION
    #: Where Chrome for Testing reads managed policies; it reads neither Chromium's folder nor Chrome's.
    policy_dir = Path("/etc/opt/chrome_for_testing/policies/managed")
    profile = PROFILE

    @install_step(scope="system", os="linux")
    def chromium(self) -> None:
        unpack(Path.cwd())


def unpack(folder: Path) -> None:
    """Unpack the arch's archive under ``folder``'s ``chromium/`` to ``chromium/chrome/``, with unzip, which keeps its
    execute bits, in its place."""
    [archive] = (folder / "chromium").glob("chrome-*.zip")
    unpacked = folder / "chromium" / "unpacked"
    subprocess.run(["unzip", "-q", str(archive), "-d", str(unpacked)], check=True)
    [top] = unpacked.iterdir()
    top.rename(folder / "chromium" / "chrome")
    unpacked.rmdir()
    archive.unlink()
