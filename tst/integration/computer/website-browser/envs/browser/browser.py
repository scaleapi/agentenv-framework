"""A website browser for the WebsiteBrowserEnvironment tests: the Chrome for Testing build agentenv's chromium ships,
launched by its command itself so agentenv's flags reach it, with a file its install step leaves in its profile as the
profile's template, which it keeps as its data, so a reset relaunches it on that template."""
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
PACKAGES = ["fonts-liberation", "libasound2t64", "libatk-bridge2.0-0t64", "libatk1.0-0t64", "libatspi2.0-0t64",
            "libcairo2", "libcups2t64", "libdbus-1-3", "libdrm2", "libexpat1", "libgbm1", "libglib2.0-0t64",
            "libgtk-3-0t64", "libnspr4", "libnss3", "libpango-1.0-0", "libvulkan1", "libx11-6", "libxcb1",
            "libxcomposite1", "libxdamage1", "libxext6", "libxfixes3", "libxkbcommon0", "libxrandr2", "libxshmfence1",
            "unzip"]
PROFILE = ".config/chromium"
TEMPLATE = "agentenv-template.txt"


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=PACKAGES), downloads=DOWNLOADS, paths=[PROFILE],
       launch=Launch(command=["chromium/chrome/chrome", "--no-sandbox", "--class=Chromium"],
                     window={"class": "Chromium"}, ready=Ready(window=True)))
class FixtureBrowser(WebsiteBrowserEnvironment):
    version = VERSION
    policy_dir = Path("/etc/opt/chrome_for_testing/policies/managed")

    @install_step(scope="system")
    def unpack(self) -> None:
        folder = Path.cwd() / "chromium"
        [archive] = folder.glob("chrome-*.zip")
        subprocess.run(["unzip", "-q", str(archive), "-d", str(folder / "unpacked")], check=True)
        [top] = (folder / "unpacked").iterdir()
        top.rename(folder / "chrome")
        (folder / "unpacked").rmdir()
        archive.unlink()

    @install_step(scope="session")
    def profile_template(self) -> None:
        (Path.home() / PROFILE).mkdir(parents=True, exist_ok=True)
        (Path.home() / PROFILE / TEMPLATE).write_text("from the profile template\n")
