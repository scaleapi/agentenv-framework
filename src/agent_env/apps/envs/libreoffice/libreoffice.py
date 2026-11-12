"""agentenv's LibreOffice app: Writer, Impress and Calc from Ubuntu's apt, with the GTK3 interface and fonts with the
metrics of Office's (Carlito for Calibri, Caladea for Cambria, Liberation for Arial, Times New Roman and Courier New),
so Office files keep their layout. Its defaults (``agentenv.xcd``), a layer of LibreOffice's own configuration under the
session user's profile, keep it from asking anything an agent didn't. It launches on demand, through its server, the
Start Center first; every LibreOffice launcher on the computer, those that open a file by its type included, goes
through ``open.py``, so its server starts every copy and a load or a reset can stop it. Its data is its profile, which
the image has none of, so each deploy and each reset starts with a clean one, under the same defaults."""
import asyncio
import re
import subprocess
from pathlib import Path

from agentenv_protocol.computer import ComputerAppEnvironment, Launch, Packages, Ready, computer_app, install_step, linux

#: Writer, Impress and Calc, with GTK3 for the desktop, and the fonts with Office's metrics.
PACKAGES = ["libreoffice-writer", "libreoffice-impress", "libreoffice-calc", "libreoffice-gtk3",
            "fonts-crosextra-carlito", "fonts-crosextra-caladea", "fonts-liberation2"]
SOFFICE = "/usr/lib/libreoffice/program/soffice"
#: Every LibreOffice main window's WM_CLASS instance, the Start Center's and each document's; its dialogs are soffice's.
WINDOW = {"instance": "libreoffice"}
PROFILE = ".config/libreoffice"
#: Where its defaults go: the first layer of its configuration on Debian and Ubuntu, read before the user's profile.
REGISTRY = Path("/etc/libreoffice/registry")
#: Its own launchers, and those that take their place: a desktop entry here has precedence over one of the same name in
#: /usr/share/applications.
DESKTOP_ENTRIES, OVERRIDES = Path("/usr/share/applications"), Path("/usr/local/share/applications")


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=PACKAGES), paths=[PROFILE], skill="SKILL.md",
       launch=Launch(at="demand", command=[SOFFICE, "--nologo"], window=WINDOW, ready=Ready(window=True)))
class LibreOffice(ComputerAppEnvironment):
    """A launch while it runs shows a Start Center in the copy that runs: a second copy would only hand it to the first
    and exit, leaving its server with no copy it knows."""

    def _launch_running(self):
        return self._start_center

    async def _start_center(self) -> None:
        shown = await asyncio.create_subprocess_exec(SOFFICE, "--nologo")
        await shown.wait()

    @install_step(scope="system", os="linux")
    def configure(self) -> None:
        """Its defaults in its configuration's first layer, at the series installed; and each of its desktop entries
        in its place, the same but for opening through ``open.py``."""
        folder = Path.cwd()
        version = subprocess.run(["dpkg-query", "-W", "-f=${Version}", "libreoffice-core"], check=True,
                                 capture_output=True, text=True).stdout
        series = re.match(r"(?:\d+:)?(\d+\.\d+)", version).group(1)
        defaults = (folder / "agentenv.xcd").read_text(encoding="utf-8")
        (REGISTRY / "agentenv.xcd").write_text(re.sub(
            r'(<prop oor:name="ooSetupLastVersion" oor:op="fuse"><value>)[^<]*', rf"\g<1>{series}", defaults),
            encoding="utf-8")
        OVERRIDES.mkdir(parents=True, exist_ok=True)
        opener = f"{folder}/venv/bin/python {folder}/open.py"
        for entry in sorted(DESKTOP_ENTRIES.glob("libreoffice-*.desktop")):
            text = entry.read_text(encoding="utf-8")
            (OVERRIDES / entry.name).write_text(re.sub(r"^Exec=libreoffice\b", f"Exec={opener}", text, flags=re.M),
                                                encoding="utf-8")
        subprocess.run(["update-desktop-database", str(OVERRIDES)], check=True)
