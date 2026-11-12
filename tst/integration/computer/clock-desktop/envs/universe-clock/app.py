"""agent-env's libfaketime on the computer, for the clock desktop test: built at compose from the recipe the test
copies into ./clock (src/agent_env/computers/clock) and upstream's tarball, which the app downloads at the sha256
build.sh pins, then preloaded into every process with its settings and the clock page's writer beside it (install.sh).
The page itself is the test's to write, as root, as the computer's clock follower will."""
import re
from pathlib import Path

from agentenv_protocol.computer import (ComputerAppEnvironment, Download, Packages, RunScript, computer_app, linux)

_BUILD = (Path(__file__).parent / "clock" / "build.sh").read_text()
TARBALL = re.search(r"https://\S+\.tar\.gz", _BUILD).group(0)
SHA256 = re.search(r"\b[0-9a-f]{64}\b", _BUILD).group(0)


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=["build-essential", "patch"]),
       downloads=[Download(url=TARBALL, to="libfaketime.tar.gz", sha256=SHA256)],
       run=[RunScript(script="install.sh")])
class UniverseClock(ComputerAppEnvironment):
    pass
