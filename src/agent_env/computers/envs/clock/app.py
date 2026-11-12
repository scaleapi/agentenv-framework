"""The computer's clock on universe time: agent-env's libfaketime and its TLS shim, which keeps certificate checks on
real time, built at compose from the recipe in ./recipe (src/agent_env/computers/clock) and upstream's tarball, which
the app downloads at the sha256 build.sh pins, preloaded into every process (install.sh), with the clock follower
beside them.

The follower keeps the clock page in step with the gateway's clock. Core configures and starts it through the sandbox's
root channel when ``sync_env_clock`` arms the env (``agent_env.env.envs.computer.utils.clock``), never through this app,
whose server runs as the session user. The app's ``clock/v1`` ``sync_time`` only reports what the follower has shown,
which is also what makes a multi env's gateway count the computer as one of its clock's followers."""
import asyncio
import json
import re
import time
from pathlib import Path

from agentenv_protocol.agent_env_environment import extension
from agentenv_protocol.computer import (ComputerAppEnvironment, Download, Packages, RunScript, computer_app, linux)
from agentenv_protocol.computer.errors import InvalidArgument, Timeout
from agentenv_protocol.website.clock import CLOCK_EXTENSION_URI

_BUILD = (Path(__file__).parent / "recipe" / "build.sh").read_text()
TARBALL = re.search(r"https://\S+\.tar\.gz", _BUILD).group(0)
SHA256 = re.search(r"\b[0-9a-f]{64}\b", _BUILD).group(0)
#: What the follower last showed, written by it (agentenv-clock) and readable by anyone.
STATUS = Path("/run/agentenv-clock/status.json")
#: How long ``sync_time`` waits for the follower to show the config core just wrote.
SYNC_WAIT_S = 20.0


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=["build-essential", "patch"]),
       downloads=[Download(url=TARBALL, to="libfaketime.tar.gz", sha256=SHA256)],
       run=[RunScript(script="install.sh")])
class Clock(ComputerAppEnvironment):

    @extension(CLOCK_EXTENSION_URI, description=(
        "The computer's clock follows the gateway's. Core writes the follower's config as root and starts it; this "
        "waits until the follower has shown the clock for that config and returns what it showed."))
    async def sync_time(self, env_get_time_url: str, follower_config: str = "") -> dict:
        """What the follower showed for ``follower_config``, the id core gave the config it wrote, once it has shown
        any version for it. ``env_get_time_url`` is the gateway-network URL every server is handed; the follower
        reaches the gateway at the URL in its config instead, as this computer's sandbox reaches it."""
        if not follower_config:
            raise InvalidArgument("sync_time needs follower_config: core writes the follower's config through the "
                                  "sandbox's root channel before it calls this")
        deadline, status = time.monotonic() + SYNC_WAIT_S, None
        while time.monotonic() < deadline:
            try:
                status = json.loads(STATUS.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                status = None
            if status and status.get("config_id") == follower_config and status.get("version") and \
                    not status.get("error"):
                return {"follower": status, "env_get_time_url": env_get_time_url}
            await asyncio.sleep(0.2)
        raise Timeout(f"the clock follower didn't show config {follower_config!r} within {SYNC_WAIT_S:g} s; "
                      f"its status: {status}")
