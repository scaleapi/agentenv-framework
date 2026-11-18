"""The proxy's shared versions and paths (its ``versions.sh``), and the Dockerfile lines that bake it into a task image,
as the README and ``bake.sh`` give them."""

import re
from pathlib import Path

import agent_env

PROXY_DIR = Path(agent_env.__file__).parent / "agents" / "agents" / "proxy"
VERSIONS = dict(re.findall(r"^(\w+)=(\S+)", (PROXY_DIR / "versions.sh").read_text(), re.M))
BAKE_RECIPE = (
    f"COPY --from=ghcr.io/astral-sh/uv:{VERSIONS['UV_VERSION']}@{VERSIONS['UV_IMAGE_DIGEST']} /uv "
    f"{VERSIONS['BAKED']}/bin/uv",
    f"COPY agentenv-proxy {VERSIONS['BAKED']}/src",
    f"RUN sh {VERSIONS['BAKED']}/src/bake.sh",
)
