"""The Docker host image AgentEnv sandboxes boot from.

Daytona builds it from this recipe the first time a sandbox asks for it and caches the result as a
snapshot in the organization, so nothing is hosted outside Daytona. AgentEnv runs its setup as
``sudo bash -c <script>``, loads and builds images, and runs docker-compose, so the image needs Docker
Engine, the Compose v2 plugin and a passwordless-sudo user. Docker's storage is left empty: the daemon
prepares it at boot.
"""

from __future__ import annotations

from daytona import Image

# Bump when the recipe changes; it is written into the image so a sandbox shows which recipe it has.
HOST_IMAGE_VERSION = "v2"
# Daytona's own sandbox image: the toolbox user ``daytona`` with passwordless sudo and the usual tools.
DEFAULT_BASE_IMAGE = "daytonaio/sandbox:0.9.0"
SANDBOX_USER = "daytona"

_RECIPE = (
    "apt-get update",
    "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends"
    " ca-certificates curl procps sudo",
    "install -m 0755 -d /etc/apt/keyrings",
    ". /etc/os-release && curl -fsSL https://download.docker.com/linux/$ID/gpg"
    " -o /etc/apt/keyrings/docker.asc && chmod a+r /etc/apt/keyrings/docker.asc",
    '. /etc/os-release && echo "deb [arch=$(dpkg --print-architecture)'
    " signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/$ID"
    ' $VERSION_CODENAME stable" > /etc/apt/sources.list.d/docker.list',
    "apt-get update",
    "DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends"
    " docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin",
    # A base without the toolbox user (e.g. plain Ubuntu) gets one, with the same passwordless sudo.
    f"(id -u {SANDBOX_USER} >/dev/null 2>&1 || useradd -m -s /bin/bash {SANDBOX_USER})",
    f"echo '{SANDBOX_USER} ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-agentenv"
    " && chmod 0440 /etc/sudoers.d/90-agentenv",
    f"echo 'agentenv-host-image={HOST_IMAGE_VERSION}' > /etc/agentenv-host-image",
    "rm -rf /var/lib/apt/lists/* /var/lib/docker /var/lib/containerd",
)

# Daytona does not start dockerd in this image, so the sandbox starts it: detached from the session
# that launches it, which ends when the command returns, then waited for until it answers.
DOCKER_START_SCRIPT = (
    "pgrep -x dockerd >/dev/null || (nohup setsid dockerd >/var/log/dockerd.log 2>&1 </dev/null &)"
)


def host_image(base_image: str = DEFAULT_BASE_IMAGE) -> Image:
    """The declarative Docker host image on ``base_image``, which must be Debian or Ubuntu based."""
    return (
        Image.base(base_image)
        .dockerfile_commands(["USER root"])
        .run_commands(" && ".join(_RECIPE))
        .dockerfile_commands([f"USER {SANDBOX_USER}"])
    )
