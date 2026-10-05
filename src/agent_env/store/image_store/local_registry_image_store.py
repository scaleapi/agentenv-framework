"""Local OCI registry implementation of the ImageStore interface (no cloud services)."""

from __future__ import annotations

import subprocess
import time
from urllib.parse import urlsplit

from agent_env.store.image_store.image_store import OciRegistryImageStore
from agent_env.store.image_store.oci_registry_credentials import OciRegistryCredentials

_REGISTRY_CONTAINER = "agentenv-registry"
_DEFAULT_PORT = 5000


def _container_name(port: int) -> str:
    """``agentenv-registry`` on the default port, ``agentenv-registry-<port>`` on any other, so
    registries on different ports coexist and each restarts with its own images."""
    return _REGISTRY_CONTAINER if port == _DEFAULT_PORT else f"{_REGISTRY_CONTAINER}-{port}"


class LocalRegistryImageStore(OciRegistryImageStore):
    """ImageStore backed by an anonymous local OCI registry.

    Repositories are created automatically on push.
    """

    def __init__(
        self,
        registry_host: str = "localhost:5000",
        repository_prefix: str = "",
        credentials: OciRegistryCredentials | None = None,
    ) -> None:
        super().__init__(
            registry_host=registry_host,
            repository_prefix=repository_prefix,
            credentials=credentials,
        )

    def ensure_repository(self, repository: str) -> None:
        """A local registry creates repositories on push, so this instead lazily brings up the
        backing ``registry:2`` — idempotently, and safe to call from many concurrent pushes. A
        no-op when something already serves the registry at ``registry_host`` (a registry you
        run yourself, or one a concurrent push already started); it never removes a container.
        Called from the image-push path, so no caller manages it."""
        port = urlsplit(f"//{self.registry_host}").port or _DEFAULT_PORT
        if self._registry_listening(port):
            return
        name = _container_name(port)
        run = self._run_registry(name, port)
        if run.returncode != 0:
            # `docker run` creates the named container atomically, so a failure is either a name
            # conflict (our container already exists) or something unrelated — disambiguate on
            # the existing container instead of blindly retrying, so a real error isn't masked.
            existing_port = self._container_configured_port(name)
            if existing_port is None:
                # No such container: the failure is real — the port is taken by another process,
                # the image pull failed, or the daemon is down. Surface it now, don't wait 30s.
                raise RuntimeError(f"could not start the local registry: {run.stderr.strip()}")
            if existing_port != port:
                # Only an older version's agentenv-registry, which it ran on any port; refuse rather than remove it.
                raise RuntimeError(
                    f"{name} is using port {existing_port}, not {port}: "
                    f"point registry_host at port {existing_port}, or remove the container"
                )
            # A concurrent push won the race, or our stopped container is on this port: (re)start it.
            start = subprocess.run(["docker", "start", name], capture_output=True, text=True)
            if start.returncode != 0:
                raise RuntimeError(
                    f"local registry container exists but could not be started: {start.stderr.strip()}"
                )
        deadline = time.time() + 30
        while time.time() < deadline:
            if self._registry_listening(port):
                return
            time.sleep(0.5)
        raise RuntimeError(f"the local registry started but did not answer /v2/ on port {port} within 30s")

    @staticmethod
    def _run_registry(name: str, port: int) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["docker", "run", "-d", "--restart", "unless-stopped", "--name", name,
             "-p", f"127.0.0.1:{port}:5000", "registry:2"],
            capture_output=True, text=True,
        )

    @staticmethod
    def _container_configured_port(name: str) -> int | None:
        """The host port ``name`` maps container-port 5000 to (works for stopped containers too),
        or None when no such container exists — i.e. the ``docker run`` failure was not a name
        clash but a real error (port taken elsewhere / pull / daemon)."""
        r = subprocess.run(
            ["docker", "inspect", "-f",
             '{{ (index (index .HostConfig.PortBindings "5000/tcp") 0).HostPort }}', name],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return None
        out = r.stdout.strip()
        return int(out) if out.isdigit() else None

    @staticmethod
    def _registry_listening(port: int) -> bool:
        """True if an OCI registry answers the ``/v2/`` ping on ``port`` (ours or one you run)."""
        import httpx

        try:
            return httpx.get(f"http://localhost:{port}/v2/", timeout=2).status_code in (200, 401)
        except Exception:
            return False
