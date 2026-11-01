"""What reaches inside a deployed computer through its driver, as its session user, at paths relative to the home:
``data/add`` writes files, in batches the driver holds in memory one at a time, each file's bytes coming through a read
grant when the object store's grants reach the computer and its network allows everything, else inline; ``data/get``
reads one back; computer-exec/v1 ``run`` runs a command."""

from __future__ import annotations

import asyncio
import base64
import uuid
from typing import TYPE_CHECKING, Optional

import httpx
from agentenv_protocol import DATA_OBJECTS_EXTENSION_URI, METHOD_ADD, METHOD_GET, FilePart
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer import COMPUTER_EXEC
from agentenv_protocol.types import FileWithBytes, FileWithUri

from agent_env.a2a_agent.object_transfer import read_object
from agent_env.config import get_config
from agent_env.env.gateway.constants import data_plane_load_timeout_s
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.store.object_store.object_store import issues_grants_to

if TYPE_CHECKING:
    from agent_env.artifact import FileArtifact
    from agent_env.env.env import DeployedSandboxEnv

#: The most bytes one ``data/add`` carries: the driver reads a call's files into memory before it writes them. A file
#: bigger than this goes alone.
BATCH_BYTES = 64 << 20
#: How much longer than a command's own timeout its call waits for the answer.
CALL_MARGIN_S = 30
_READ_TIMEOUT_S = 120


def missing(deployed: DeployedSandboxEnv) -> list[str]:
    """What the env's card doesn't serve of computer-exec/v1 ``run``, ``data/add`` and ``data/get``."""
    operations = protocol_v1.operations(deployed.environment_card or {})
    absent = [] if deployed.supports(COMPUTER_EXEC, "run") else [f"{COMPUTER_EXEC} run"]
    return absent + [op for op in (METHOD_ADD, METHOD_GET) if operations is not None and op not in operations]


def max_timeout_s(deployed: DeployedSandboxEnv) -> float:
    """The longest a command may run, as the card's computer-exec/v1 says."""
    return protocol_v1.extension_params(deployed.environment_card, COMPUTER_EXEC)["max_timeout_s"]


def loopback(deployed: DeployedSandboxEnv) -> dict:
    """The env's id, address and MCP endpoint as code inside the computer reaches them: the driver on loopback."""
    url = f"http://127.0.0.1:{deployed.metadata['computer']['port']}"
    return {"id": deployed.env_id, "url": url, "mcp_url": url + deployed.mcp_url.removeprefix(deployed.environment_url)}


async def add_artifacts(deployed: DeployedSandboxEnv, files: dict[str, FileArtifact],
                        url: Optional[str] = None) -> None:
    """Write each file artifact at its path, through the data plane at ``url``, the driver's or an app's it fronts; the
    driver's by default."""
    network = (deployed.metadata or {}).get("computer", {}).get("network")
    granted = not (network and NetworkPolicy.from_dict(network).restricts_egress)
    batch: list[FilePart] = []
    size = 0
    for path, artifact in files.items():
        part, part_size = await asyncio.to_thread(_part, path, artifact, deployed.sandbox_type, granted)
        if batch and size + part_size > BATCH_BYTES:
            await _add(deployed, batch, size, url)
            batch, size = [], 0
        batch.append(part)
        size += part_size
    if batch:
        await _add(deployed, batch, size, url)


async def add_files(deployed: DeployedSandboxEnv, files: dict[str, bytes]) -> None:
    """Write each file's bytes at its path, inline, in one call."""
    await _add(deployed, [_inline(path, data) for path, data in files.items()], sum(map(len, files.values())))


async def read_file(deployed: DeployedSandboxEnv, path: str) -> bytes:
    """The bytes of the file at ``path``; FileNotFoundError when there is none."""
    try:
        answer = await protocol_v1.get_data(deployed.environment_url, timeout=_READ_TIMEOUT_S, paths=[path])
    except httpx.TransportError as e:
        raise _broken(deployed, "data/get", e) from e
    for part in answer.parts:
        if (part.metadata or {}).get(DATA_OBJECTS_EXTENSION_URI, {}).get("path") == path:
            return base64.b64decode(part.file.bytes)
    raise FileNotFoundError(f"no file {path} in the computer")


async def run(deployed: DeployedSandboxEnv, argv: list[str], *, timeout_s: float,
              env: Optional[dict[str, str]] = None) -> dict:
    """computer-exec/v1 ``run``'s answer for ``argv``, under a request id of its own. A call whose connection breaks
    fails rather than being sent again: a driver that restarted would run the command a second time, beside the first,
    which outlives the driver that started it."""
    params = {"argv": argv, "request_id": uuid.uuid4().hex, "timeout_s": timeout_s, **({"env": env} if env else {})}
    try:
        return await deployed.invoke(COMPUTER_EXEC, "run", params, timeout=timeout_s + CALL_MARGIN_S)
    except httpx.TransportError as e:
        raise _broken(deployed, "computer-exec/v1 run", e) from e


async def _add(deployed: DeployedSandboxEnv, parts: list[FilePart], size: int, url: Optional[str] = None) -> None:
    try:
        await protocol_v1.add_data(url or deployed.environment_url, parts, timeout=data_plane_load_timeout_s(size))
    except httpx.TransportError as e:
        raise _broken(deployed, "data/add, so some of the files it carried may be written, each whole", e) from e


def _broken(deployed: DeployedSandboxEnv, during: str, error: httpx.TransportError) -> RuntimeError:
    """What a call to the computer's driver failed by when its connection broke, which httpx often says nothing of."""
    return RuntimeError(f"env '{deployed.env_id}': the connection to its driver broke during {during} "
                        f"({type(error).__name__}{f': {error}' if str(error) else ''})")


def _part(path: str, artifact: FileArtifact, sandbox_type: Optional[str], granted: bool) -> tuple[FilePart, int]:
    """The ``data/add`` part writing ``artifact`` at ``path``, and how many bytes the driver reads for it."""
    store = get_config().get_object_store_at(artifact.object_url)
    if granted and issues_grants_to(store, sandbox_type):
        grant = read_object(store, artifact.object_url)
        return FilePart(file=FileWithUri(uri=path, name=artifact.filename, mimeType=artifact.content_type),
                        metadata={DATA_OBJECTS_EXTENSION_URI: {"path": path, "read": grant.model_dump(mode="json")}}
                        ), grant.size_bytes
    data = artifact.load()
    return _inline(path, data, artifact.filename, artifact.content_type), len(data)


def _inline(path: str, data: bytes, name: Optional[str] = None, mime_type: Optional[str] = None) -> FilePart:
    return FilePart(file=FileWithBytes(bytes=base64.b64encode(data).decode("ascii"), name=name, mimeType=mime_type),
                    metadata={DATA_OBJECTS_EXTENSION_URI: {"path": path}})
