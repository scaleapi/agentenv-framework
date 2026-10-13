"""The infra envs agent-env builds from Dockerfiles it ships: the gateway every gateway deploy runs, the service-db its
local Postgres state runs from, and the website browser it adds for websites. Each has the bare id config names
(``default_gateway_env_id``, ``default_service_db_env_id``, ``default_website_browser_env_id``).

The put commands build them one at a time; ``ensure_default_envs`` builds the ones a run needs, into local stores only.
Each image keeps its build context beside its tar.gz, so a VM or Modal running another platform than the one it was
built on builds it again from that.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_env.artifact import DockerImageArtifact
from agent_env.config import get_config
from agent_env.env.env import Env
from agent_env.env.envs import GatewayEnv, MCPServerEnv
from agent_env.env.envs.service_db import ServiceDBEnv
from agent_env.env.envs.website_browser import WEBSITE_BROWSER_ENVIRONMENT_NAME, WEBSITE_BROWSER_IMAGE_TAG
from agent_env.store.base import NotFoundError
from agent_env.store.document_store import LocalSqliteDocumentStore
from agent_env.store.ids import derive_id
from agent_env.store.image_store import LocalRegistryImageStore
from agent_env.store.local_state import holding_locks
from agent_env.store.object_store import LocalFilesystemObjectStore
from agent_env.store.routing import configured_store
from agent_env.utils.build_context import BuildContext
from agent_env.utils.build_metadata import detect_env_metadata
from agent_env.utils.docker_build import build_image, docker_unreachable

_ENV_PACKAGE = Path(__file__).parent
GATEWAY_DOCKERFILE = _ENV_PACKAGE / "gateway" / "Dockerfile"
GATEWAY_CONTEXT = GATEWAY_DOCKERFILE.parent
GATEWAY_IMAGE_TAG = "env-gateway"
SERVICE_DB_DOCKERFILE = _ENV_PACKAGE / "envs" / "service_db" / "Dockerfile"
SERVICE_DB_IMAGE_NAME = "agent-env-service-db"
DB_WEB_DOCKERFILE = _ENV_PACKAGE / "envs" / "service_db" / "Dockerfile.db-web"
DB_WEB_IMAGE_NAME = "agent-env-db-web"
DB_MCP_DOCKERFILE = _ENV_PACKAGE / "envs" / "service_db" / "Dockerfile.db-mcp"
DB_MCP_IMAGE_NAME = "agent-env-db-mcp"
WEBSITE_BROWSER_DOCKERFILE = _ENV_PACKAGE / "envs" / "website_browser" / "Dockerfile"
WEBSITE_BROWSER_CONTEXT = WEBSITE_BROWSER_DOCKERFILE.parent

GATEWAY, SERVICE_DB, WEBSITE_BROWSER = "gateway", "service-db", "website-browser"  # the kinds, named as their commands


def _quiet(line: str) -> None:
    pass


def put_gateway_env(env_id: str, *, platform: str | None, metadata: Mapping[str, str] | None = None,
                    say: Callable[[str], None] = _quiet) -> GatewayEnv:
    """Build the gateway image for ``platform`` (this host's when None) and write it as the gateway env ``env_id``."""
    artifact = _build_and_put("gateway", derive_id(env_id, "env_image"), "Created from agent-env CLI",
                              GATEWAY_DOCKERFILE, GATEWAY_IMAGE_TAG, platform=platform, say=say)
    say("Creating GatewayEnv...")
    env = GatewayEnv.put(id=env_id, docker_image_artifact=artifact, metadata=_metadata(GATEWAY, metadata))
    say(f"Created GatewayEnv: id={env.id} version={env.version}")
    return env


def put_service_db_env(env_id: str, *, platform: str | None, metadata: Mapping[str, str] | None = None,
                       say: Callable[[str], None] = _quiet) -> ServiceDBEnv:
    """Build the service-db, db-web and db-mcp images for ``platform`` (this host's when None) and write them as the
    service-db env ``env_id``."""
    db = _build_and_put("ServiceDB", derive_id(env_id, "db_image"), f"ServiceDB PostgreSQL image for {env_id}",
                        SERVICE_DB_DOCKERFILE, SERVICE_DB_IMAGE_NAME, platform=platform, say=say)
    db_web = _build_and_put("db-web", derive_id(env_id, "db_web_image"),
                            "db-web lightweight web UI for database inspection", DB_WEB_DOCKERFILE, DB_WEB_IMAGE_NAME,
                            platform=platform, say=say)
    db_mcp = _build_and_put("db-mcp", derive_id(env_id, "db_mcp_image"),
                            "db-mcp PostgreSQL MCP server for direct DB access", DB_MCP_DOCKERFILE, DB_MCP_IMAGE_NAME,
                            platform=platform, say=say)
    say("Creating ServiceDBEnv...")
    env = ServiceDBEnv.put(id=env_id, db_docker_image_artifact=db, db_web_docker_image_artifact=db_web,
                           db_mcp_docker_image_artifact=db_mcp, metadata=_metadata(SERVICE_DB, metadata))
    say(f"Created ServiceDBEnv: id={env.id} version={env.version}")
    return env


def put_website_browser_env(env_id: str, *, platform: str | None, metadata: Mapping[str, str] | None = None,
                            say: Callable[[str], None] = _quiet) -> MCPServerEnv:
    """Build the website browser image for ``platform`` (this host's when None) and write it as the MCP server env
    ``env_id``."""
    artifact = _build_and_put("website browser", derive_id(env_id, "env_image"), "Website browser MCP server",
                              WEBSITE_BROWSER_DOCKERFILE, WEBSITE_BROWSER_IMAGE_TAG, platform=platform, say=say)
    say("Creating MCPServerEnv...")
    env = MCPServerEnv.put(id=env_id, docker_image_artifact=artifact, environment_name=WEBSITE_BROWSER_ENVIRONMENT_NAME,
                           metadata=_metadata(WEBSITE_BROWSER, metadata))
    say(f"Created MCPServerEnv: id={env.id} version={env.version} environment_name={env.environment_name}")
    return env


def _build_and_put(label: str, artifact_id: str, description: str, dockerfile: Path, image_name: str, *,
                   platform: str | None, say: Callable[[str], None]) -> DockerImageArtifact:
    """Build ``dockerfile``'s image for ``platform`` from the folder it's in, and write it with that folder as its build
    context, which a sandbox that can't run its tar.gz builds it from."""
    say(f"Building {label} image from {dockerfile}...")
    build_image(dockerfile, dockerfile.parent, image_name, platform=platform)
    say(f"Creating {label} DockerImageArtifact...")
    artifact = DockerImageArtifact.put(id=artifact_id, description=description, image_name=image_name,
                                       build_context_path=str(dockerfile.parent), dockerfile_path=dockerfile.name)
    say(f"Created {label} DockerImageArtifact: id={artifact.id} version={artifact.version}")
    return artifact


def _metadata(kind: str, extra: Mapping[str, str] | None) -> dict[str, str]:
    dockerfile = _DOCKERFILES[kind][0]
    return {**detect_env_metadata(dockerfile, dockerfile.parent), **(extra or {}),
            BUILD_INPUTS_KEY: build_inputs_digest(kind)}


BUILD_INPUTS_KEY = "build_inputs_sha256"  # the metadata key a put records build_inputs_digest under
# Each kind's Dockerfiles, each built from the folder it's in; the first is the one its env records.
_DOCKERFILES = {
    GATEWAY: [GATEWAY_DOCKERFILE],
    SERVICE_DB: [SERVICE_DB_DOCKERFILE, DB_WEB_DOCKERFILE, DB_MCP_DOCKERFILE],
    WEBSITE_BROWSER: [WEBSITE_BROWSER_DOCKERFILE],
}


def build_inputs_digest(kind: str) -> str:
    """A digest of the build contexts agent-env builds ``kind``'s images from, as each put stores them: the Dockerfiles
    it ships and the files beside them. They hold every build arg's value too, so a build from a build context alone
    builds the same thing. What a build fetches (base images, packages) isn't an input."""
    digest = hashlib.sha256()
    for dockerfile in _DOCKERFILES[kind]:
        digest.update(BuildContext.of(dockerfile.parent, dockerfile).source_digest(None).encode() + b"\0")
    return digest.hexdigest()


_PUTS = {GATEWAY: put_gateway_env, SERVICE_DB: put_service_db_env, WEBSITE_BROWSER: put_website_browser_env}


class InfraError(ValueError):
    """Infra envs a run needs and can't build, one line each in ``problems``."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class InfraBuild:
    """An infra env a run builds, and why."""

    kind: str  # GATEWAY, SERVICE_DB or WEBSITE_BROWSER
    id: str
    reason: str  # "missing", or which agent-env built the one in the store

    def __str__(self) -> str:
        return f"{self.kind} env {self.id!r} ({self.reason})"


def default_env_id(kind: str) -> str:
    config = get_config()
    return {GATEWAY: config.default_gateway_env_id, SERVICE_DB: config.default_service_db_env_id,
            WEBSITE_BROWSER: config.default_website_browser_env_id}[kind]


def put_command(kind: str) -> str:
    return f"agent-env env {kind} put --id {default_env_id(kind)}"


def infra_to_build(kinds: Iterable[str]) -> list[InfraBuild]:
    """The infra envs of ``kinds`` a run builds: each one missing from the store and, in local stores, each one agent-env
    built from the Dockerfile it ships whose recorded build inputs aren't this release's. Raises InfraError naming the
    put command for each one missing from stores that aren't all local, which agent-env never builds into."""
    kinds = _in_order(kinds)
    if not kinds:
        return []
    builds, problems = [], []
    local = stores_are_local()
    for kind in kinds:
        env_id = default_env_id(kind)
        try:
            env = Env.get(env_id)
        except NotFoundError:
            if local:
                builds.append(InfraBuild(kind, env_id, "missing"))
            else:
                problems.append(f"the {kind} env {env_id!r} isn't in the store, and agent-env builds infra envs only into "
                                f"local stores; put it with `{put_command(kind)}`")
            continue
        recorded = (env.metadata or {}).get(BUILD_INPUTS_KEY)
        if local and _built_from_stock(env, kind) and recorded != build_inputs_digest(kind):
            reason = "its build inputs changed" if recorded else "built before agent-env recorded its build inputs"
            builds.append(InfraBuild(kind, env_id, reason))
    if problems:
        raise InfraError(problems)
    return builds


def needs_docker(builds: Iterable[InfraBuild], reason: str) -> str:
    """Why ``builds`` can't be built here, given why docker can't be used (``docker_unreachable``)."""
    return f"building the {', '.join(build.kind for build in builds)} env needs docker, and {reason}"


def ensure_default_envs(kinds: Iterable[str], *, say: Callable[[str], None] = _quiet) -> list[InfraBuild]:
    """Build the infra envs of ``kinds`` that ``infra_to_build`` names, for this host's platform, and return them. Each
    id is locked while it's checked and built, so concurrent runs build it once. Raises InfraError before building
    anything when one can't be built: missing from stores that aren't local, or docker unreachable."""
    kinds = _in_order(kinds)
    if not kinds:
        return []
    ids = [default_env_id(kind) for kind in kinds]
    with holding_locks(ids, on_wait=lambda: say("waiting for another agent-env run to finish building the infra envs")):
        builds = infra_to_build(kinds)
        if builds and (reason := docker_unreachable()):
            raise InfraError([needs_docker(builds, reason)])
        for build in builds:
            say(f"{build}: building, which can take minutes")
            _PUTS[build.kind](build.id, platform=None)
    return builds


def stores_are_local() -> bool:
    """Whether the configured document, object and image stores are all the local ones."""
    config = get_config()
    return (isinstance(configured_store(config.get_document_store()), LocalSqliteDocumentStore)
            and isinstance(configured_store(config.get_object_store()), LocalFilesystemObjectStore)
            and isinstance(configured_store(config.get_image_store()), LocalRegistryImageStore))


def _built_from_stock(env: Env, kind: str) -> bool:
    """Whether ``env`` was built from the Dockerfile agent-env ships for ``kind``, in this install or another one, as
    the put commands record it; an env someone built from their own Dockerfile is theirs to rebuild."""
    recorded = (env.metadata or {}).get("dockerfile_path")
    shipped = _DOCKERFILES[kind][0].relative_to(_ENV_PACKAGE.parent.parent).parts  # ("agent_env", "env", ...)
    return recorded is not None and Path(recorded).parts[-len(shipped):] == shipped


def _in_order(kinds: Iterable[str]) -> list[str]:
    wanted = set(kinds)
    return [kind for kind in (SERVICE_DB, GATEWAY, WEBSITE_BROWSER) if kind in wanted]
