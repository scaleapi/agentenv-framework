"""The infra envs agent-env builds from Dockerfiles it ships: the gateway every gateway deploy runs, the service-db its
local Postgres state runs from, and the website browser it adds for websites. Each has the bare id config names
(``default_gateway_env_id``, ``default_service_db_env_id``, ``default_website_browser_env_id``).

The put commands build them one at a time; ``ensure_default_envs`` builds the ones a run needs, into local stores only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from agent_env.artifact import DockerImageArtifact
from agent_env.config import get_config
from agent_env.env.env import Env
from agent_env.env.envs import GatewayEnv, MCPServerEnv
from agent_env.env.envs.service_db import ServiceDBEnv
from agent_env.env.envs.website_browser import (
    PLAYWRIGHT_MCP_VERSION,
    WEBSITE_BROWSER_ENVIRONMENT_NAME,
    WEBSITE_BROWSER_IMAGE_TAG,
)
from agent_env.store.base import NotFoundError
from agent_env.store.document_store import LocalSqliteDocumentStore
from agent_env.store.image_store import LocalRegistryImageStore
from agent_env.store.local_state import holding_locks
from agent_env.store.object_store import LocalFilesystemObjectStore
from agent_env.store.routing import configured_store
from agent_env.utils.build_metadata import detect_env_metadata
from agent_env.utils.docker_build import build_image, docker_unreachable

_ENV_PACKAGE = Path(__file__).parent
GATEWAY_DOCKERFILE = _ENV_PACKAGE / "gateway" / "Dockerfile"
GATEWAY_CONTEXT = _ENV_PACKAGE
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
    say("Building gateway Docker image...")
    build_image(GATEWAY_DOCKERFILE, GATEWAY_CONTEXT, GATEWAY_IMAGE_TAG, platform=platform)
    say("Creating DockerImageArtifact...")
    artifact = DockerImageArtifact.put(id=f"gateway-{env_id}", description="Created from agent-env CLI",
                                       image_name=GATEWAY_IMAGE_TAG)
    say(f"Created artifact: id={artifact.id} version={artifact.version}")
    say("Creating GatewayEnv...")
    env = GatewayEnv.put(id=env_id, docker_image_artifact=artifact,
                         metadata=_metadata(GATEWAY_DOCKERFILE, GATEWAY_CONTEXT, metadata))
    say(f"Created GatewayEnv: id={env.id} version={env.version}")
    return env


def put_service_db_env(env_id: str, *, platform: str | None, metadata: Mapping[str, str] | None = None,
                       say: Callable[[str], None] = _quiet) -> ServiceDBEnv:
    """Build the service-db, db-web and db-mcp images for ``platform`` (this host's when None) and write them as the
    service-db env ``env_id``."""
    artifacts = []
    for label, dockerfile, image, artifact_id, description in (
        ("ServiceDB", SERVICE_DB_DOCKERFILE, SERVICE_DB_IMAGE_NAME, f"service-db-{env_id}",
         f"ServiceDB PostgreSQL image for {env_id}"),
        ("db-web", DB_WEB_DOCKERFILE, DB_WEB_IMAGE_NAME, f"db-web-{env_id}",
         "db-web lightweight web UI for database inspection"),
        ("db-mcp", DB_MCP_DOCKERFILE, DB_MCP_IMAGE_NAME, f"db-mcp-{env_id}",
         "db-mcp PostgreSQL MCP server for direct DB access"),
    ):
        say(f"Building {label} image from {dockerfile}...")
        build_image(dockerfile, dockerfile.parent, image, platform=platform)
        say(f"Creating {label} DockerImageArtifact...")
        artifacts.append(DockerImageArtifact.put(id=artifact_id, description=description, image_name=image))
        say(f"Created {label} DockerImageArtifact: id={artifacts[-1].id} version={artifacts[-1].version}")
    say("Creating ServiceDBEnv...")
    env = ServiceDBEnv.put(
        id=env_id,
        db_docker_image_artifact=artifacts[0],
        db_web_docker_image_artifact=artifacts[1],
        db_mcp_docker_image_artifact=artifacts[2],
        metadata=_metadata(SERVICE_DB_DOCKERFILE, SERVICE_DB_DOCKERFILE.parent, metadata),
    )
    say(f"Created ServiceDBEnv: id={env.id} version={env.version}")
    return env


def put_website_browser_env(env_id: str, *, platform: str | None, metadata: Mapping[str, str] | None = None,
                            say: Callable[[str], None] = _quiet) -> MCPServerEnv:
    """Build the website browser image for ``platform`` (this host's when None) and write it as the MCP server env
    ``env_id``."""
    say("Building website browser Docker image...")
    build_image(WEBSITE_BROWSER_DOCKERFILE, WEBSITE_BROWSER_CONTEXT, WEBSITE_BROWSER_IMAGE_TAG, platform=platform,
                build_args={"PLAYWRIGHT_MCP_VERSION": PLAYWRIGHT_MCP_VERSION})
    say("Creating DockerImageArtifact...")
    artifact = DockerImageArtifact.put(id=f"website-browser-{env_id}", description="Website browser MCP server",
                                       image_name=WEBSITE_BROWSER_IMAGE_TAG)
    say(f"Created artifact: id={artifact.id} version={artifact.version}")
    say("Creating MCPServerEnv...")
    env = MCPServerEnv.put(id=env_id, docker_image_artifact=artifact, environment_name=WEBSITE_BROWSER_ENVIRONMENT_NAME,
                           metadata=_metadata(WEBSITE_BROWSER_DOCKERFILE, WEBSITE_BROWSER_CONTEXT, metadata))
    say(f"Created MCPServerEnv: id={env.id} version={env.version} environment_name={env.environment_name}")
    return env


def _metadata(dockerfile: Path, context: Path, extra: Mapping[str, str] | None) -> dict[str, str] | None:
    return {**detect_env_metadata(dockerfile, context), **(extra or {})} or None


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
    """The infra envs of ``kinds`` a run builds: each one missing from the store and, in local stores, each one another
    agent-env release built, whose images may predate this release's gateway. Raises InfraError naming the put command
    for each one missing from stores that aren't all local, which agent-env never builds into."""
    builds, problems = [], []
    local = stores_are_local()
    running = _agent_env_version()
    for kind in _in_order(kinds):
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
        built_by = (env.metadata or {}).get("agent_env_version")
        if local and built_by and running and built_by != running:
            builds.append(InfraBuild(kind, env_id, f"built by agent-env {built_by}, this is {running}"))
    if problems:
        raise InfraError(problems)
    return builds


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
            raise InfraError([f"building the {', '.join(build.kind for build in builds)} env needs docker, and {reason}"])
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


def _in_order(kinds: Iterable[str]) -> list[str]:
    wanted = set(kinds)
    return [kind for kind in (SERVICE_DB, GATEWAY, WEBSITE_BROWSER) if kind in wanted]


def _agent_env_version() -> str | None:
    try:
        return version("agentenv-framework")
    except PackageNotFoundError:
        return None
