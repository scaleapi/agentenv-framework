import asyncio

import click

from agent_env.config import get_config
from agent_env.env import Env
from agent_env.providers.env_providers.env_provider import _env_provider_class
from agent_env.providers.env_providers.env_server_provider import EnvironmentServerProvider
from agent_env.store.base import NotFoundError
from agent_env.utils.build_metadata import detect_base_metadata, detect_env_metadata  # noqa: F401  re-exported for the put commands
from agent_env.utils.docker_build import DEFAULT_BUILD_PLATFORM


def build_platform_option(f):
    """Shared `--platform` option for CLI commands that run `docker build` locally.

    Defaults to linux/amd64 (the remote sandbox VMs are amd64). Pass linux/arm64 (or an
    empty string for a host-native build) to produce an image that runs under
    `env deploy --sandbox local` on Apple Silicon; such an image will NOT run on
    remote amd64 sandboxes.
    """
    return click.option(
        "--platform",
        "build_platform",
        default=DEFAULT_BUILD_PLATFORM,
        show_default=True,
        help=(
            "docker build target platform. Default linux/amd64 (the remote sandbox VMs). "
            "Use linux/arm64 for a host-native `env deploy --sandbox local` build on "
            "Apple Silicon; pass an empty string to omit --platform entirely."
        ),
    )(f)


def deprecated_option(old: str, dest: str, replacement: str):
    """A hidden `old` flag, the deprecated spelling of `replacement`: using it notes the replacement on stderr.
    Merge the two with :func:`renamed_value`."""
    def note(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
        if value is not None:
            click.echo(f"Warning: {old} is deprecated and will be removed; use {replacement}.", err=True)
        return value

    return click.option(old, dest, default=None, hidden=True, callback=note)


def renamed_value(new: str, new_value: str | None, old: str, old_value: str | None) -> str | None:
    """The value of flag `new`, or of its deprecated spelling `old`; giving both is a usage error."""
    if new_value is not None and old_value is not None:
        raise click.UsageError(f"{old} is the deprecated spelling of {new}; give only {new}")
    return old_value if new_value is None else new_value


def refuse_unwritable_ids(*ids: str) -> None:
    """Refuse, before anything is built, an id the store a put writes it to wouldn't take, such as an @local id the
    image's suffix makes too long."""
    store = get_config().get_document_store()
    for entity_id in ids:
        try:
            store.check_id(entity_id)
        except ValueError as e:
            raise click.ClickException(str(e)) from None


def env_provider_type_option(help: str, env_type: str = "mcp_server"):
    """Shared `--env-provider-type` option (default `gateway`), refused at parse time unless an installed provider has that type, and
    for an env other than one MCP server, unless that provider deploys more than one."""
    def check(ctx: click.Context, param: click.Parameter, value: str) -> str:
        try:
            provider_class = _env_provider_class(value)
        except ValueError as e:
            raise click.BadParameter(str(e)) from e
        if env_type != "mcp_server" and issubclass(provider_class, EnvironmentServerProvider):
            raise click.BadParameter(f"'{value}' deploys one MCP server, not a {env_type} env")
        return value

    return click.option("--env-provider-type", "env_provider_type", default="gateway", show_default=True, callback=check, help=help)


def environment_name_options(f):
    """Shared `--environment-name` option.

    Not declared `required`: commands that can derive the name from the environment card
    resolve it through :func:`resolve_environment_name` with `allow_missing=True`.
    """
    return click.option(
        "--environment-name",
        "environment_name",
        default=None,
        help="Environment name (Env Card identity; used for tool prefixes, DB schema, compose naming)",
    )(f)


def resolve_environment_name(environment_name: str | None, *, allow_missing: bool = False) -> str | None:
    """Resolve `--environment-name` to a name. When it is omitted and `allow_missing` is set (the
    MCP-server / website put path), returns None so the caller can derive the name from the
    environment's card; otherwise a missing name is a UsageError.
    """
    if not environment_name:
        if allow_missing:
            return None
        raise click.UsageError("Missing option '--environment-name'")
    return environment_name


def parse_artifact_ref(ref: str) -> tuple[str, int | None]:
    """Parse 'id' or 'id:version'. Raises click.BadParameter on malformed input."""
    if ":" not in ref:
        if not ref:
            raise click.BadParameter("artifact ref must be non-empty")
        return ref, None
    id_part, _, version_part = ref.partition(":")
    if not id_part or not version_part or ":" in version_part:
        raise click.BadParameter(
            f"Expected 'id' or 'id:<positive int>', got {ref!r}"
        )
    if not version_part.isdecimal():
        raise click.BadParameter(
            f"Version must be a positive integer, got {version_part!r} in {ref!r}"
        )
    version = int(version_part)
    if version < 1:
        raise click.BadParameter(
            f"Version must be >= 1, got {version} in {ref!r}"
        )
    return id_part, version


def deployed_env_from_instance(env_id: str | None, instance_id: str) -> Env:
    """Rehydrate the live deployed env behind ``instance_id``.

    Goes through ``Env.from_instance_id``, which reconnects the sandbox via the
    provider named by the instance's ``sandbox_type`` — so it works on every
    backend, unlike deriving an id from the MCP URL.
    """
    try:
        env = asyncio.run(Env.from_instance_id(instance_id))
    except NotFoundError:
        raise click.UsageError(f"instance '{instance_id}' not found")
    if env_id and env_id != env.id:
        raise click.UsageError(
            f"--id '{env_id}' does not match instance '{instance_id}' (env '{env.id}')")
    return env
