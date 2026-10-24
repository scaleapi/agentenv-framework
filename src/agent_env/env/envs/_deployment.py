"""The env side of the environment-provider contract: deploying an env through its provider, the options a provider takes,
the checks on a plugin's record, loading into a deployment a plugin's provider made, and closing a deployment."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Optional

from agentenv_protocol import METHOD_ADD, METHOD_RESET, RPC_PATH, FilePart, client as protocol_v1
from agent_env.env import legacy_protocol
from agent_env.env.env import DeployedSandboxEnv, gateway_url_of
from agent_env.env.store import register_env_instance
from agent_env.store.base import ObjectNotFoundError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from agent_env.artifact import FileArtifact
    from agent_env.env.env import DeployedEnv, Env
    from agent_env.env.envs.mcp_server import MCPServerEnv
    from agent_env.env.envs.website import WebsiteEnv
    from agent_env.providers.env_providers.env_provider import EnvironmentProvider, _SandboxEnvironmentProvider

logger = logging.getLogger(__name__)

LOAD_OPERATIONS = (METHOD_RESET, METHOD_ADD)  # the data-plane calls a load makes, in order, as a card lists them


async def deploy_through_provider(env: Env, *, environment_name: str | None, ttl_seconds: int, sandbox_type: str | None,
                                  **options) -> DeployedEnv:
    """Deploy env through the provider its env_provider_type names and register the record, the way each built-in env deploys.

    The env's provider deploys it the first time; a deploy of an env that already has a deployment gets a provider of its own,
    so that a failure closes only what it started and leaves the earlier deployment as it was. A deploy onto the env state
    instance that deployment holds is refused, since the attempt's close() would retire it. A success keeps the earlier
    deployment's provider and sandbox on the env, so its close() still tears that deployment down. A deploy refused (see
    ``deploy_refusal``) or a bad attribution fails before anything changes. A built-in's sandbox is set on the env before the
    record is registered, so close() reaches it; in container mode that is the container of the server named
    ``environment_name``, None for a MultiEnv, whose servers each have their own. A plugin's record is checked instead.
    """
    from agent_env.providers import build_env_provider, build_sandbox_provider, get_env_sandbox_provider

    held = getattr(env._deployed, "env_state_instance_ids", None) or []
    if (instance := options.get("env_state_instance_id")) and instance in held:
        raise ValueError(f"env '{env.id}' already has a deployment, instance {env._deployed.instance_id!r}, on env state instance "
                         f"{instance!r}; one state instance backs one deployment, so deploy another env object onto it")
    reuse = env._env_provider is not None and env._deployed is None
    provider = env._env_provider if reuse else build_env_provider(env.env_provider_type)
    if options.get("attribution") is not None:
        options["attribution"] = dict(options["attribution"])
    options = provider_options(env, provider, ttl_seconds=ttl_seconds, **options)
    sandbox_provider = build_sandbox_provider(sandbox_type) if sandbox_type else get_env_sandbox_provider()
    earlier = env._env_provider, env._sandbox, env._gateway_url, env._instance_id, env._deployed, env._replaced
    owned = env._deployed is not None and env._deployed.env_id == env.id  # a MultiEnv's child holds its parent's record
    replaced = [*env._replaced, (env._env_provider, env._sandbox)] if owned else env._replaced
    env._env_provider, env._sandbox, env._replaced = provider, None, []
    try:
        deployed = await provider.deploy(env, sandbox_provider, **options)
        if builtin := as_builtin(provider):
            env._sandbox = builtin.environment_sandbox(environment_name) or builtin.sandbox  # first, so a refusal's close() reaches it
            if not isinstance(deployed, DeployedSandboxEnv):
                raise TypeError(f"env_provider_type '{provider.type}' subclasses a built-in provider, so it is handled as one: its "
                                f"deploy() must return the built-in's kind of record, not a {type(deployed).__name__}")
        else:
            check_plugin_record(env, deployed)
        env._gateway_url = gateway_url_of(deployed)
        deployed = register_env_instance(deployed, ttl_seconds)
        env._instance_id = deployed.instance_id
        env._deployed, env._replaced = deployed, replaced
        return deployed
    except BaseException:
        try:
            await env.close()
        finally:
            env._env_provider, env._sandbox, env._gateway_url, env._instance_id, env._deployed, env._replaced = earlier
        raise


async def close_replaced(env: Env) -> None:
    """Tear down each deployment a later deploy of env replaced: its provider, then its sandbox. Each close() takes an entry
    off the env before working on it, so two at once never share one, and a cancellation puts back what it didn't finish."""
    while env._replaced:
        provider, sandbox = env._replaced.pop(0)
        try:
            if provider is not None:
                try:
                    await provider.close()
                except Exception as e:
                    logger.warning(f"Failed to close the env provider of a replaced deployment: {e}")
                provider = None
            if sandbox is not None:
                try:
                    await sandbox.terminate()
                except Exception as e:
                    logger.warning(f"Failed to terminate sandbox {sandbox.sandbox_id}: {e}")
        except BaseException:
            env._replaced.insert(0, (provider, sandbox))
            raise


async def load_by_signed_url(env: MCPServerEnv | WebsiteEnv, file_artifact: FileArtifact) -> None:
    """Load into a server a plugin's provider deployed, which has no sandbox of ours to stage into: it fetches the file from a
    signed URL. The server may be a MultiEnv's child, whose record is the MultiEnv's."""
    from agent_env.config import get_config
    from agent_env.env.gateway.constants import data_plane_load_timeout_s

    deployed_by = f"env '{env._deployed.env_id}' was deployed by env_provider_type '{env._deployed.env_provider_type}'"
    if not env._deployed.environment_card:
        raise RuntimeError(f"{deployed_by}, whose record carries no env card, so '{env.environment_name}' has no data plane to load into")
    base_url = await legacy_protocol.v1_base_url(env._deployed, env._gateway_url, env.environment_name, mcp=True)
    operations = ((env._deployed.get_child_env_card(env.environment_name) or {}).get("capabilities") or {}).get("operations")
    if base_url is None:
        raise RuntimeError(f"{deployed_by}, whose card offers '{env.environment_name}' no data plane to load into")
    if operations is not None and (missing := [op for op in LOAD_OPERATIONS if op not in operations]):
        raise RuntimeError(f"{deployed_by}, whose card offers '{env.environment_name}' no {', '.join(missing)} operation to load with")
    store = get_config().get_object_store_at(file_artifact.object_url)
    metadata = await asyncio.to_thread(store.get_object_metadata_at, file_artifact.object_url)
    if metadata is None:
        raise ObjectNotFoundError(f"No object at {file_artifact.object_url}")
    timeout = data_plane_load_timeout_s(metadata.size)
    # Signed before the reset, so a store that can't sign leaves the env's data as it is; it lasts one timeout per call.
    url = await asyncio.to_thread(store.signed_get_url, file_artifact.object_url, len(LOAD_OPERATIONS) * timeout)
    if url is None:
        raise RuntimeError(f"{deployed_by}, so its server fetches {file_artifact.object_url} itself, and {type(store).__name__} "
                           "can't sign a URL for it; loading into such an env needs an object store that signs URLs (S3, Cloud Storage)")
    await protocol_v1.reset_data(base_url, timeout=timeout)
    await protocol_v1.add_data(base_url, [FilePart(file={
        "uri": url,
        "mimeType": file_artifact.content_type,
        "name": file_artifact.filename,
    })], timeout=timeout)


async def close_deployed(deployed: DeployedEnv, env_class: type[Env]) -> None:
    """Close what a task deployed of an env_class env: a built-in's through its env, a plugin's by terminating the sandboxes its record names."""
    from agent_env.task_step.task_steps.teardown_sandboxes import _env_sandbox_ids, _terminate

    env = await env_class.from_deployed_env(deployed)
    if as_builtin(env._env_provider):
        await env.close()
        return
    sandboxes = _env_sandbox_ids(deployed)
    if not sandboxes:
        logger.warning(f"env '{deployed.env_id}' was deployed by env_provider_type '{deployed.env_provider_type}' outside "
                       "agent-env's sandboxes, so it isn't closed here; the deployment must end by itself")
    results = await asyncio.gather(*(_terminate(sandbox_id, sandbox_type, "env") for sandbox_id, sandbox_type in sandboxes),
                                   return_exceptions=True)
    for (sandbox_id, _), result in zip(sandboxes, results):
        if isinstance(result, BaseException):
            logger.warning(f"env '{deployed.env_id}': terminate {sandbox_id} failed: {result!r}")


def check_plugin_record(env: Env, deployed: DeployedEnv) -> None:
    """Refuse a record a plugin's provider returned that the env couldn't use: one of another type, without an MCP URL, whose card
    isn't the env's or gives a child env a url that isn't a path, or for a MultiEnv doesn't list each of its servers and websites
    or names its MCP server other than the env's name."""
    from agent_env.env.envs.multi_env import MultiEnv

    returned = f"env '{env.id}': env_provider_type '{env.env_provider_type}' returned a record"
    if deployed.env_provider_type != env.env_provider_type:
        raise RuntimeError(f"{returned} with env_provider_type {deployed.env_provider_type!r}; set it to {env.env_provider_type!r}")
    if bool(deployed.environment_card) != bool(deployed.environment_card_url):
        missing = "environment_card_url" if deployed.environment_card else "environment_card"
        raise RuntimeError(f"{returned} with no {missing}; an env card and its environment_card_url come together")
    if not deployed.mcp_url:
        raise RuntimeError(f"{returned} with no MCP URL; a record carries it in its env card (environment_card and "
                           "environment_card_url) or in mcp_url")
    if not deployed.environment_card:
        if isinstance(env, MultiEnv):
            raise RuntimeError(f"{returned} with no env card; a MultiEnv's children are reached through the card, which lists each "
                               "of its MCP servers and websites")
        return
    if isinstance(env, MultiEnv):
        names = [child.environment_name for child in [*env.mcp_server_envs, *env.website_envs]]
        if missing := [name for name in names if deployed.get_child_env_card(name) is None]:
            raise RuntimeError(f"{returned} whose env card lists no child env named {', '.join(map(repr, missing))}; a MultiEnv's card "
                               "lists each of its MCP servers and websites by environment_name")
        if env.name and deployed.mcp_server_name != env.name:
            raise RuntimeError(f"{returned} whose MCP server is named {deployed.mcp_server_name!r}; agents see it under the "
                               f"record's mcp_server_name, or its card's name, which must be the env's name {env.name!r}")
    elif deployed.get_child_env_card(env.environment_name) is None:
        raise RuntimeError(f"{returned} whose env card is named {deployed.environment_card.get('name')!r}; it must be the env's "
                           f"environment_name {env.environment_name!r}, or list a child env of that name")
    else:
        names = [env.environment_name]
    if not_paths := [name for name in names if not deployed.get_child_env_card(name).get("url", RPC_PATH).startswith("/")]:
        raise RuntimeError(f"{returned} whose env card gives {', '.join(map(repr, not_paths))} a url that isn't a path; a child env's "
                           "url is a path under the record's address, such as '/svc/<name>/agentenv'")


def plugin_deployment(env: Env) -> DeployedEnv | None:
    """The env's record if a plugin's provider deployed it, leaving no host of ours to stage onto; None for a built-in's, closed or not,
    or no deployment."""
    from agent_env.providers.env_providers.env_provider import record_class_for

    deployed = env._deployed
    if deployed is None or as_builtin(env._env_provider) is not None or record_class_for(deployed.env_provider_type) is not None:
        return None
    return deployed


def host_staging_refusal(env: Env, what: str) -> str | None:
    """Why ``what``, which stages files onto the env's host, is refused: a plugin's provider deployed it; None otherwise."""
    if (deployed := plugin_deployment(env)) is None:
        return None
    return f"{what} needs a built-in env provider; env '{deployed.env_id}' was deployed by env_provider_type '{deployed.env_provider_type}'"


def builtin_provider_for(env_provider_type: str) -> Optional[_SandboxEnvironmentProvider]:
    """A fresh built-in provider of that type, or None for a plugin's type."""
    from agent_env.providers.env_providers.env_provider import _builtin_env_providers
    builtin = _builtin_env_providers().get(env_provider_type)
    return builtin() if builtin is not None else None


def plugin_provider_like_a_builtin(env_provider_type: str) -> Optional[_SandboxEnvironmentProvider]:
    """A fresh provider of a plugin's type that subclasses a built-in, so it reattaches as one; None for any other, or one not installed."""
    from agent_env.providers.env_providers.env_provider import _SandboxEnvironmentProvider, _env_provider_class
    try:
        provider_class = _env_provider_class(env_provider_type)
    except ValueError:
        return None
    return provider_class() if issubclass(provider_class, _SandboxEnvironmentProvider) else None


def as_builtin(provider: Optional[EnvironmentProvider]) -> Optional[_SandboxEnvironmentProvider]:
    """The provider if it is a built-in or subclasses one, whose sandboxes loads stage into and reattach and close() reach; None for any other plugin's."""
    from agent_env.providers.env_providers.env_provider import _SandboxEnvironmentProvider
    return provider if isinstance(provider, _SandboxEnvironmentProvider) else None


def provider_options(env: Env, provider: EnvironmentProvider, **options) -> dict:
    """The options provider takes: every one for a deploy(**options), else those it names, one it doesn't being dropped at deploy()'s
    default; raises for a deploy refused (see ``deploy_refusal``)."""
    if refusal := deploy_refusal(env, provider, options):
        raise ValueError(refusal)
    taken = inspect.signature(provider.deploy).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in taken.values()):
        return options
    return {name: value for name, value in options.items() if name in taken}


def provider_or_class(env: Env) -> EnvironmentProvider | type[EnvironmentProvider]:
    """The env's provider, or before one is built, the registered class its env_provider_type names; raises for a type this process can't find."""
    from agent_env.providers.env_providers.env_provider import _env_provider_class
    return env._env_provider if env._env_provider is not None else _env_provider_class(env.env_provider_type)


def deploy_refusal(env: Env, provider: EnvironmentProvider | type[EnvironmentProvider], options: dict) -> str | None:
    """Why deploying env through provider, or a provider of that class, is refused before anything is built, or None: the provider
    can't deploy an env like it (see ``provider_refusal``), or an option it can't take."""
    provider_class = provider if isinstance(provider, type) else type(provider)
    reason = provider_refusal(provider_class, type(env),
                              mcp_names=[child.environment_name for child in getattr(env, "mcp_server_envs", ())],
                              website_names=[child.environment_name for child in getattr(env, "website_envs", ())])
    if reason is not None:
        return f"env '{env.id}' has env_provider_type '{env.env_provider_type}', which {reason}"
    return option_refusal(env, provider, options)


def provider_refusal(provider_class: type[EnvironmentProvider], env_class: type[Env], *, mcp_names: Iterable[str] = (),
                     website_names: Iterable[str] = ()) -> str | None:
    """Why a provider of provider_class can't deploy an env of env_class, a multi's with children of these environment_names, worded
    to follow the provider's type, or None: the server provider deploys one MCP server, and a plugin's provider gives a multi one
    env card, which can't tell an MCP server and a website of one name apart. It needs no env, so a bundle checks it before writing one."""
    from agent_env.env.envs.mcp_server import MCPServerEnv
    from agent_env.env.envs.multi_env import MultiEnv
    from agent_env.providers.env_providers.env_provider import _SandboxEnvironmentProvider
    from agent_env.providers.env_providers.env_server_provider import EnvironmentServerProvider

    if issubclass(provider_class, EnvironmentServerProvider) and not issubclass(env_class, MCPServerEnv):
        return f"deploys one MCP server, not a {env_class.type} env"
    if issubclass(env_class, MultiEnv) and not issubclass(provider_class, _SandboxEnvironmentProvider):
        if shared := set(mcp_names) & set(website_names):
            return (f"gives a multi one env card, so it can't tell an MCP server and a website apart by name, and both are named "
                    f"{', '.join(map(repr, sorted(shared)))}")
    return None


def option_refusal(env: Env, provider: EnvironmentProvider | type[EnvironmentProvider], options: dict) -> str | None:
    """Why provider, or a provider of that class, wouldn't take options: one its deploy() doesn't name, set away from the env's deploy() default."""
    from agent_env.providers.env_providers.env_provider import _SandboxEnvironmentProvider
    taken = inspect.signature(provider.deploy).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in taken.values()):
        return None
    defaults = inspect.signature(type(env).deploy).parameters
    refused = [name for name, value in options.items() if name not in taken and value != defaults[name].default]
    if not refused:
        return None
    builtin = issubclass(provider if isinstance(provider, type) else type(provider), _SandboxEnvironmentProvider)
    how = "" if builtin else "; name it in the provider's deploy() or take **options"
    return f"env '{env.id}' has env_provider_type '{env.env_provider_type}', which doesn't take {', '.join(refused)}{how}"
