"""Pluggable sandbox provider — the compute backend behind every env and agent sandbox."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Self
from urllib.parse import urlparse

from agent_env.plugins import _registration
from agent_env.attribution import Attribution
from agent_env.config import runtime
from agent_env.config import ENV_REF_PREFIX
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.sandbox import (
    NetworkPolicy,
    NetworkPolicyUnsupportedError,
    Sandbox,
    VmSandbox,
    port_bindings,
)

if TYPE_CHECKING:
    from agent_env.config.runtime import Config

logger = logging.getLogger(__name__)

SANDBOX_MODE_VM = "vm"
SANDBOX_MODE_CONTAINER = "container"


class SandboxProviderTypeError(ConfigError):
    """A config-registered provider produced a Sandbox whose ``.type`` != its ``[sandbox.providers.<name>]`` key."""


async def _pull(sandbox: VmSandbox, image_name: str) -> None:
    """``docker pull image_name``. An image built for linux/amd64 only has nothing for an arm64 host (an
    Apple Silicon Mac running the local provider), so that pull falls back to the amd64 image, which the
    host's Docker runs emulated."""
    try:
        await sandbox.exec_script(f"docker pull {shlex.quote(image_name)}")
    except RuntimeError as e:
        if "no matching manifest" not in str(e):
            raise
        logger.warning("%s has no image for this host's platform; pulling linux/amd64, which runs emulated", image_name)
        await sandbox.exec_script(f"docker pull --platform linux/amd64 {shlex.quote(image_name)}")


class SandboxProvider(ABC):
    """Compute backend that provisions sandboxes. Selected via [sandbox] default / build_sandbox_provider(); swap via set_sandbox_provider()."""

    @abstractmethod
    async def create_sandbox(
        self,
        *,
        image_name: str,
        port: int,
        env: dict[str, str],
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        attribution: Optional[Attribution] = None,
        network_policy: Optional[NetworkPolicy] = None,
    ) -> Sandbox: ...

    async def create_vm(
        self,
        *,
        image: Optional[str] = None,
        boot_mode: Optional[str] = None,
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        exposed_ports: Optional[list[int]] = None,
        setup_for_gateway: bool = True,
        attribution: Optional[Attribution] = None,
        network_policy: Optional[NetworkPolicy] = None,
    ) -> VmSandbox:
        raise NotImplementedError(f"{type(self).__name__} does not support create_vm")

    async def create_container(
        self,
        *,
        image_name: str,
        port: int,
        env: dict[str, str],
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        attribution: Optional[Attribution] = None,
        network_policy: Optional[NetworkPolicy] = None,
    ) -> Sandbox:
        """Provision a sandbox with the registry image already running as a container.

        On return, ``tunnel_urls[port]`` is populated and the container listens on ``port``.
        Caller still owns readiness polling for the container's own endpoints.

        Default implementation: create_vm, authenticate Docker to ECR if applicable, docker pull,
        docker run. Backends that bake the image in at sandbox-creation time (Modal) override.
        """
        sandbox = await self.create_vm(
            cpu=cpu, memory=memory, disk_size_gb=disk_size_gb,
            exposed_ports=[port], timeout=timeout,
            attribution=attribution,
            network_policy=network_policy,
        )
        try:
            from agent_env.config import get_config

            auth = await asyncio.to_thread(get_config().get_image_store().auth, image_name)
            if auth is not None:
                await sandbox.exec_script(
                    f"echo {shlex.quote(auth.password)} | docker login "
                    f"--username {shlex.quote(auth.username)} --password-stdin {shlex.quote(auth.registry)}"
                )
            await _pull(sandbox, image_name)
            await self._start_container(sandbox, image_name=image_name, port=port, env=env)
            sandbox.mode = SANDBOX_MODE_CONTAINER
            return sandbox
        except BaseException:
            try:
                await sandbox.terminate()
            except Exception:
                pass
            raise

    async def _start_container(self, sandbox: VmSandbox, *, image_name: str, port: int, env: dict[str, str]) -> None:
        """Run the pulled ``image_name`` as the sandbox's container, publishing ``port``."""
        args = self._container_args(sandbox, image_name=image_name, port=port, env=env)
        await sandbox.exec_script(f"docker run -d {args} > /dev/null")

    def _container_args(self, sandbox: VmSandbox, *, image_name: str, port: int, env: dict[str, str]) -> str:
        """The ``docker run`` / ``docker create`` arguments for the sandbox's container."""
        env_flags = " \\\n    ".join(f"-e {shlex.quote(k)}={shlex.quote(v)}" for k, v in env.items())
        extra_args = f"{self.EXTRA_CONTAINER_RUN_ARGS} " if self.EXTRA_CONTAINER_RUN_ARGS else ""
        publish = " ".join(f"-p {spec}" for spec in port_bindings(sandbox.host_ips, sandbox.host_port(port), port))
        return (
            f"--name {shlex.quote(sandbox.container_name)} {publish} {extra_args}\\\n    "
            f"{env_flags} \\\n    {shlex.quote(image_name)}"
        )

    async def get_sandbox(self, sandbox_id: str) -> Sandbox:
        raise NotImplementedError(f"{type(self).__name__} does not support get_sandbox")

    async def close(self) -> None:
        pass

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        """Construct from a resolved config table; backends that need custom wiring override this."""
        return cls(**config)

    # Internal→external host rewrites for URLs issued on this platform ({} = none).
    URL_REWRITES: ClassVar[dict[str, str]] = {}

    # Extra flags spliced into the agent container's `docker run` in create_container ("" = none).
    # Backends set this for host-specific needs (e.g. local Linux needs --add-host so the
    # externalized host.docker.internal endpoint resolves inside the container).
    EXTRA_CONTAINER_RUN_ARGS: ClassVar[str] = ""

    @classmethod
    def get_external_url(cls, url: str) -> str:
        """Rewrite a platform-issued URL to its externally reachable form.

        Applies ``URL_REWRITES`` longest-key-first (keys may nest); identity
        when nothing matches.
        """
        for host in sorted(cls.URL_REWRITES, key=len, reverse=True):
            if host in url:
                return url.replace(host, cls.URL_REWRITES[host])
        return url

    @classmethod
    def shares_network_with(cls, sandbox_type: Optional[str]) -> bool:
        """Whether a workload on ``sandbox_type`` can reach this platform's internal URLs.

        Default: sandboxes of the same provider share a network. Override to widen
        (multi-type platform families) or narrow (isolated tenancy).
        """
        if sandbox_type is None:
            return False
        try:
            return _sandbox_provider_class(sandbox_type) is cls
        except Exception:
            return False

    # Env vars workloads need to reach this platform's URLs: name -> env:/secret: ref or literal.
    CONTAINER_ENV: ClassVar[dict[str, str]] = {}

    # Headers for this platform's URLs: host pattern ("*.suffix" or exact) -> header -> env:NAME
    # (a CONTAINER_ENV entry) or literal.
    REQUEST_HEADERS: ClassVar[dict[str, dict[str, str]]] = {}

    # Hosts on this platform a restricted workload must still reach.
    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        """Whether this backend can enforce ``policy``. Default: unrestricted only."""
        return not policy.restricts_egress

    @classmethod
    def effective_network_policy(cls, policy: Optional[NetworkPolicy]) -> NetworkPolicy:
        """``policy`` with the platform's own hosts unioned in; identity unless it allowlists."""
        if policy is None:
            return NetworkPolicy()
        return policy.with_hosts(all_sandbox_egress_hosts())


# --- Provider registry: built-ins + config.toml [sandbox.providers] ---

_DEFAULT_SANDBOX_SPEC = "local"
_DEFAULT_AGENT_SANDBOX_SPEC = "local"

_BUILTIN_SANDBOX_PROVIDERS: dict[str, str] = {
    "modal": "agent_env.providers.sandbox_providers.modal_sandbox:ModalSandboxProvider",
    "modal_vm": "agent_env.providers.sandbox_providers.modal_vm_sandbox:ModalVmSandboxProvider",
    "e2b": "agent_env.providers.sandbox_providers.e2b:E2BSandboxProvider",
    "sail": "agent_env.providers.sandbox_providers.sail:SailSandboxProvider",
    "local": "agent_env.providers.sandbox_providers.local_sandbox:LocalSandboxProvider",
}



def _sandbox_config(source: Config | None = None) -> dict:
    

    return (source or runtime.get_config()).section("sandbox")


def apply_default_attribution(attribution: Attribution) -> Attribution:
    """``attribution`` with every dimension the caller left unset (absent or None) filled
    from config.toml ``[sandbox.attribution]``. Values take ``env:NAME?default`` references."""
    from agent_env.config import interpolate

    resolved = dict(attribution)
    for name, value in interpolate(_sandbox_config().get("attribution", {})).items():
        if resolved.get(name) is None:
            resolved[name] = value
    return resolved


def _default_sandbox_spec() -> str:
    return _sandbox_config().get("default", _DEFAULT_SANDBOX_SPEC)


def _agent_sandbox_spec() -> str:
    return _sandbox_config().get("agent_default", _DEFAULT_AGENT_SANDBOX_SPEC)


def _build_registry(source: Config | None = None) -> dict[str, dict]:
    """Built-ins, then ``agent_env.sandbox_providers`` plugins, then config.toml
    ``[sandbox.providers]`` declared in ``source``."""
    registry: dict[str, dict] = {name: {"impl": impl} for name, impl in _BUILTIN_SANDBOX_PROVIDERS.items()}
    from_plugins = _registration.merge(
        registry,
        _registration.SANDBOX_PROVIDERS,
        _validate_plugin,
        source=source,
        entry=lambda cls: {"impl": cls},
    )
    _merge_config_toml_sandbox_providers(registry, source=source, from_plugins=from_plugins)
    return registry


def _validate_plugin(name: str, loaded: Any) -> type[SandboxProvider]:
    cls = _registration.require_subclass(loaded, SandboxProvider)
    if problem := _registration.unimplemented(cls, SandboxProvider):
        raise TypeError(problem)
    return cls


def _get_sandbox_registry() -> dict[str, dict]:
    from agent_env.config import runtime

    return runtime.get_config().sandbox_registry()


def _merge_config_toml_sandbox_providers(
    registry: dict[str, dict],
    *,
    source: Config | None = None,
    from_plugins: _registration.Registrations | None = None,
) -> None:
    """Register ``[sandbox.providers]`` entries (a ``module:Class`` string or a table with ``impl`` +
    optional ``config``) under their name. A table for a BUILT-IN name may carry ``config`` only —
    a built-in's impl cannot be replaced from config. A table for a PLUGIN's name may carry
    ``config`` only, or an ``impl`` that replaces the plugin's with a warning. Other collisions
    and bad impls fail loud."""
    from agent_env.config import ConfigError, load_impl

    from_plugins = from_plugins if from_plugins is not None else _registration.Registrations.empty()
    for name, entry in _sandbox_config(source).get("providers", {}).items():
        where = f"[sandbox.providers.{name}]"
        plugin = from_plugins.plugin(name)
        config_only = isinstance(entry, dict) and "impl" not in entry
        if config_only and name not in registry and from_plugins.failed(name):
            if not from_plugins.refuse(name, where):
                logger.warning("%s configures a plugin that failed to load; skipped", where)
            continue
        if name in registry and (config_only or plugin is None):
            if not config_only:
                raise ConfigError(
                    f"config.toml sandbox provider {name!r} collides with a built-in backend: "
                    "a built-in's impl cannot be replaced; supply a config-only table instead"
                )
            owner = "a built-in backend" if plugin is None else f"plugin {plugin}"
            unknown = sorted(set(entry) - {"config"})
            if unknown:
                raise ConfigError(
                    f"[sandbox.providers.{name}] configures {owner} and accepts "
                    f"only a 'config' table; got {unknown}"
                )
            config = entry.get("config", {})
            if not isinstance(config, dict):
                raise ConfigError(f"[sandbox.providers.{name}].config must be a table, got {type(config).__name__}")
            registry[name] = {**registry[name], "config": dict(config)}
            continue
        section = {"impl": entry} if isinstance(entry, str) else dict(entry)
        impl = section.get("impl")
        if not impl:
            raise ConfigError(f"config.toml sandbox provider {name!r} is missing an 'impl' dotted path")
        cls = load_impl(impl, SandboxProvider)
        if problem := _registration.unimplemented(cls, SandboxProvider):
            raise ConfigError(f"{where} impl {impl!r}: {problem}")
        if from_plugins.refuse(name, where):
            continue
        from_plugins.release(name, f"sandbox.providers.{name}", impl, cls)
        registry[name] = section


def _sandbox_provider_class(sandbox_type: str) -> type[SandboxProvider]:
    """The provider class registered for a single sandbox type (no instantiation)."""
    from agent_env.config import load_impl

    registry = _get_sandbox_registry()
    if sandbox_type not in registry:
        raise ValueError(
            f"Unknown sandbox type {sandbox_type!r}{_registration.failure_note(_registration.SANDBOX_PROVIDERS, sandbox_type)}; "
            f"known: {sorted(registry)}"
        )
    return load_impl(registry[sandbox_type]["impl"], SandboxProvider)


def reachable_url(url: str, *, from_sandbox_type: Optional[str], to_sandbox_type: Optional[str]) -> str:
    """``url``, issued on ``from_sandbox_type``, in the form reachable by a
    workload on ``to_sandbox_type``.

    Internal URLs are kept when both sides share a network. Unknown platforms
    leave the URL unchanged; ``from_sandbox_type=None`` resolves to the default
    spec's primary provider.
    """
    from_type = (from_sandbox_type or _default_sandbox_spec()).split(",")[0].strip()
    try:
        provider_cls = _sandbox_provider_class(from_type)
    except ValueError:
        return url
    if provider_cls.shares_network_with(to_sandbox_type):
        return url
    return provider_cls.get_external_url(url)


def registered_sandbox_provider_classes() -> list[type[SandboxProvider]]:
    """Every registered provider class, de-duplicated, in registry order.

    Best-effort: an unbuildable registry or an unloadable provider is skipped
    rather than raised. These feed *contribution* helpers on deploy paths — a
    provider nobody is deploying to must not be able to fail a deploy. Genuine
    config errors still fail loud wherever a provider is actually built.
    """
    try:
        names = list(_get_sandbox_registry())
    except Exception:
        logger.warning("Sandbox registry unavailable; no provider reachability contributions", exc_info=True)
        return []
    classes: list[type[SandboxProvider]] = []
    for name in names:
        try:
            cls = _sandbox_provider_class(name)
        except Exception:
            logger.debug("Skipping reachability contributions for unloadable sandbox provider %r", name)
            continue
        if cls not in classes:
            classes.append(cls)
    return classes


def all_sandbox_url_rewrites() -> dict[str, str]:
    """Union of every registered provider's ``URL_REWRITES``."""
    rewrites: dict[str, str] = {}
    for cls in registered_sandbox_provider_classes():
        rewrites.update(cls.URL_REWRITES)
    return rewrites


def all_sandbox_container_env() -> dict[str, str]:
    """Union of every registered provider's resolved ``CONTAINER_ENV``.

    Deliberately provider-agnostic: a workload gets the credentials for every
    registered platform, because at injection time we do not know which
    platform's URLs it will be handed.
    """
    env: dict[str, str] = {}
    for cls in registered_sandbox_provider_classes():
        try:
            env.update(_resolve_container_env(cls))
        except Exception:
            logger.warning("Provider %s failed to contribute container env; skipping", cls.__name__, exc_info=True)
    return env


def all_sandbox_egress_hosts() -> tuple[str, ...]:
    """Every sandbox platform's own hosts, so a restricted workload can still reach its env.

    Provider-agnostic like ``all_sandbox_container_env``: at deploy time we do not know
    which platform's URLs the workload will be handed.
    """
    hosts: list[str] = []
    for cls in registered_sandbox_provider_classes():
        hosts += [*cls.URL_REWRITES, *cls.URL_REWRITES.values(), *cls.EGRESS_HOSTS]
    kept: dict[str, None] = {}
    for host in hosts:
        if host == "*":
            logger.warning("Dropping a bare '*' egress host; it would defeat the allowlist")
        elif host:
            kept.setdefault(host, None)
    return tuple(kept)


def sandbox_request_headers_for_url(url: str) -> dict[str, str]:
    """Union of every registered provider's headers for ``url``.

    Each provider's host patterns gate it, so a URL only collects headers
    from the platform that issued it.
    """
    headers: dict[str, str] = {}
    for cls in registered_sandbox_provider_classes():
        try:
            if not cls.REQUEST_HEADERS:
                continue
            host = urlparse(url).hostname or ""
            wanted = {
                header: value
                for pattern, entries in cls.REQUEST_HEADERS.items()
                if (host.endswith(pattern[1:]) if pattern.startswith("*.") else host == pattern)
                for header, value in entries.items()
            }
            if not wanted:
                continue
            needs_env = any(v.startswith(ENV_REF_PREFIX) for v in wanted.values())
            env = _resolve_container_env(cls) if needs_env else {}
            if needs_env and not env:
                if not cls.CONTAINER_ENV:
                    logger.warning("%s REQUEST_HEADERS reference env vars but CONTAINER_ENV is empty", cls.__name__)
                continue
            headers.update({
                header: env[value[len(ENV_REF_PREFIX):]] if value.startswith(ENV_REF_PREFIX) else value
                for header, value in wanted.items()
            })
        except Exception:
            logger.warning("Provider %s failed to contribute request headers; skipping", cls.__name__, exc_info=True)
    return headers


def _resolve_container_env(provider_cls: type[SandboxProvider]) -> dict[str, str]:
    """``provider_cls.CONTAINER_ENV`` resolved to values.

    A process env var named like the entry wins over its reference. All-or-nothing:
    when any entry is unresolvable the provider contributes nothing, so a
    half-configured platform cannot inject half its credentials.
    """
    from agent_env.config import get_config, interpolate

    env: dict[str, str] = {}
    for name, ref in provider_cls.CONTAINER_ENV.items():
        value = os.environ.get(name)
        if not value:
            try:
                value = interpolate(ref, secret_resolver=lambda key: get_config().get_secret_store().get(key))
            except Exception:
                logger.warning(
                    "%s container env unresolvable (%s); workloads off its network may not reach its URLs",
                    provider_cls.__name__, name,
                )
                return {}
        env[name] = value
    return env


def refuse_unenforceable_policy(provider: SandboxProvider, policy: Optional[NetworkPolicy]) -> None:
    """Fail before provisioning rather than return a sandbox that ignores ``policy``."""
    if policy is not None and not provider.supports_network_policy(policy):
        raise NetworkPolicyUnsupportedError(
            f"{type(provider).__name__} cannot enforce network policy mode {policy.mode.value!r}"
        )


def _install_sandbox_type_guard(provider: SandboxProvider, name: str) -> SandboxProvider:
    """Wrap the provider's ``create_*`` so every sandbox it makes has ``.type == name``; on mismatch,
    terminate it and raise. Decorated in place to preserve provider identity."""
    import functools

    def _guarded(method):
        @functools.wraps(method)
        async def wrapper(*args, **kwargs):
            sandbox = await method(*args, **kwargs)
            if sandbox is not None and sandbox.type != name:
                actual = sandbox.type
                try:
                    await sandbox.terminate()
                except Exception as e:
                    logger.warning(f"Failed to terminate mis-typed sandbox from [sandbox.providers.{name}]: {e}")
                raise SandboxProviderTypeError(
                    f"[sandbox.providers.{name}] produced a sandbox with .type={actual!r}; the config name "
                    f"must equal the produced Sandbox.type. Rename the key to {actual!r} or set the type to {name!r}."
                )
            return sandbox

        return wrapper

    for method_name in ("create_sandbox", "create_vm", "create_container"):
        try:
            setattr(provider, method_name, _guarded(getattr(provider, method_name)))
        except AttributeError as e:
            raise ConfigError(
                f"Cannot install the deploy-time type guard on sandbox provider {name!r} "
                f"({type(provider).__name__}): its instances don't allow attribute assignment "
                f"(e.g. __slots__ without __dict__)."
            ) from e
    return provider


def _build_sandbox_provider_from_section(name: str, section: dict) -> SandboxProvider:
    from agent_env.config import interpolate, load_impl
    from agent_env.config import get_config

    cls = load_impl(section["impl"], SandboxProvider)
    config = interpolate(
        section.get("config", {}), secret_resolver=lambda key: get_config().get_secret_store().get(key)
    )
    provider = cls.from_config(**config)
    if name in _BUILTIN_SANDBOX_PROVIDERS:
        return provider
    return _install_sandbox_type_guard(provider, name)


def build_sandbox_provider(spec: str) -> SandboxProvider:
    """Build a provider (or a comma-separated fallback chain) from backend names,
    resolved through the open registry (built-ins + config.toml [sandbox.providers])."""
    registry = _get_sandbox_registry()
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if not names:
        raise ValueError("sandbox spec must include at least one backend")
    for n in names:
        if n not in registry:
            raise ValueError(
                f"Unknown sandbox backend: {n!r}{_registration.failure_note(_registration.SANDBOX_PROVIDERS, n)} "
                f"(expected one of {sorted(registry)})"
            )
    providers = [_build_sandbox_provider_from_section(n, registry[n]) for n in names]
    if len(providers) == 1:
        return providers[0]
    from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider

    return ChainedSandboxProvider(providers)


# --- Default sandbox provider (envs, agent provider, CLI, task steps) ---

_sandbox_provider: Optional[SandboxProvider] = None


def get_sandbox_provider() -> SandboxProvider:
    global _sandbox_provider
    if _sandbox_provider is None:
        _sandbox_provider = build_sandbox_provider(_default_sandbox_spec())
    return _sandbox_provider


def set_sandbox_provider(provider: SandboxProvider) -> None:
    global _sandbox_provider
    _sandbox_provider = provider


def reset_sandbox_provider() -> None:
    global _sandbox_provider
    _sandbox_provider = None


_env_sandbox_provider: Optional[SandboxProvider] = None


def get_env_sandbox_provider() -> SandboxProvider:
    global _env_sandbox_provider
    if _env_sandbox_provider is None:
        _env_sandbox_provider = build_sandbox_provider(_default_sandbox_spec())
    return _env_sandbox_provider


def set_env_sandbox_provider(provider: SandboxProvider) -> None:
    global _env_sandbox_provider
    _env_sandbox_provider = provider


def reset_env_sandbox_provider() -> None:
    global _env_sandbox_provider
    _env_sandbox_provider = None


_agent_sandbox_provider: Optional[SandboxProvider] = None


def get_agent_sandbox_provider() -> SandboxProvider:
    global _agent_sandbox_provider
    if _agent_sandbox_provider is None:
        _agent_sandbox_provider = build_sandbox_provider(_agent_sandbox_spec())
    return _agent_sandbox_provider


def set_agent_sandbox_provider(provider: SandboxProvider) -> None:
    global _agent_sandbox_provider
    _agent_sandbox_provider = provider


def reset_agent_sandbox_provider() -> None:
    global _agent_sandbox_provider
    _agent_sandbox_provider = None
