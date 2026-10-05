"""E2B VM sandbox provider.

E2B templates are immutable once built.  The configured ``base_template`` is
therefore the only template input accepted by this provider; the template
resolver derives a versioned resource-specific template for each CPU/memory
pair.  This keeps an arbitrary caller-supplied image from bypassing the
reviewed base image.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from ipaddress import ip_network
from typing import Any, ClassVar, Self

from agent_env.attribution import Attribution
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.e2b.sandbox import E2B_ALL_TRAFFIC, E2BSandbox
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SANDBOX_MODE_VM,
    SandboxProvider,
    apply_default_attribution,
)

logger = logging.getLogger(__name__)

_EXPOSED_PORTS_METADATA_KEY = "agent_env_exposed_ports"


class E2BSandboxProvider(SandboxProvider):
    """Provision Docker-capable E2B sandboxes from a configured base template.

    ``api_key`` is deliberately supplied by resolved provider config rather
    than read from the process environment.  Configure it with a ``secret:``
    interpolation in ``[sandbox.providers.e2b.config]``.
    """

    # AsyncSandbox.get_host(port) returns ``<port>-<sandbox-id>.e2b.app``.
    # Workloads receive those public URLs, so an allowlisted sandbox must be
    # able to reach the E2B ingress domain just as it can reach Modal's.
    EGRESS_HOSTS: ClassVar[tuple[str, ...]] = ("*.e2b.app",)

    def __init__(
        self,
        *,
        api_key: str,
        base_template: str,
        template_resolver: Any | None = None,
        sandbox_cls: Any | None = None,
    ):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("E2B api_key must be a non-empty resolved secret")
        if not isinstance(base_template, str) or not base_template.strip():
            raise ValueError(
                "E2B base_template must be a non-empty immutable template version"
            )
        self._api_key = api_key
        self._base_template = base_template.strip()
        self._template_resolver = template_resolver
        # Kept injectable so unit tests never instantiate an SDK client.  The
        # import stays lazy so consumers that do not select E2B do not initialize
        # SDK state while loading the provider package.
        self._sandbox_cls = sandbox_cls

    @classmethod
    def from_config(cls, **config: Any) -> Self:
        """Construct from provider config with an actionable missing-template error."""
        if not config.get("base_template"):
            raise ConfigError(
                "[sandbox.providers.e2b.config] requires a non-empty 'base_template'"
            )
        return cls(**config)

    @classmethod
    def supports_network_policy(cls, policy: NetworkPolicy) -> bool:
        # E2B's SandboxNetworkOpts accepts an outbound allow-list (domains and
        # CIDRs) at create time.
        return True

    @property
    def base_template(self) -> str:
        """The immutable, versioned base template selected in provider config."""
        return self._base_template

    def _get_sandbox_cls(self) -> Any:
        if self._sandbox_cls is None:
            from e2b import AsyncSandbox

            self._sandbox_cls = AsyncSandbox
        return self._sandbox_cls

    async def _resolve_template(self, *, cpu: float, memory: int) -> str:
        if self._template_resolver is None:
            from agent_env.providers.sandbox_providers.e2b.template import E2BTemplateResolver

            self._template_resolver = E2BTemplateResolver(api_key=self._api_key)
        return await self._template_resolver.resolve(
            self._base_template,
            cpu=cpu,
            memory_mb=memory,
        )

    async def create_vm(
        self,
        *,
        image: str | None = None,
        boot_mode: str | None = None,
        cpu: float = 1.0,
        memory: int = 8192,
        disk_size_gb: float = 10,
        timeout: int = 3600 * 2,
        exposed_ports: list[int] | None = None,
        setup_for_gateway: bool = True,
        attribution: Attribution | None = None,
        priority: int | None = None,
        network_policy: NetworkPolicy | None = None,
    ) -> E2BSandbox:
        """Create an E2B VM using its derived immutable template.

        ``boot_mode``, ``disk_size_gb`` and ``priority`` exist for
        ``SandboxProvider`` parity only: E2B's sandbox-create API has no
        corresponding controls.  In particular, disk size is never sent to
        E2B.  ``image`` is rejected rather than silently overriding the
        configured base template.
        """
        if image is not None:
            raise ValueError(
                "E2B base_template is immutable and configured by the provider; "
                "image overrides are unsupported"
            )
        del boot_mode, priority
        # ``10`` is the interface default inherited from SandboxProvider.
        # Avoid a noisy warning for every existing caller while making any
        # meaningful disk request explicit: E2B fixes disk capacity in its
        # template and does not accept a per-sandbox disk-size control.
        if disk_size_gb != 10:
            logger.warning(
                "Ignoring disk_size_gb=%s for E2B sandbox: E2B disk capacity "
                "is fixed by the selected template and cannot be configured per sandbox",
                disk_size_gb,
            )
        metadata = {
            key: value
            for key, value in apply_default_attribution(dict(attribution or {})).items()
            if value is not None
        }
        if _EXPOSED_PORTS_METADATA_KEY in metadata:
            raise ValueError(
                f"attribution key {_EXPOSED_PORTS_METADATA_KEY!r} is reserved: "
                "the E2B provider stores the sandbox's exposed ports under it"
            )
        effective_policy = self.effective_network_policy(network_policy)
        template = await self._resolve_template(cpu=cpu, memory=memory)
        ports = list(exposed_ports or [])
        if ports:
            # E2B can derive a URL for any port, so the adapter's cache is the
            # capability boundary. Persist the ports that agent-env actually
            # exposed so reconnect can restore that boundary.
            metadata[_EXPOSED_PORTS_METADATA_KEY] = ",".join(
                str(port) for port in ports
            )
        logger.info(
            "Creating E2B sandbox (template=%s, ports=%s, cpu=%s, memory=%sMB, "
            "timeout=%ss, metadata=%s)",
            template,
            ports,
            cpu,
            memory,
            timeout,
            metadata,
        )
        create_kwargs: dict[str, Any] = {
            "template": template,
            "timeout": timeout,
            "api_key": self._api_key,
            "metadata": metadata,
            # Public endpoints are the v1 behavior. E2B's authenticated
            # traffic-token flow is intentionally deferred.
            "network": {"allow_public_traffic": True},
        }
        if effective_policy.mode is NetworkMode.ALLOWLIST:
            create_kwargs["network"]["allow_out"] = [
                *effective_policy.allow_hosts,
                *effective_policy.allow_cidrs,
            ]
            create_kwargs["network"]["deny_out"] = [E2B_ALL_TRAFFIC]
        raw_sandbox = await self._get_sandbox_cls().create(**create_kwargs)
        sandbox: E2BSandbox | None = None
        try:
            sandbox = E2BSandbox(
                raw_sandbox,
                exposed_ports=ports,
                network_policy=effective_policy,
            )
            # Make the mode explicit even if an adapter implementation changes its
            # constructor default; gateway dispatch relies on this exact value.
            sandbox.mode = SANDBOX_MODE_VM
            if setup_for_gateway:
                await sandbox.setup_vm_for_gateway(ports)
            return sandbox
        except BaseException:
            try:
                if sandbox is not None:
                    await sandbox.terminate()
                else:
                    await raw_sandbox.kill()
            except Exception as cleanup_error:  # noqa: BLE001 - cleanup must not mask create failure
                logger.warning(
                    "Failed to terminate E2B sandbox %s after setup failure: %s",
                    raw_sandbox.sandbox_id,
                    cleanup_error,
                )
            raise

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
        attribution: Attribution | None = None,
        priority: int | None = None,
        network_policy: NetworkPolicy | None = None,
    ) -> E2BSandbox:
        # Like the other remote VM backends, E2B is a VM backend.  The gateway/agent
        # path loads and starts image_name after receiving the VM, so image_name
        # and env are intentionally not SDK create parameters here.
        del image_name, env
        return await self.create_vm(
            cpu=cpu,
            memory=memory,
            disk_size_gb=disk_size_gb,
            timeout=timeout,
            exposed_ports=[port],
            attribution=attribution,
            priority=priority,
            network_policy=network_policy,
        )

    @staticmethod
    def _info_field(value: Any, name: str, default: Any) -> Any:
        """Read an E2B SDK-model field, accepting its JSON spelling in tests."""
        json_name = "".join(
            part.capitalize() if index else part
            for index, part in enumerate(name.split("_"))
        )
        if isinstance(value, dict):
            return value.get(name, value.get(json_name, default))
        return getattr(value, name, default)

    @staticmethod
    def _is_sdk_unset(value: Any) -> bool:
        # ``e2b.api.client.types.UNSET`` intentionally stays out of the module
        # imports so this provider's SDK import remains lazy.
        return type(value).__name__ == "Unset"

    @classmethod
    def _network_policy_from_info(cls, info: Any, *, sandbox_id: str) -> NetworkPolicy:
        """Translate E2B's current egress config into agent-env's contract.

        E2B returns the applied network configuration from ``get_info``.  Do
        not infer an unrestricted policy when that response contains a policy
        this adapter cannot represent: image loading may extend a restrictive
        policy with object-store hosts, and extending the wrong one is a
        security bug.
        """
        missing = object()
        network = cls._info_field(info, "network", missing)
        if network is missing or cls._is_sdk_unset(network) or network is None:
            raise RuntimeError(
                f"Cannot recover E2B network policy for sandbox {sandbox_id}: "
                "get_info returned no network configuration"
            )

        allow_internet = cls._info_field(info, "allow_internet_access", missing)
        deny_out = cls._info_field(network, "deny_out", missing)
        deny_out_missing = deny_out is missing or cls._is_sdk_unset(deny_out)
        if allow_internet is False:
            raise RuntimeError(
                f"Cannot recover E2B network policy for sandbox {sandbox_id}: "
                "the applied deny/allow-internet policy is not representable by agent-env"
            )

        if not deny_out_missing and deny_out not in (None, [], [E2B_ALL_TRAFFIC]):
            raise RuntimeError(
                f"Cannot recover E2B network policy for sandbox {sandbox_id}: "
                "the applied deny/allow-internet policy is not representable by agent-env"
            )

        allow_out = cls._info_field(network, "allow_out", missing)
        allow_out_missing = allow_out is missing or cls._is_sdk_unset(allow_out)
        has_deny_all = deny_out == [E2B_ALL_TRAFFIC]
        if allow_out_missing and not has_deny_all:
            return NetworkPolicy()
        if allow_out_missing:
            allow_out = []
        elif not has_deny_all:
            raise RuntimeError(
                f"Cannot recover E2B network policy for sandbox {sandbox_id}: "
                "allow_out was present without E2B's canonical deny-all rule"
            )
        if not isinstance(allow_out, (list, tuple)) or not all(
            isinstance(destination, str) and destination for destination in allow_out
        ):
            raise RuntimeError(
                f"Cannot recover E2B network policy for sandbox {sandbox_id}: "
                f"unexpected allow_out value {allow_out!r}"
            )

        hosts: list[str] = []
        cidrs: list[str] = []
        for destination in allow_out:
            try:
                ip_network(destination, strict=False)
            except ValueError:
                hosts.append(destination)
            else:
                cidrs.append(destination)
        return NetworkPolicy(
            mode=NetworkMode.ALLOWLIST,
            allow_hosts=tuple(hosts),
            allow_cidrs=tuple(cidrs),
        )

    @classmethod
    def _exposed_ports_from_info(cls, info: Any, *, sandbox_id: str) -> tuple[int, ...]:
        """Recover the provider-managed tunnel capabilities stored at creation."""
        missing = object()
        metadata = cls._info_field(info, "metadata", missing)
        if metadata is missing or cls._is_sdk_unset(metadata) or metadata is None:
            return ()
        if not isinstance(metadata, Mapping):
            logger.warning(
                "Cannot recover exposed ports for E2B sandbox %s: unexpected metadata %r",
                sandbox_id,
                metadata,
            )
            return ()
        encoded = metadata.get(_EXPOSED_PORTS_METADATA_KEY)
        if not encoded:
            return ()
        try:
            ports = tuple(
                dict.fromkeys(int(value) for value in str(encoded).split(","))
            )
            if any(port < 1 or port > 65535 for port in ports):
                raise ValueError
            return ports
        except ValueError:
            logger.warning(
                "Cannot recover exposed ports for E2B sandbox %s: invalid %s=%r",
                sandbox_id,
                _EXPOSED_PORTS_METADATA_KEY,
                encoded,
            )
            return ()

    async def get_sandbox(self, sandbox_id: str) -> E2BSandbox:
        sandbox = await E2BSandbox.reconnect(
            sandbox_id,
            api_key=self._api_key,
            sandbox_cls=self._get_sandbox_cls(),
        )
        try:
            info = await sandbox._sandbox.get_info()
            for port in self._exposed_ports_from_info(info, sandbox_id=sandbox_id):
                sandbox.get_host(port)
            sandbox.network_policy = self._network_policy_from_info(
                info, sandbox_id=sandbox_id
            )
        except Exception as exc:
            logger.warning(
                "Could not recover the applied E2B network policy for sandbox %s; "
                "reconnected for inspection/cleanup, but image loading will fail closed: %s",
                sandbox_id,
                exc,
            )
            sandbox.network_policy = None
        return sandbox
