"""Provider settings, and how an AgentEnv request becomes a Daytona sandbox's size, name and labels."""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from daytona import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams, Resources

from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.daytona.image import DEFAULT_BASE_IMAGE, HOST_IMAGE_VERSION, host_image

SECTION = "[sandbox.providers.daytona.config]"
DEFAULT_API_URL = "https://app.daytona.io/api"
CONFIG_KEYS = ("api_key", "api_url", "target", "snapshot", "base_image", "egress_hosts")

# Docker-in-Docker needs at least 2 vCPU and 4 GiB (Daytona's guidance); 10 GiB holds images.
MIN_CPU = 2
MIN_MEMORY_GIB = 4
MIN_DISK_GIB = 10

MANAGED_LABEL = "agentenv.managed"
PORTS_LABEL = "agentenv.ports"
HOST_IMAGE_LABEL = "agentenv.host-image"
ATTRIBUTION_LABEL_PREFIX = "agentenv.attr."
LABEL_VALUE_MAX = 255
_NAME_MAX = 63
# The first create from the host image builds it; a snapshot only boots.
IMAGE_CREATE_TIMEOUT = 1800
SNAPSHOT_CREATE_TIMEOUT = 300

CreateParams = CreateSandboxFromImageParams | CreateSandboxFromSnapshotParams
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Settings:
    """``[sandbox.providers.daytona.config]``, parsed. The key never appears in ``repr``."""

    api_key: str = field(repr=False)
    api_url: str = DEFAULT_API_URL
    target: str | None = None
    snapshot: str | None = None
    base_image: str = DEFAULT_BASE_IMAGE
    egress_hosts: tuple[str, ...] = ()


def _optional_string(config: Mapping[str, Any], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{SECTION} '{key}' must be a non-empty string")
    return value.strip()


def parse_settings(config: Mapping[str, Any]) -> Settings:
    """Parse the provider's config table, refusing anything it does not understand."""
    if unknown := sorted(set(config) - set(CONFIG_KEYS)):
        raise ConfigError(f"{SECTION} has unknown key(s): {unknown}")
    api_key = config.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ConfigError(
            f"{SECTION} requires a non-empty 'api_key' (e.g. \"env:DAYTONA_API_KEY\")"
        )
    snapshot, base_image = _optional_string(config, "snapshot"), _optional_string(
        config, "base_image"
    )
    if snapshot and base_image:
        raise ConfigError(f"{SECTION} takes either 'snapshot' or 'base_image', not both")
    hosts = config.get("egress_hosts", [])
    if not isinstance(hosts, list) or not all(
        isinstance(h, str) and h.strip() and h.strip() != "*" for h in hosts
    ):
        raise ConfigError(f"{SECTION} 'egress_hosts' must be a list of hostnames (no bare '*')")
    return Settings(
        api_key=api_key.strip(),
        api_url=_optional_string(config, "api_url") or DEFAULT_API_URL,
        target=_optional_string(config, "target"),
        snapshot=snapshot,
        base_image=base_image or DEFAULT_BASE_IMAGE,
        egress_hosts=tuple(h.strip() for h in hosts),
    )


def sandbox_resources(*, cpu: float, memory_mb: int, disk_gb: float) -> Resources:
    """The request rounded up to whole units and raised to the Docker floor."""
    return Resources(
        cpu=max(MIN_CPU, math.ceil(cpu)),
        memory=max(MIN_MEMORY_GIB, math.ceil(memory_mb / 1024)),
        disk=max(MIN_DISK_GIB, math.ceil(disk_gb)),
    )


def ttl_minutes(timeout_seconds: int) -> int:
    """AgentEnv's sandbox ``timeout`` as Daytona's hard wall-clock lifetime."""
    return max(1, math.ceil(timeout_seconds / 60))


def _slug(value: object) -> str:
    return _UNSAFE.sub("-", str(value)).strip("-")


def sandbox_labels(
    attribution: Mapping[str, Any], ports: Iterable[int], *, host_image: bool
) -> dict[str, str]:
    """Labels for attribution (cost joins), reconnect (ports), sweeping (the managed marker) and, on a
    sandbox booted from the host image, the recipe version that built it."""
    labels = {MANAGED_LABEL: "true", PORTS_LABEL: ",".join(str(port) for port in ports)}
    if host_image:
        labels[HOST_IMAGE_LABEL] = HOST_IMAGE_VERSION
    for key, value in attribution.items():
        if (
            value is not None
            and (key_slug := _slug(key))
            and (slug := _slug(value)[:LABEL_VALUE_MAX])
        ):
            labels[f"{ATTRIBUTION_LABEL_PREFIX}{key_slug}"] = slug
    return labels


def sandbox_name(attribution: Mapping[str, Any]) -> str:
    """``ae-<random>`` plus the attribution values in key order: unique, findable by people."""
    slugs = [
        re.sub(r"[^A-Za-z0-9-]+", "-", str(attribution[key])).strip("-")
        for key in sorted(attribution)
        if attribution[key] is not None
    ]
    return "-".join(["ae", uuid.uuid4().hex[:8], *filter(None, slugs)])[:_NAME_MAX].rstrip("-")


def create_params(
    settings: Settings, common: Mapping[str, Any], resources: Resources
) -> tuple[CreateParams, int]:
    """Params booting the configured snapshot, or the host image at ``resources``; and how long a
    create may take."""
    if settings.snapshot:
        params: CreateParams = CreateSandboxFromSnapshotParams(snapshot=settings.snapshot, **common)
        return params, SNAPSHOT_CREATE_TIMEOUT
    image = host_image(settings.base_image)
    return (
        CreateSandboxFromImageParams(image=image, resources=resources, **common),
        IMAGE_CREATE_TIMEOUT,
    )
