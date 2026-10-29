"""Provider settings, and how an AgentEnv request becomes a boat sandbox's type, name and labels."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_env.config.errors import ConfigError

logger = logging.getLogger(__name__)

SECTION = "[sandbox.providers.boat.config]"
DEFAULT_API_URL = "https://boat.dev/api/v1"
CONFIG_KEYS = ("api_key", "api_url")

# boat's sandbox types, smallest first: (type, vCPU, GiB of memory, GB of disk).
SIZES = (("small", 2, 4, 51), ("default", 4, 8, 51), ("large", 8, 16, 111), ("xlarge", 16, 32, 251))

MANAGED_LABEL = "agentenv.managed"
PORTS_LABEL = "agentenv.ports"
# Names a sandbox by the create that made it, so a create that failed or was cancelled is found and deleted.
CREATE_LABEL = "agentenv.create"
ATTRIBUTION_LABEL_PREFIX = "agentenv.attr."
MAX_LABELS = 20
LABEL_KEY_MAX = 63
LABEL_VALUE_MAX_BYTES = 256
_NAME_MAX = 120
_UNSAFE_KEY = re.compile(r"[^A-Za-z0-9._/-]+")
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9-]+")


@dataclass(frozen=True)
class Settings:
    """``[sandbox.providers.boat.config]``, parsed. The key never appears in ``repr``."""

    api_key: str = field(repr=False)
    api_url: str = DEFAULT_API_URL


def parse_settings(config: Mapping[str, Any]) -> Settings:
    """Parse the provider's config table, refusing anything it does not understand."""
    if unknown := sorted(set(config) - set(CONFIG_KEYS)):
        raise ConfigError(f"{SECTION} has unknown key(s): {unknown}")
    api_key = config.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ConfigError(f"{SECTION} requires a non-empty 'api_key' (e.g. \"secret:boat_api_key\")")
    api_url = config.get("api_url", DEFAULT_API_URL)
    if not isinstance(api_url, str) or not api_url.strip():
        raise ConfigError(f"{SECTION} 'api_url' must be a non-empty string")
    return Settings(api_key=api_key.strip(), api_url=api_url.strip().rstrip("/"))


def sandbox_type(*, cpu: float, memory_mb: int, disk_gb: float) -> str:
    """The smallest boat type that covers the request; a request bigger than the largest is refused."""
    for name, vcpu, memory_gib, disk in SIZES:
        if cpu <= vcpu and memory_mb <= memory_gib * 1024 and disk_gb <= disk:
            return name
    name, vcpu, memory_gib, disk = SIZES[-1]
    raise ValueError(
        f"boat cannot size a sandbox for cpu={cpu}, memory={memory_mb}MB, disk={disk_gb}GB: its largest type, "
        f"{name!r}, has {vcpu} vCPU, {memory_gib} GiB memory and {disk} GB disk"
    )


def _label_value(value: object) -> str:
    """``value`` cut to boat's limit on a label value, at a character boundary."""
    return str(value).encode()[:LABEL_VALUE_MAX_BYTES].decode(errors="ignore")


def sandbox_labels(attribution: Mapping[str, Any], ports: Iterable[int], create_token: str) -> dict[str, str]:
    """Labels for reconnect (ports), cleanup (the create token, the managed marker) and attribution (cost joins). boat
    keeps at most ``MAX_LABELS``; attribution past that is logged and left off."""
    labels = {MANAGED_LABEL: "true", PORTS_LABEL: ",".join(str(port) for port in ports), CREATE_LABEL: create_token}
    for key, value in attribution.items():
        slug = _UNSAFE_KEY.sub("-", str(key)).strip("-./")
        label = f"{ATTRIBUTION_LABEL_PREFIX}{slug}"[:LABEL_KEY_MAX]
        if value is None or not slug:
            continue
        if len(labels) == MAX_LABELS:
            logger.warning("Ignoring attribution %r on a boat sandbox: boat keeps at most %s labels", key, MAX_LABELS)
            continue
        labels[label] = _label_value(value)
    return labels


def sandbox_name(attribution: Mapping[str, Any]) -> str:
    """``ae-<random>`` plus the attribution values in key order: unique, findable by people."""
    slugs = [
        _UNSAFE_NAME.sub("-", str(attribution[key])).strip("-")
        for key in sorted(attribution)
        if attribution[key] is not None
    ]
    return "-".join(["ae", uuid.uuid4().hex[:8], *filter(None, slugs)])[:_NAME_MAX].rstrip("-")


def ports_from_labels(labels: Mapping[str, str] | None) -> list[int]:
    """The exposed ports a sandbox's labels record."""
    return [int(port) for port in str((labels or {}).get(PORTS_LABEL, "")).split(",") if port.strip().isdigit()]
