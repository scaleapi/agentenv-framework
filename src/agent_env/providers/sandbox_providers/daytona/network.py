"""AgentEnv ``NetworkPolicy`` <-> Daytona egress settings.

Daytona keeps hostnames (``domain_allow_list``, ``*.`` wildcards) and IPv4 CIDRs
(``network_allow_list``) in separate, mutually exclusive lists, and blocks everything with
``network_block_all``. A policy is translated exactly or refused; nothing is resolved or dropped.
"""

from __future__ import annotations

from ipaddress import ip_address, ip_network
from typing import Any

from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy

MAX_DOMAIN_ENTRIES = 100
MAX_CIDR_ENTRIES = 10


def _is_ip(entry: str) -> bool:
    try:
        ip_address(entry)
    except ValueError:
        return False
    return True


def policy_problem(policy: NetworkPolicy) -> str | None:
    """Why Daytona cannot enforce ``policy``, or None when it can."""
    if not policy.restricts_egress:
        return None
    hosts, cidrs = policy.allow_hosts, policy.allow_cidrs
    if hosts and cidrs:
        return (
            "Daytona's domain and CIDR allowlists are mutually exclusive; allow hostnames or CIDRs"
        )
    if ips := [host for host in hosts if _is_ip(host)]:
        return f"Daytona allows hostnames, not IP addresses, as hosts: {ips}"
    if len(hosts) > MAX_DOMAIN_ENTRIES:
        return f"Daytona allows at most {MAX_DOMAIN_ENTRIES} hostnames, got {len(hosts)}"
    if len(cidrs) > MAX_CIDR_ENTRIES:
        return f"Daytona allows at most {MAX_CIDR_ENTRIES} CIDRs, got {len(cidrs)}"
    for cidr in cidrs:
        try:
            network = ip_network(cidr, strict=False)
        except ValueError:
            return f"{cidr!r} is not a CIDR"
        if network.version == 6:
            return f"Daytona filters IPv4 only; {cidr!r} is IPv6"
    return None


def refusal(policy: NetworkPolicy) -> str | None:
    """Why the provider refuses ``policy``: ``policy_problem``, or CIDRs, which AgentEnv's always-added
    platform hostnames would put in the mutually exclusive domain list beside."""
    if problem := policy_problem(policy):
        return problem
    if policy.restricts_egress and policy.allow_cidrs:
        return (
            "Daytona cannot combine CIDRs with the platform hostnames AgentEnv always allows; "
            "allow hostnames instead"
        )
    return None


def create_fields(policy: NetworkPolicy) -> dict[str, Any]:
    """``policy`` as ``CreateSandbox*Params`` fields; ``policy_problem(policy)`` must be None."""
    if not policy.restricts_egress:
        return {}
    if policy.allow_hosts:
        return {"domain_allow_list": ",".join(policy.allow_hosts)}
    if policy.allow_cidrs:
        return {"network_allow_list": ",".join(policy.allow_cidrs)}
    return {"network_block_all": True}


def _entries(value: str | None) -> tuple[str, ...]:
    return tuple(entry.strip() for entry in (value or "").split(",") if entry.strip())


def policy_from_sandbox(raw: Any) -> NetworkPolicy | None:
    """The policy a Daytona sandbox applies, or None when AgentEnv cannot express it."""
    hosts = _entries(getattr(raw, "domain_allow_list", None))
    cidrs = _entries(getattr(raw, "network_allow_list", None))
    block_all = bool(getattr(raw, "network_block_all", False))
    if (hosts and cidrs) or (block_all and (hosts or cidrs)):
        return None
    if block_all:
        return NetworkPolicy(mode=NetworkMode.ALLOWLIST)
    if hosts or cidrs:
        return NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=hosts, allow_cidrs=cidrs)
    return NetworkPolicy()


def allows_host(policy: NetworkPolicy | None, host: str) -> bool:
    """Whether a sandbox under ``policy`` can reach ``host``: allow-all, an exact hostname, or a ``*.domain`` entry,
    which covers every name under ``domain``. An unknown policy counts as no."""
    if policy is None:
        return False
    if not policy.restricts_egress:
        return True
    host = host.lower().rstrip(".")
    for entry in policy.allow_hosts:
        entry = entry.lower().rstrip(".")
        if entry == host or (entry.startswith("*.") and host.endswith(entry[1:])):
            return True
    return False
