"""Offline unit tests for Daytona's egress translation and the Docker host image Daytona builds."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_env.providers.sandbox_providers.daytona.image import (
    DEFAULT_BASE_IMAGE,
    DOCKER_START_SCRIPT,
    HOST_IMAGE_VERSION,
    host_image,
)
from agent_env.providers.sandbox_providers.daytona.network import (
    MAX_CIDR_ENTRIES,
    MAX_DOMAIN_ENTRIES,
    allows_host,
    create_fields,
    policy_from_sandbox,
    policy_problem,
    refusal,
)
from agent_env.providers.sandbox_providers.daytona.provider import DaytonaSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy

def allow(*hosts: str, cidrs: tuple[str, ...] = ()) -> NetworkPolicy:
    return NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=hosts, allow_cidrs=cidrs)


def raw(domains: str | None = None, cidrs: str | None = None, block_all: bool = False) -> SimpleNamespace:
    return SimpleNamespace(domain_allow_list=domains, network_allow_list=cidrs, network_block_all=block_all)


@pytest.mark.parametrize("policy", [
    NetworkPolicy(), allow(), allow("pypi.org", "*.github.com"), allow(cidrs=("10.0.0.0/8", "1.1.1.1/32")),
])
def test_enforceable_policies_have_no_problem(policy):
    assert policy_problem(policy) is None


@pytest.mark.parametrize(("policy", "reason"), [
    (allow("pypi.org", cidrs=("10.0.0.0/8",)), "mutually exclusive"),
    (allow("1.2.3.4"), "IP address"),
    (allow(cidrs=("2001:db8::/32",)), "IPv6"),
    (allow(cidrs=("not-a-cidr",)), "is not a CIDR"),
    (allow(*(f"h{i}.example.com" for i in range(MAX_DOMAIN_ENTRIES + 1))), str(MAX_DOMAIN_ENTRIES)),
    (allow(cidrs=tuple(f"10.0.{i}.0/24" for i in range(MAX_CIDR_ENTRIES + 1))), str(MAX_CIDR_ENTRIES)),
])
def test_unenforceable_policies_name_why(policy, reason):
    problem = policy_problem(policy)
    assert problem is not None and reason in problem
    assert refusal(policy) == problem


@pytest.mark.parametrize(("policy", "refused"), [
    (NetworkPolicy(), False),
    (allow(), False),
    (allow("pypi.org", "*.github.com"), False),
    (allow(cidrs=("10.0.0.0/8",)), True),
])
def test_the_provider_also_refuses_cidrs_it_would_have_to_send_beside_the_platform_hostnames(policy, refused):
    assert (refusal(policy) is not None) is refused
    assert DaytonaSandboxProvider.supports_network_policy(policy) is not refused
    if refused:
        assert "cannot combine CIDRs" in refusal(policy)


def test_create_fields_allow_all_sends_nothing():
    assert create_fields(NetworkPolicy()) == {}


def test_create_fields_empty_allowlist_blocks_everything():
    assert create_fields(allow()) == {"network_block_all": True}


def test_create_fields_hosts_go_to_the_domain_list_and_cidrs_to_the_network_list():
    assert create_fields(allow("pypi.org", "*.github.com")) == {"domain_allow_list": "pypi.org,*.github.com"}
    assert create_fields(allow(cidrs=("10.0.0.0/8",))) == {"network_allow_list": "10.0.0.0/8"}


@pytest.mark.parametrize("policy", [
    NetworkPolicy(), allow(), allow("pypi.org", "*.github.com"), allow(cidrs=("10.0.0.0/8", "1.1.1.1/32")),
])
def test_applied_fields_read_back_as_the_same_policy(policy):
    fields = create_fields(policy)
    applied = raw(fields.get("domain_allow_list"), fields.get("network_allow_list"),
                  bool(fields.get("network_block_all")))
    assert policy_from_sandbox(applied) == policy


def test_read_back_tolerates_spaces_and_empty_entries():
    assert policy_from_sandbox(raw(" pypi.org , ,*.github.com ")) == allow("pypi.org", "*.github.com")


@pytest.mark.parametrize("applied", [
    raw("pypi.org", block_all=True), raw(cidrs="10.0.0.0/8", block_all=True), raw("pypi.org", "10.0.0.0/8"),
])
def test_read_back_of_a_state_agentenv_cannot_express_is_unknown(applied):
    assert policy_from_sandbox(applied) is None


@pytest.mark.parametrize(("host", "allowed"), [
    ("pypi.org", True),
    ("PyPI.org.", True),
    ("sub.pypi.org", False),
    ("files.pythonhosted.org", True),
    ("a.b.pythonhosted.org", True),
    ("pythonhosted.org", False),
    ("evilpythonhosted.org", False),
    ("example.com", True),
    ("evil.com", False),
])
def test_allows_host_takes_exact_names_and_wildcards_covering_every_subdomain(host, allowed):
    assert allows_host(allow("pypi.org", "*.pythonhosted.org", "Example.COM."), host) is allowed
    assert allows_host(NetworkPolicy(), host)
    assert not allows_host(allow(), host)
    assert not allows_host(None, host)


def test_default_image_builds_on_daytonas_sandbox_base_and_installs_docker_engine_and_compose():
    dockerfile = host_image().dockerfile()
    assert dockerfile.splitlines()[0] == f"FROM {DEFAULT_BASE_IMAGE}"
    for package in ("docker-ce", "docker-ce-cli", "containerd.io", "docker-compose-plugin"):
        assert package in dockerfile


def test_install_runs_as_root_and_the_image_ends_as_the_sudo_capable_daytona_user():
    lines = host_image().dockerfile().splitlines()
    assert (lines[1], lines[-1]) == ("USER root", "USER daytona")
    assert "daytona ALL=(ALL) NOPASSWD:ALL" in "\n".join(lines)


def test_docker_storage_is_left_empty_so_dockerd_prepares_it_at_boot():
    assert "rm -rf /var/lib/apt/lists/* /var/lib/docker /var/lib/containerd" in host_image().dockerfile()


def test_another_base_image_changes_only_the_from_line():
    default = host_image().dockerfile().splitlines()
    other = host_image("ubuntu:24.04").dockerfile().splitlines()
    assert other[0] == "FROM ubuntu:24.04" and other[1:] == default[1:]


def test_recipe_is_one_layer_and_versioned():
    dockerfile = host_image().dockerfile()
    assert sum(line.startswith("RUN ") for line in dockerfile.splitlines()) == 1
    assert f"agentenv-host-image={HOST_IMAGE_VERSION}" in dockerfile


def test_docker_start_script_detaches_dockerd_and_skips_a_running_daemon():
    assert DOCKER_START_SCRIPT.startswith("pgrep -x dockerd >/dev/null ||") and "setsid dockerd" in DOCKER_START_SCRIPT


def test_the_image_installs_procps_for_the_start_scripts_pgrep_on_slim_bases():
    assert "procps" in host_image().dockerfile()
