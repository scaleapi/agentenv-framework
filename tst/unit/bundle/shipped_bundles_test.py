"""Every bundle agentenv-framework ships parses, resolves and passes the checks a run makes, and a task naming no agent
plans with the shipped default. A run that pulls one entity from an installed bundle reads all of that bundle, so one
broken folder in a shipped bundle would fail every run pulling from it, the default agent's among them."""

from __future__ import annotations

import json

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.bundle import BundleKind, parse_bundle
from agent_env.bundle.installed import CORE, installed_bundles
from agent_env.bundle.plan import check_bundle, plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.config import get_config
from agent_env.plugins._bundles import ENV_ALIASES, installed_aliases
from tst.unit.bundle._support import layout


def _shipped() -> dict:
    return {bundle.name: bundle for bundle in installed_bundles() if bundle.dist == CORE}


SHIPPED = sorted(_shipped())


def test_agentenv_framework_ships_its_bundles():
    assert SHIPPED == ["agents", "computers", "drivers", "echo-mcp", "hello", "mail-mcp", "next-env-event"]


@pytest.mark.parametrize("name", SHIPPED)
def test_a_shipped_bundle_parses_resolves_and_passes_the_checks_a_run_makes(name):
    bundle = _shipped()[name]
    assert bundle.problem is None and bundle.root is not None, bundle.problem

    check_bundle(resolve_bundle(parse_bundle(bundle.root, id_root=bundle.id_root)), installed=installed_bundles)


def test_the_shipped_aliases_are_agent_envs_alone_and_claude_code_names_the_default_agent():
    """Two shipped bundles giving one alias would leave it naming neither."""
    table = installed_aliases(tuple(_shipped().values()))

    assert table.refused == {} and table.refused_claims == {}
    assert {alias: named.target_id for alias, named in table.aliases.items()} == {
        "claude-code": "@local/agentenv-framework/agents/claude-code", "pi": "@local/agentenv-framework/agents/pi",
        "openhands": "@local/agentenv-framework/agents/openhands", "codex": "@local/agentenv-framework/agents/codex",
        "gemini": "@local/agentenv-framework/agents/gemini", "opencode": "@local/agentenv-framework/agents/opencode",
        "letta-code": "@local/agentenv-framework/agents/letta-code"}
    assert table.target("claude-code").target_id == get_config().get_default_a2a_agent_id()



def test_the_shipped_env_aliases_name_the_mail_env_the_validator_deploys_and_the_reference_driver():
    table = installed_aliases(tuple(_shipped().values()), ENV_ALIASES)

    assert table.refused == {} and table.refused_claims == {}
    assert {alias: named.target_id for alias, named in table.aliases.items()} == {
        "mail": "@local/agentenv-framework/mail-mcp/mail", "reference": "@local/agentenv-framework/drivers/reference"}
    assert A2AAgent.VALIDATION_ENV_ID == "mail"

def test_a_task_naming_no_agent_plans_with_the_shipped_default(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    layout(tmp_path / "bare", {"tasks/t.json": json.dumps([{"id": "agent", "type": "deploy_agent", "env_ids": []}])})
    default = get_config().get_default_a2a_agent_id()

    plan = plan_bundle(resolve_bundle(parse_bundle(tmp_path / "bare")), installed=installed_bundles)

    assert (BundleKind.AGENT, default) in [(write.kind, write.id) for write in plan.writes]
    assert plan.origins and {bundle.id_root for bundle in plan.origins.values()} == {_shipped()["agents"].id_root}
