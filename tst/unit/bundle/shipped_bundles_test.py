"""Every bundle agentenv-framework ships parses, resolves and passes the checks a run makes, and a task naming no agent
plans with the shipped default. A run that pulls one entity from an installed bundle reads all of that bundle, so one
broken folder in a shipped bundle would fail every run pulling from it, the default agent's among them."""

from __future__ import annotations

import json

import pytest

from agent_env.bundle import BundleKind, parse_bundle
from agent_env.bundle.installed import CORE, installed_bundles
from agent_env.bundle.plan import check_bundle, plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.config import get_config
from tst.unit.bundle._support import layout


def _shipped() -> dict:
    return {bundle.name: bundle for bundle in installed_bundles() if bundle.dist == CORE}


SHIPPED = sorted(_shipped())


def test_agentenv_framework_ships_its_bundles():
    assert SHIPPED == ["agents", "echo-mcp", "hello"]


@pytest.mark.parametrize("name", SHIPPED)
def test_a_shipped_bundle_parses_resolves_and_passes_the_checks_a_run_makes(name):
    bundle = _shipped()[name]
    assert bundle.problem is None and bundle.root is not None, bundle.problem

    check_bundle(resolve_bundle(parse_bundle(bundle.root, id_root=bundle.id_root)), installed=installed_bundles)


def test_a_task_naming_no_agent_plans_with_the_shipped_default(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    layout(tmp_path / "bare", {"tasks/t.json": json.dumps([{"id": "agent", "type": "deploy_agent", "env_ids": []}])})
    default = get_config().get_default_a2a_agent_id()

    plan = plan_bundle(resolve_bundle(parse_bundle(tmp_path / "bare")), installed=installed_bundles)

    assert (BundleKind.AGENT, default) in [(write.kind, write.id) for write in plan.writes]
    assert plan.origins and {bundle.id_root for bundle in plan.origins.values()} == {_shipped()["agents"].id_root}
