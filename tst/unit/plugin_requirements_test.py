"""A plugin whose own requirements exclude the installed agent-env, read from its metadata."""

import pytest

from agent_env.plugins import _requirements
from agent_env.plugins._requirements import incompatibility


class _Dist:
    def __init__(self, *requires: str):
        self.requires = list(requires)


@pytest.fixture(autouse=True)
def _installed(monkeypatch):
    versions = {"agentenv-framework": "0.9.1218", "agentenv-framework-protocol": "0.1.231"}
    monkeypatch.setattr(_requirements, "installed", versions.get)


@pytest.mark.parametrize("requires", [
    [],
    ["agentenv-framework>=0.9.1193"],
    ["agentenv_framework[explorer] >=0.9"],
    ["agentenv-framework>=0.9,<0.10", "agentenv-framework-protocol==0.1.231"],
    ["agentenv-framework>=999; extra == 'dev'"],
    ["agentenv-framework>=999; python_version < '3'"],
    ["requests>=999", "not a requirement !!"],
])
def test_a_requirement_the_installed_agent_env_meets_or_that_does_not_apply_passes(requires):
    assert incompatibility(_Dist(*requires)) is None


@pytest.mark.parametrize(("requires", "why"), [
    (["agentenv-framework>=0.9.1220"], "needs agentenv-framework>=0.9.1220 (installed: 0.9.1218)"),
    (["agentenv-framework<0.9.1218"], "needs agentenv-framework<0.9.1218 (installed: 0.9.1218)"),
    (["Agentenv.Framework==0.9.1200"], "needs agentenv-framework==0.9.1200 (installed: 0.9.1218)"),
    (["agentenv-framework-protocol>=0.2"], "needs agentenv-framework-protocol>=0.2 (installed: 0.1.231)"),
    (["agentenv-framework>=999", "agentenv-framework-protocol>=999"],
     "needs agentenv-framework>=999 (installed: 0.9.1218); agentenv-framework-protocol>=999 (installed: 0.1.231)"),
])
def test_a_requirement_the_installed_agent_env_does_not_meet_names_both_versions(requires, why):
    assert incompatibility(_Dist(*requires)) == why


def test_a_dev_build_does_not_meet_the_floor_of_the_release_it_precedes(monkeypatch):
    monkeypatch.setattr(_requirements, "installed", {"agentenv-framework": "0.9.1219.dev0"}.get)

    assert incompatibility(_Dist("agentenv-framework>=0.9.1219")) is not None
    assert incompatibility(_Dist("agentenv-framework>=0.9.1218")) is None


def test_agent_env_without_installed_metadata_is_not_checked(monkeypatch):
    monkeypatch.setattr(_requirements, "installed", lambda name: None)

    assert incompatibility(_Dist("agentenv-framework>=999")) is None


def test_a_distribution_whose_metadata_cannot_be_read_is_not_checked_here():
    class _Unreadable:
        @property
        def requires(self):
            raise OSError("METADATA vanished")

    assert incompatibility(_Unreadable()) is None
    assert incompatibility(None) is None
