"""`agent-env a2a-agent list` and `get`: the agents a task can name, and the agent a name deploys."""

import importlib
import json

import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent import a2a_agent as a2a_agent_module
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.cli import cli
from agent_env.plugins._bundles import ALIASES_FILE, InstalledBundle, installed_aliases

SHIPPED = "@local/agentenv-framework/agents/claude-code"
# The package exports each command under its module's name.
get_module = importlib.import_module("agent_env.cli.a2a_agent.get")
list_module = importlib.import_module("agent_env.cli.a2a_agent.list_agents")


def _bundle(tmp_path, package, name, agents, aliases):
    root = tmp_path / package / name
    for agent in agents:
        (root / "agents" / agent).mkdir(parents=True)
        (root / "agents" / agent / "Dockerfile").write_text("FROM scratch\n")
    (root / ALIASES_FILE).write_text(aliases)
    return InstalledBundle(name, package, "1.0", package.replace("-", "_"), root, None, None)


@pytest.fixture
def installed(tmp_path, monkeypatch, local_stores, cli_routing):
    """agent-env's agents bundle, and two plugin bundles that both give an agent the alias helper."""
    bundles = (
        _bundle(tmp_path, "agentenv-framework", "agents", ["claude-code", "pi"],
                '[agents]\nclaude-code = ["claude-code"]\npi = ["pi"]\n'),
        _bundle(tmp_path, "acme-kit", "kit", ["solver"], '[agents]\nsolver = ["helper", "fixer"]\n'),
        _bundle(tmp_path, "zeta-tools", "tools", ["helper"], '[agents]\nhelper = ["helper"]\n'),
    )
    monkeypatch.setattr(list_module, "installed_bundles", lambda: bundles)
    for module in (a2a_agent_module, get_module):
        monkeypatch.setattr(module, "installed_aliases", lambda: installed_aliases(bundles))
    return bundles


def _put(agent_id):
    image = get_artifact_store().put_document(DockerImageArtifact(
        id=f"{agent_id}__agent_image", description="d", image_name="ghcr.io/acme/agent:1", tar_gz_s3_url="s3://b/x"))
    return A2AAgent.put(id=agent_id, docker_image_artifact=image)


def _lines(*args):
    result = CliRunner().invoke(cli, ["a2a-agent", *args])
    assert result.exit_code == 0, result.output
    return [" ".join(line.split()) for line in result.output.splitlines()]


def test_the_list_shows_each_installed_agent_with_its_aliases_then_the_stores_agents_newest_first(installed):
    for agent_id in ("older", "newer", SHIPPED):  # a bundle run writes the shipped agent it deploys
        _put(agent_id)

    assert _lines("list") == [
        "ID ALIASES FROM",
        f"{SHIPPED} claude-code agentenv-framework 1.0, agents bundle",
        "@local/agentenv-framework/agents/pi pi agentenv-framework 1.0, agents bundle",
        "@local/acme-kit/kit/solver fixer acme-kit 1.0, kit bundle",
        "@local/zeta-tools/tools/helper zeta-tools 1.0, tools bundle",
        "newer the store, v1",
        "older the store, v1",
        "",
        "A task names an agent in a2a_agent_id by its id, or by an alias.",
        "",
        "Refused aliases:",
        "'helper' is an alias acme-kit's kit bundle and zeta-tools's tools bundle each give an agent, so it names none "
        "of them; name the agent by its id",
    ]


def test_get_prints_the_agent_a_name_deploys_and_the_stores_agent_of_an_alias_id_shadows_the_alias(installed):
    _put(SHIPPED)

    aliased = CliRunner().invoke(cli, ["a2a-agent", "get", "--id", "claude-code"])

    assert aliased.exit_code == 0, aliased.output
    assert json.loads(aliased.stdout)["id"] == SHIPPED
    assert aliased.stderr == f"claude-code → {SHIPPED} (the alias agentenv-framework's agents bundle gives it)\n"

    _put("claude-code")
    own = CliRunner().invoke(cli, ["a2a-agent", "get", "--id", "claude-code"])
    listed = json.loads(CliRunner().invoke(cli, ["a2a-agent", "list", "--json"]).output)

    assert (json.loads(own.stdout)["id"], own.stderr) == ("claude-code", "")
    assert [(a["id"], a["shadowed_aliases"]) for a in listed["agents"] if a["shadowed_aliases"]] == [
        (SHIPPED, ["claude-code"])]
    assert f"{SHIPPED} claude-code (shadowed) agentenv-framework 1.0, agents bundle" in _lines("list")
    assert "A shadowed alias names the store's agent of that id instead." in _lines("list")


def test_the_list_takes_a_limit_on_the_store_and_says_how_many_more_it_has(installed):
    for i in range(3):
        _put(f"agent-{i}")

    limited = json.loads(CliRunner().invoke(cli, ["a2a-agent", "list", "--limit", "2", "--json"]).output)
    everything = json.loads(CliRunner().invoke(cli, ["a2a-agent", "list", "--limit", "0", "--json"]).output)

    assert [a["id"] for a in limited["agents"] if a["source"] == "store"] == ["agent-2", "agent-1"]
    assert limited["more_in_store"] == 1 and everything["more_in_store"] == 0
    assert [a["id"] for a in everything["agents"] if a["source"] == "store"] == ["agent-2", "agent-1", "agent-0"]
    assert "The store has 1 more agent: --limit 0 lists them all." in _lines("list", "--limit", "2")


def test_the_list_shows_the_installed_agents_when_the_store_or_a_bundle_cant_be_read(installed, monkeypatch):
    broken = InstalledBundle("broken", "demo-a", "1.0", "demo_a", None, "invalid-plugin", "it's broken")
    monkeypatch.setattr(list_module, "installed_bundles", lambda: (*installed, broken))

    class _Down:
        def get_document_store(self):
            raise ConnectionError("the store is down")

    monkeypatch.setattr(list_module, "get_config", lambda: _Down())

    lines = _lines("list")

    assert f"{SHIPPED} claude-code agentenv-framework 1.0, agents bundle" in lines
    assert "The store can't be read, so only the installed agents are listed: ConnectionError: the store is down" in lines
    assert "1 installed bundle can't be read: agent-env run says why." in lines
