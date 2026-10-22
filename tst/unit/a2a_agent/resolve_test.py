"""An agent looked up as a step deploys it: the store's of that id, else the agent an installed bundle's alias names."""

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent import a2a_agent as a2a_agent_module
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.plugins._bundles import ALIASES_FILE, InstalledBundle, installed_aliases
from agent_env.store.base import NotFoundError

SHIPPED = "@local/agentenv-framework/agents/claude-code"


@pytest.fixture
def aliases(tmp_path, monkeypatch, local_stores, cli_routing):
    """agent-env's agents bundle gives Claude Code the alias claude-code, and two plugin bundles both give helper."""
    bundles = []
    for package, name, toml in [("agentenv-framework", "agents", '[agents]\nclaude-code = ["claude-code"]\n'),
                                ("acme-kit", "kit", '[agents]\na = ["helper"]\n'),
                                ("zeta-tools", "tools", '[agents]\nb = ["helper"]\n')]:
        root = tmp_path / package / name
        root.mkdir(parents=True)
        (root / ALIASES_FILE).write_text(toml)
        bundles.append(InstalledBundle(name, package, "1.0", package, root, None, None))
    monkeypatch.setattr(a2a_agent_module, "installed_aliases", lambda: installed_aliases(bundles))


def _put(agent_id):
    image = get_artifact_store().put_document(DockerImageArtifact(
        id=f"{agent_id}__agent_image", description="d", image_name="ghcr.io/acme/agent:1", tar_gz_s3_url="s3://b/x"))
    return A2AAgent.put(id=agent_id, docker_image_artifact=image)


def test_an_alias_names_the_agent_its_bundle_gives_it(aliases):
    _put(SHIPPED)

    assert A2AAgent.resolve("claude-code").id == SHIPPED


def test_the_stores_agent_of_an_aliass_id_is_the_one_it_names(aliases):
    _put(SHIPPED)
    _put("claude-code")

    assert A2AAgent.resolve("claude-code").id == "claude-code"


def test_an_id_that_is_no_alias_is_not_found_as_before(aliases):
    with pytest.raises(NotFoundError) as e:
        A2AAgent.resolve("nobody")

    assert "nobody" in str(e.value)


def test_an_alias_takes_no_version(aliases):
    _put(SHIPPED)

    with pytest.raises(ValueError, match=f"'claude-code' is an alias of {SHIPPED}, which comes with agentenv-framework's "
                                         "agents bundle, so it has no fixed version; drop the pin"):
        A2AAgent.resolve("claude-code", 1)


def test_an_alias_whose_agent_no_run_has_written_says_so(aliases):
    with pytest.raises(NotFoundError, match=f"'claude-code' is an alias of {SHIPPED}, which comes with "
                                            "agentenv-framework's agents bundle and isn't in the store yet"):
        A2AAgent.resolve("claude-code")


def test_an_alias_two_bundles_give_is_refused_naming_both(aliases):
    with pytest.raises(NotFoundError, match="'helper' is an alias acme-kit's kit bundle and zeta-tools's tools bundle "
                                            "each give an agent"):
        A2AAgent.resolve("helper")


def test_a_version_the_stores_agent_of_an_alias_id_lacks_is_not_found(aliases):
    _put(SHIPPED)
    _put("claude-code")

    assert A2AAgent.resolve("claude-code", 1).id == "claude-code"
    with pytest.raises(NotFoundError):
        A2AAgent.resolve("claude-code", 2)


def test_the_stores_agent_of_a_refused_alias_id_is_the_one_it_names(aliases):
    _put("helper")

    assert A2AAgent.resolve("helper").id == "helper"


def test_a_name_close_to_an_alias_is_pointed_at_it_unless_it_takes_a_version(aliases):
    with pytest.raises(NotFoundError, match="there is no agent 'claude_code' in the store; did you mean the alias "
                                            "'claude-code'"):
        A2AAgent.resolve("claude_code")
    with pytest.raises(NotFoundError) as pinned:
        A2AAgent.resolve("claude_code", 2)

    assert "did you mean" not in str(pinned.value)
