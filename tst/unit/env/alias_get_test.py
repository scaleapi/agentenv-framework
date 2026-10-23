"""Env.get: the store's env of an id, else the env an installed bundle's alias of that id names, under the alias's
id."""

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.env import env as env_module
from agent_env.env.env import Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.plugins._bundles import ALIASES_FILE, InstalledBundle, installed_aliases
from agent_env.store.base import NotFoundError
from agent_env.store.routing import namespace_routing

SHIPPED = "@local/agentenv-framework/mail-mcp/mail"


@pytest.fixture
def bundles(tmp_path, monkeypatch, local_stores):
    """agent-env's mail-mcp bundle gives its env the alias mail-mcp, and two plugin bundles both give helper."""
    installed = []
    for package, name, toml in [("agentenv-framework", "mail-mcp", '[envs]\nmail = ["mail-mcp"]\n'),
                                ("acme-kit", "kit", '[envs]\na = ["helper"]\n'),
                                ("zeta-tools", "tools", '[envs]\nb = ["helper"]\n')]:
        root = tmp_path / package / name
        root.mkdir(parents=True)
        (root / ALIASES_FILE).write_text(toml)
        installed.append(InstalledBundle(name, package, "1.0", package, root, None, None))
    monkeypatch.setattr(env_module, "installed_aliases", lambda table: installed_aliases(installed, table))


@pytest.fixture
def aliases(bundles, cli_routing):
    """``bundles``, with namespace routing on, as in the CLI."""


def _put(env_id):
    image = get_artifact_store().put_document(DockerImageArtifact(
        id=f"{env_id}/env_image", description="d", image_name="ghcr.io/acme/mail:1", tar_gz_s3_url="s3://b/x"))
    return MCPServerEnv.put(id=env_id, docker_image_artifact=image, environment_name="mail")


def test_an_alias_names_the_env_its_bundle_gives_it_under_the_aliass_id(aliases):
    _put(SHIPPED)
    _put(SHIPPED)

    env = Env.get("mail-mcp")

    assert (env.id, env.version, env.environment_name) == ("mail-mcp", 2, "mail")
    assert Env.get("mail-mcp", 1).version == 1  # a deployment's record reloads the version it deployed
    assert Env.get(SHIPPED).id == SHIPPED



def test_an_env_got_by_an_alias_writes_its_metadata_to_the_env_the_alias_names(aliases):
    _put(SHIPPED)
    env = Env.get("mail-mcp")

    env.update_metadata({**env.metadata, "checked": "yes"})
    env.merge_metadata({"merged": "too"})

    assert (env.id, env.stored_id) == ("mail-mcp", SHIPPED)
    assert {key: Env.get(SHIPPED).metadata[key] for key in ("checked", "merged")} == {"checked": "yes", "merged": "too"}

def test_the_stores_env_of_an_aliass_id_is_the_one_it_names(aliases):
    _put(SHIPPED)
    _put("mail-mcp")

    assert Env.get("mail-mcp").docker_image_artifact.id == "mail-mcp/env_image"


def test_a_version_the_stores_env_of_an_alias_id_lacks_is_not_found(aliases):
    _put(SHIPPED)
    _put(SHIPPED)
    _put("mail-mcp")

    with pytest.raises(NotFoundError):
        Env.get("mail-mcp", 2)


def test_an_alias_whose_env_no_run_has_written_says_so(aliases):
    with pytest.raises(NotFoundError, match=f"'mail-mcp' is an alias of {SHIPPED}, which comes with "
                                            "agentenv-framework's mail-mcp bundle and isn't in the store yet"):
        Env.get("mail-mcp")


def test_an_alias_two_bundles_give_is_refused_naming_both(aliases):
    with pytest.raises(NotFoundError, match="'helper' is an alias acme-kit's kit bundle and zeta-tools's tools bundle "
                                            "each give an env"):
        Env.get("helper")


def test_an_id_close_to_an_alias_points_at_it(aliases):
    with pytest.raises(NotFoundError, match="there is no env 'mail_mcp' in the store; did you mean the alias "
                                            "'mail-mcp'"):
        Env.get("mail_mcp")


def test_without_namespace_routing_an_alias_names_no_env_and_says_to_put_one(bundles):
    with namespace_routing():
        _put(SHIPPED)

    with pytest.raises(NotFoundError, match="an alias names its env only in the CLI; put an env of the id "
                                            "'mail-mcp' in the store"):
        Env.get("mail-mcp")
