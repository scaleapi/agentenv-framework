"""The infra envs a run builds: the ones missing from the store, or built by another agent-env release, into local
stores only, for this host's platform, once each however many runs ask."""

import importlib

import pytest
from click.testing import CliRunner

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.cli.up import _bootstrap_envs, up
from agent_env.env import GatewayEnv
from agent_env.env import bootstrap
from agent_env.env.bootstrap import GATEWAY, SERVICE_DB, WEBSITE_BROWSER, InfraError, ensure_default_envs


@pytest.fixture
def puts(local_stores, monkeypatch):
    """Stands in for each put: records the kind, id and platform, and writes the env as this release's."""
    calls = []

    def put(kind):
        def write(env_id, *, platform):
            calls.append((kind, env_id, platform))
            artifact = get_artifact_store().put_document(DockerImageArtifact(
                id=f"{kind}-{env_id}", description=kind, image_name=f"{kind}:v1", tar_gz_s3_url="file:///x.tar.gz"))
            GatewayEnv.put(id=env_id, docker_image_artifact=artifact, metadata={"agent_env_version": "0.9.2"})
        return write

    for kind in (GATEWAY, SERVICE_DB, WEBSITE_BROWSER):
        monkeypatch.setitem(bootstrap._PUTS, kind, put(kind))
    monkeypatch.setattr(bootstrap, "_agent_env_version", lambda: "0.9.2")
    return calls


def test_missing_infra_is_built_for_this_host_in_order_and_once(puts):
    said = []

    builds = ensure_default_envs([WEBSITE_BROWSER, GATEWAY, SERVICE_DB], say=said.append)

    assert puts == [(SERVICE_DB, "default-db", None), (GATEWAY, "default", None), (WEBSITE_BROWSER, "website-browser", None)]
    assert [str(build) for build in builds] == [
        "service-db env 'default-db' (missing)", "gateway env 'default' (missing)",
        "website-browser env 'website-browser' (missing)"]
    assert said[0] == "service-db env 'default-db' (missing): building, which can take minutes"
    assert ensure_default_envs([GATEWAY, SERVICE_DB]) == []
    assert len(puts) == 3


def test_infra_another_release_built_is_rebuilt(puts, monkeypatch):
    ensure_default_envs([GATEWAY])
    monkeypatch.setattr(bootstrap, "_agent_env_version", lambda: "0.9.3")

    (build,) = ensure_default_envs([GATEWAY])

    assert build.reason == "built by agent-env 0.9.2, this is 0.9.3"
    assert len(puts) == 2


def test_infra_with_no_recorded_release_is_left_as_it_is(puts):
    artifact = get_artifact_store().put_document(DockerImageArtifact(
        id="gateway-default", description="g", image_name="g:v1", tar_gz_s3_url="file:///g.tar.gz"))
    GatewayEnv.put(id="default", docker_image_artifact=artifact)

    assert ensure_default_envs([GATEWAY]) == []
    assert puts == []


def test_nothing_is_built_without_docker(puts, monkeypatch):
    monkeypatch.setattr(bootstrap, "docker_unreachable", lambda: "docker isn't on PATH")

    with pytest.raises(InfraError) as e:
        ensure_default_envs([GATEWAY, SERVICE_DB])

    assert e.value.problems == ["building the service-db, gateway env needs docker, and docker isn't on PATH"]
    assert puts == []


def test_stores_that_arent_local_are_never_built_into(puts, monkeypatch):
    ensure_default_envs([GATEWAY])
    monkeypatch.setattr(bootstrap, "stores_are_local", lambda: False)
    monkeypatch.setattr(bootstrap, "_agent_env_version", lambda: "0.9.3")

    with pytest.raises(InfraError) as e:
        ensure_default_envs([GATEWAY, SERVICE_DB])

    assert e.value.problems == ["the service-db env 'default-db' isn't in the store, and agent-env builds infra envs "
                                "only into local stores; put it with `agent-env env service-db put --id default-db`"]
    assert ensure_default_envs([GATEWAY]) == []  # another release's gateway, in a store it doesn't build into
    assert len(puts) == 1


def test_the_default_stores_are_local(local_stores):
    assert bootstrap.stores_are_local()


def test_up_builds_whats_missing_and_says_what_was_already_there(puts, capsys):
    ensure_default_envs([GATEWAY])

    _bootstrap_envs()

    out = capsys.readouterr().out
    assert "  service-db env 'default-db' (missing): building, which can take minutes" in out
    assert "  gateway      already registered (default)" in out
    assert [kind for kind, *_ in puts] == [GATEWAY, SERVICE_DB]


def test_up_reports_infra_it_cant_build_as_one_line(puts, monkeypatch):
    monkeypatch.setattr(bootstrap, "docker_unreachable", lambda: "docker isn't on PATH")

    monkeypatch.setattr(importlib.import_module("agent_env.cli.up"), "_require_explorer_deps", lambda: None)
    monkeypatch.setattr("agent_env.config.runtime.Config.config_path", lambda self: "config.toml")
    monkeypatch.setattr("agent_env.config.configure", lambda *a, **k: None)
    result = CliRunner().invoke(up, [])

    assert result.exit_code == 1
    assert result.output.rstrip().endswith(
        "Error: building the service-db, gateway env needs docker, and docker isn't on PATH; or run "
        "`agent-env up --no-bootstrap`")
