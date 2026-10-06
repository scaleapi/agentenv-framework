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
            shipped = bootstrap._STOCK_DOCKERFILES[kind].relative_to(bootstrap._ENV_PACKAGE.parent.parent)
            GatewayEnv.put(id=env_id, docker_image_artifact=artifact, metadata={
                "dockerfile_path": f"/another/install/site-packages/{shipped}",
                bootstrap.BUILD_INPUTS_KEY: bootstrap.build_inputs_digest(kind)})
        return write

    for kind in (GATEWAY, SERVICE_DB, WEBSITE_BROWSER):
        monkeypatch.setitem(bootstrap._PUTS, kind, put(kind))
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


def test_infra_built_from_other_inputs_is_rebuilt_once(puts, monkeypatch):
    ensure_default_envs([GATEWAY])
    monkeypatch.setattr(bootstrap, "build_inputs_digest", lambda kind: "this release's")

    (build,) = ensure_default_envs([GATEWAY])

    assert build.reason == "its build inputs changed"
    assert ensure_default_envs([GATEWAY]) == []
    assert len(puts) == 2


def test_infra_built_before_its_inputs_were_recorded_is_rebuilt(puts):
    artifact = get_artifact_store().put_document(DockerImageArtifact(
        id="gateway-default", description="g", image_name="g:v1", tar_gz_s3_url="file:///g.tar.gz"))
    GatewayEnv.put(id="default", docker_image_artifact=artifact,
                   metadata={"agent_env_version": "0.9.1", "dockerfile_path": str(bootstrap.GATEWAY_DOCKERFILE)})

    (build,) = ensure_default_envs([GATEWAY])

    assert build.reason == "built before agent-env recorded its build inputs"
    assert puts == [(GATEWAY, "default", None)]


def test_infra_that_records_no_shipped_dockerfile_is_left_as_it_is(puts):
    artifact = get_artifact_store().put_document(DockerImageArtifact(
        id="gateway-default", description="g", image_name="g:v1", tar_gz_s3_url="file:///g.tar.gz"))
    GatewayEnv.put(id="default", docker_image_artifact=artifact)

    assert ensure_default_envs([GATEWAY]) == []
    assert puts == []


def test_an_env_built_from_someone_elses_dockerfile_is_theirs_to_rebuild(puts, monkeypatch):
    artifact = get_artifact_store().put_document(DockerImageArtifact(
        id="website-browser-website-browser", description="b", image_name="b:v1", tar_gz_s3_url="file:///b.tar.gz"))
    GatewayEnv.put(id="website-browser", docker_image_artifact=artifact,
                   metadata={"dockerfile_path": "/home/me/my-browser/Dockerfile", bootstrap.BUILD_INPUTS_KEY: "theirs"})

    assert ensure_default_envs([WEBSITE_BROWSER]) == []
    assert puts == []


def test_a_run_that_needs_no_infra_reads_no_store(monkeypatch):
    monkeypatch.setattr(bootstrap, "stores_are_local", lambda: pytest.fail("read the configured stores"))

    assert bootstrap.infra_to_build([]) == []


def test_nothing_is_built_without_docker(puts, monkeypatch):
    monkeypatch.setattr(bootstrap, "docker_unreachable", lambda: "docker isn't on PATH")

    with pytest.raises(InfraError) as e:
        ensure_default_envs([GATEWAY, SERVICE_DB])

    assert e.value.problems == ["building the service-db, gateway env needs docker, and docker isn't on PATH"]
    assert puts == []


def test_stores_that_arent_local_are_never_built_into(puts, monkeypatch):
    ensure_default_envs([GATEWAY])
    monkeypatch.setattr(bootstrap, "stores_are_local", lambda: False)
    monkeypatch.setattr(bootstrap, "build_inputs_digest", lambda kind: "this release's")

    with pytest.raises(InfraError) as e:
        ensure_default_envs([GATEWAY, SERVICE_DB])

    assert e.value.problems == ["the service-db env 'default-db' isn't in the store, and agent-env builds infra envs "
                                "only into local stores; put it with `agent-env env service-db put --id default-db`"]
    assert ensure_default_envs([GATEWAY]) == []  # built from other inputs, in a store it doesn't build into
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


def test_the_website_browser_runs_playwrights_chromium_which_is_built_for_amd64_and_arm64():
    """Chrome has no Linux arm64 build, so a browser built for an Apple Silicon host couldn't install it."""
    assert "npx playwright install chromium" in bootstrap.WEBSITE_BROWSER_DOCKERFILE.read_text()
    assert "--browser chromium" in (bootstrap.WEBSITE_BROWSER_CONTEXT / "entrypoint.sh").read_text()


def test_a_put_records_what_the_env_was_built_from_whatever_metadata_it_was_given(local_stores, monkeypatch):
    monkeypatch.setattr(bootstrap, "build_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap.DockerImageArtifact, "put", lambda id, **kwargs: get_artifact_store().put_document(
        DockerImageArtifact(id=id, description="b", image_name=kwargs["image_name"], tar_gz_s3_url="file:///b.tar.gz")))

    env = bootstrap.put_website_browser_env("website-browser", platform=None,
                                            metadata={bootstrap.BUILD_INPUTS_KEY: "mine", "owner": "me"})

    assert env.metadata[bootstrap.BUILD_INPUTS_KEY] == bootstrap.build_inputs_digest(WEBSITE_BROWSER)
    assert env.metadata["owner"] == "me"


def test_the_build_inputs_are_the_shipped_files_and_build_args_less_python_caches(tmp_path, monkeypatch):
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/gateway.cpython-312.pyc").write_text("compiled")
    (tmp_path / "gateway.py").write_text("served")
    before = bootstrap.build_inputs_digest(WEBSITE_BROWSER)
    monkeypatch.setitem(bootstrap._BUILD_ARGS, WEBSITE_BROWSER, {"PLAYWRIGHT_MCP_VERSION": "9.9.9"})

    assert bootstrap._files_under(tmp_path) == [tmp_path / "gateway.py"]
    assert bootstrap.build_inputs_digest(WEBSITE_BROWSER) != before
