"""The infra envs a run sets up: the ones missing from the store, or built by another agent-env release, into local
stores only, once each however many runs ask; pulled from the images published for their sources when every one of a
kind's is, else built for this host's platform. One built here switches to them once they're published."""

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.cli.up import _bootstrap_envs, up
from agent_env.env import GatewayEnv
from agent_env.env import bootstrap
from agent_env.env.bootstrap import GATEWAY, SERVICE_DB, WEBSITE_BROWSER, InfraError, ensure_default_envs


_DIGEST = "sha256:" + "5" * 64


@pytest.fixture
def puts(local_stores, monkeypatch):
    """Stands in for each put: records the kind, id and platform, or that its images were pulled, and writes the env as
    this release's."""
    calls = []

    def put(kind):
        def write(env_id, *, platform, metadata=None, published=None):
            calls.append((kind, env_id, platform) if published is None else (kind, env_id, "pulled"))
            artifact = get_artifact_store().put_document(DockerImageArtifact(
                id=f"{kind}-{env_id}", description=kind, image_name=f"{kind}:v1",
                tar_gz_s3_url=None if published else "file:///x.tar.gz", image_digest=_DIGEST if published else None))
            shipped = bootstrap._DOCKERFILES[kind][0].relative_to(bootstrap._ENV_PACKAGE.parent.parent)
            GatewayEnv.put(id=env_id, docker_image_artifact=artifact, metadata={
                "dockerfile_path": f"/another/install/site-packages/{shipped}",
                bootstrap.BUILD_INPUTS_KEY: bootstrap.build_inputs_digest(kind)})
        return write

    for kind in (GATEWAY, SERVICE_DB, WEBSITE_BROWSER):
        monkeypatch.setitem(bootstrap._PUTS, kind, put(kind))
    return calls


@pytest.mark.parametrize("kind, put", [
    (GATEWAY, bootstrap.put_gateway_env), (SERVICE_DB, bootstrap.put_service_db_env),
    (WEBSITE_BROWSER, bootstrap.put_website_browser_env),
], ids=[GATEWAY, SERVICE_DB, WEBSITE_BROWSER])
def test_each_infra_image_keeps_the_build_context_it_is_built_from(local_stores, monkeypatch, kind, put):
    """So a VM or Modal running another platform than this machine builds it again from that; and those contexts are
    what its recorded build inputs cover. Each Dockerfile is named by its whole path: a bare name is looked up from
    the current folder first."""
    kept = []

    def keep(id, *, description, image_name, build_context_path=None, dockerfile_path=None, platform=None):
        kept.append((Path(build_context_path), dockerfile_path))
        return get_artifact_store().put_document(DockerImageArtifact(
            id=id, description=description, image_name=image_name, tar_gz_s3_url="file:///x.tar.gz"))

    monkeypatch.setattr(bootstrap, "build_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap.DockerImageArtifact, "put", keep)

    put("x", platform=None)

    assert kept == [(dockerfile.parent, str(dockerfile)) for dockerfile in bootstrap._DOCKERFILES[kind]]


@pytest.fixture
def published(monkeypatch):
    """The kinds whose images the registry serves, all under ``_DIGEST``: those added to the set returned."""
    kinds: set[str] = set()

    def served(ref):
        refs = {bootstrap._PUBLISHED[dockerfile].ref() for kind in kinds for dockerfile in bootstrap._DOCKERFILES[kind]}
        return _DIGEST if ref in refs else None

    monkeypatch.setattr("agent_env.env.published_images._registry_digest", served)
    return kinds


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


def test_an_env_built_from_someone_elses_dockerfile_is_theirs_to_rebuild(puts, published):
    artifact = get_artifact_store().put_document(DockerImageArtifact(
        id="website-browser__env_image", description="b", image_name="b:v1", tar_gz_s3_url="file:///b.tar.gz"))
    GatewayEnv.put(id="website-browser", docker_image_artifact=artifact,
                   metadata={"dockerfile_path": "/home/me/my-browser/Dockerfile", bootstrap.BUILD_INPUTS_KEY: "theirs"})
    published.add(WEBSITE_BROWSER)

    assert ensure_default_envs([WEBSITE_BROWSER]) == []
    assert puts == []


def test_a_run_that_needs_no_infra_reads_no_store(monkeypatch):
    monkeypatch.setattr(bootstrap, "stores_are_local", lambda: pytest.fail("read the configured stores"))

    assert bootstrap.infra_to_set_up([]) == []


def test_nothing_is_built_without_docker(puts, monkeypatch):
    monkeypatch.setattr(bootstrap, "docker_unreachable", lambda: "docker isn't on PATH")

    with pytest.raises(InfraError) as e:
        ensure_default_envs([GATEWAY, SERVICE_DB])

    assert e.value.problems == ["building the service-db, gateway env needs docker, and docker isn't on PATH"]
    assert puts == []


def test_stores_that_arent_local_are_never_built_into(puts, published, monkeypatch):
    ensure_default_envs([GATEWAY])
    monkeypatch.setattr(bootstrap, "stores_are_local", lambda: False)
    monkeypatch.setattr(bootstrap, "build_inputs_digest", lambda kind: "this release's")
    published.update({GATEWAY, SERVICE_DB})

    with pytest.raises(InfraError) as e:
        ensure_default_envs([GATEWAY, SERVICE_DB])

    assert e.value.problems == ["the service-db env 'default-db' isn't in the store, and agent-env builds infra envs "
                                "only into local stores; put it with `agent-env env service-db put --id default-db`"]
    assert ensure_default_envs([GATEWAY]) == []  # built from other inputs, in a store it doesn't set up infra in
    assert len(puts) == 1


def test_missing_infra_whose_images_are_all_published_is_registered_from_them_with_no_docker(puts, published,
                                                                                            monkeypatch):
    published.update({GATEWAY, SERVICE_DB})
    monkeypatch.setattr(bootstrap, "docker_unreachable", lambda: "docker isn't on PATH")
    said = []

    builds = ensure_default_envs([GATEWAY, SERVICE_DB], say=said.append)

    assert puts == [(SERVICE_DB, "default-db", "pulled"), (GATEWAY, "default", "pulled")]
    assert [build.action for build in builds] == [bootstrap.Setup.PULL, bootstrap.Setup.PULL]
    assert said == ["service-db env 'default-db' (missing): registering its published images, which each deploy pulls",
                    "gateway env 'default' (missing): registering its published image, which each deploy pulls"]
    assert ensure_default_envs([GATEWAY, SERVICE_DB]) == []


def test_a_kinds_images_are_pulled_only_when_every_one_is_published(puts, monkeypatch):
    db_web = bootstrap._PUBLISHED[bootstrap.DB_WEB_DOCKERFILE].ref()
    monkeypatch.setattr("agent_env.env.published_images._registry_digest",
                        lambda ref: _DIGEST if ref == db_web else None)

    (build,) = ensure_default_envs([SERVICE_DB])

    assert build.builds and puts == [(SERVICE_DB, "default-db", None)]


def test_infra_built_here_is_kept_until_its_images_are_published_then_switched_to_them_once(puts, published):
    ensure_default_envs([GATEWAY])
    assert ensure_default_envs([GATEWAY]) == []
    published.add(GATEWAY)
    said = []

    (build,) = ensure_default_envs([GATEWAY], say=said.append)

    assert (build.action, build.reason) == (bootstrap.Setup.SWITCH, "built on this machine")
    assert said == ["gateway env 'default' (built on this machine): switching to its published image, which each "
                    "deploy pulls"]
    assert ensure_default_envs([GATEWAY]) == []
    assert puts == [(GATEWAY, "default", None), (GATEWAY, "default", "pulled")]


def test_infra_built_from_other_inputs_is_pulled_when_its_images_are_published(puts, published, monkeypatch):
    ensure_default_envs([GATEWAY])
    monkeypatch.setattr(bootstrap, "build_inputs_digest", lambda kind: "this release's")
    published.add(GATEWAY)

    (build,) = ensure_default_envs([GATEWAY])

    assert (build.action, build.reason) == (bootstrap.Setup.PULL, "its build inputs changed")
    assert puts == [(GATEWAY, "default", None), (GATEWAY, "default", "pulled")]


def test_only_the_infra_a_run_builds_needs_docker(puts, published, monkeypatch):
    published.add(GATEWAY)
    monkeypatch.setattr(bootstrap, "docker_unreachable", lambda: "docker isn't on PATH")

    with pytest.raises(InfraError) as e:
        ensure_default_envs([GATEWAY, SERVICE_DB])

    assert e.value.problems == ["building the service-db env needs docker, and docker isn't on PATH"]
    assert puts == []


def test_an_env_set_up_again_keeps_its_own_metadata_and_records_how_it_was_set_up_afresh(local_stores, published):
    artifact = get_artifact_store().put_document(DockerImageArtifact(
        id="gateway-default", description="g", image_name="env-gateway:v1", tar_gz_s3_url="file:///g.tar.gz"))
    GatewayEnv.put(id="default", docker_image_artifact=artifact, metadata={
        "owner": "me", "git_commit": "0ld", "agent_env_version": "0.0.1",
        "dockerfile_path": str(bootstrap.GATEWAY_DOCKERFILE),
        bootstrap.BUILD_INPUTS_KEY: bootstrap.build_inputs_digest(GATEWAY)})
    published.add(GATEWAY)

    (build,) = ensure_default_envs([GATEWAY])

    metadata = GatewayEnv.get("default").metadata
    assert build.action is bootstrap.Setup.SWITCH and metadata["owner"] == "me"
    assert metadata.get("git_commit") != "0ld" and metadata.get("agent_env_version") != "0.0.1"
    assert metadata[bootstrap.BUILD_INPUTS_KEY] == bootstrap.build_inputs_digest(GATEWAY)


def test_a_published_image_is_registered_by_its_tag_and_digest_with_the_context_it_is_built_from(local_stores):
    env = bootstrap.put_gateway_env("default", platform=None, published={bootstrap.GATEWAY_DOCKERFILE: _DIGEST})

    image, published = env.docker_image_artifact, bootstrap._PUBLISHED[bootstrap.GATEWAY_DOCKERFILE]
    assert (image.image_name, image.image_digest, image.tar_gz_object_url) == (published.ref(), _DIGEST, None)
    assert image.pinned_name == f"{published.ref()}@{_DIGEST}"
    assert (image.dockerfile_path, image.source_digest) == ("Dockerfile", f"sha256:{published.tag().removeprefix('src-')}")
    assert get_artifact_store().get_object(image.build_context_object_url)
    assert env.metadata[bootstrap.BUILD_INPUTS_KEY] == bootstrap.build_inputs_digest(GATEWAY)
    assert bootstrap._built_from_stock(env, GATEWAY)


def test_a_published_image_is_registered_from_its_own_dockerfile_whatever_folder_the_run_is_in(local_stores, tmp_path,
                                                                                               monkeypatch):
    """A run started from a folder holding a Dockerfile of its own registers the gateway's, with its sources."""
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    monkeypatch.chdir(tmp_path)

    image = bootstrap.put_gateway_env("default", platform=None,
                                      published={bootstrap.GATEWAY_DOCKERFILE: _DIGEST}).docker_image_artifact

    tag = bootstrap._PUBLISHED[bootstrap.GATEWAY_DOCKERFILE].tag()
    assert (image.dockerfile_path, image.source_digest) == ("Dockerfile", f"sha256:{tag.removeprefix('src-')}")


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


def test_the_build_inputs_are_every_file_in_the_build_context_less_python_caches(tmp_path, monkeypatch):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12\nCOPY . /app\n")
    (tmp_path / "gateway.py").write_text("served")
    monkeypatch.setitem(bootstrap._DOCKERFILES, GATEWAY, [tmp_path / "Dockerfile"])
    before = bootstrap.build_inputs_digest(GATEWAY)

    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/gateway.cpython-312.pyc").write_text("compiled")
    assert bootstrap.build_inputs_digest(GATEWAY) == before
    (tmp_path / "gateway.py").write_text("served, changed")
    assert bootstrap.build_inputs_digest(GATEWAY) != before


# What Docker's legacy builder refuses, by what it is.
_BUILDKIT_ONLY = {
    "a syntax directive": r"^\s*#\s*syntax\s*=",
    "RUN --mount, --network or --security": r"^\s*RUN\s+(--\S+\s+)*--(mount|network|security)\b",
    "COPY or ADD --chmod, --link, --parents, --exclude, --checksum or --keep-git-dir":
        r"^\s*(COPY|ADD)\s+(--\S+\s+)*--(chmod|link|parents|exclude|checksum|keep-git-dir)\b",
    "a heredoc": r"<<-?\s*[\"']?\w+",
    "a build arg with no default": r"^\s*ARG\s+\w+\s*$",
}


def _legacy_build_problems(dockerfile: str) -> list[str]:
    return [what for what, pattern in _BUILDKIT_ONLY.items()
            if re.search(pattern, dockerfile, re.IGNORECASE | re.MULTILINE)]


@pytest.mark.parametrize("dockerfile", [dockerfile for dockerfiles in bootstrap._DOCKERFILES.values()
                                        for dockerfile in dockerfiles],
                         ids=lambda dockerfile: dockerfile.relative_to(bootstrap._ENV_PACKAGE).as_posix())
def test_each_infra_image_builds_from_its_context_alone_on_dockers_legacy_builder(dockerfile):
    """A sandbox rebuilds an infra image from its build context, which carries no build args, and a VM with no buildx,
    as modal_vm's has none, builds with Docker's legacy builder."""
    assert _legacy_build_problems(dockerfile.read_text()) == []


@pytest.mark.parametrize("line, problem", [
    ("# syntax=docker/dockerfile:1", "a syntax directive"),
    ("RUN --mount=type=cache,target=/root/.npm npm ci", "RUN --mount, --network or --security"),
    ("copy --chown=1:1 --chmod=755 entrypoint.sh /app/",
     "COPY or ADD --chmod, --link, --parents, --exclude, --checksum or --keep-git-dir"),
    ("RUN <<EOF\necho hi\nEOF", "a heredoc"),
    ("ARG PLAYWRIGHT_MCP_VERSION", "a build arg with no default"),
], ids=["syntax", "run-mount", "copy-chmod", "heredoc", "arg"])
def test_the_legacy_builder_check_names_what_only_buildkit_or_a_build_arg_builds(line, problem):
    dockerfile = f"FROM python:3.12\n{line}\nCOPY --from=builder --chown=app:app /app /app\n"

    assert _legacy_build_problems(dockerfile) == [problem]
