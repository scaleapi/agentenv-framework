"""Every core type ``bundle init`` writes into a folder through its ``to_toml`` reads back with ``from_toml`` as the entity
the store held: on every field its toml carries, and every byte of its files."""

import contextlib
import io
import tarfile

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact import FileArtifact
from agent_env.artifact.artifacts.cli import CliArtifact
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.artifacts.environment_universe import EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.artifacts.skill import SkillArtifact
from agent_env.artifact.artifact import Artifact
from agent_env.bundle import parse_bundle
from agent_env.bundle.importer import import_task
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import build_step, resolve_bundle
from agent_env.config import get_config
from agent_env.entity.scopes import FieldScope
from agent_env.env.env import Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.store.routing import namespace_routing
from agent_env.task import Task
from tst.unit.bundle._support import context_image, layout

LOAD = {"id": "load", "type": "load_artifact", "env_id": "crm"}


@pytest.fixture
def home(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path




def _env(home, id, name=None):
    return MCPServerEnv.put(id=id, docker_image_artifact=context_image(home, f"{id}-image"), environment_name=name or id)


def _context(image):
    """The files an image's stored build context holds, by path, its bundle toml left out."""
    url = image.build_context_object_url
    data = get_config().get_object_store_at(url).get(url)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {member.name: tar.extractfile(member).read() for member in tar if member.isreg()
                and member.name not in ("env.toml", "agent.toml")}


@contextlib.contextmanager
def _read_back(home, steps, kind, name, *, user=False):
    """Import a task of ``steps``, materialize its folder into a namespace of its own, and read back what the folder
    holds as ``name``, of ``kind``, in that namespace."""
    Task.put(id="t", steps=[build_step(dict(step)) for step in steps])
    imported = import_task("t", home / "t", include_scopes=[FieldScope.USER] if user else [])
    bundle = parse_bundle(imported.root)
    with namespace_routing():
        materialization = materialize(plan_bundle(resolve_bundle(bundle)))
        entity_id = f"{bundle.id_root}/{name}"
        getter = {"env": Env.get, "agent": A2AAgent.get, "artifact": Artifact.get}[kind]
        yield getter(entity_id, materialization.version_of(kind, entity_id)), imported


def _bytes(file: FileArtifact) -> bytes:
    return file.load()


def test_an_mcp_server_env_reads_back_with_its_name_provider_and_the_image_its_dockerfile_reads(home):
    stored = _env(home, "crm", "crm-server")

    with _read_back(home, [{"id": "env", "type": "deploy_env", "env_id": "crm"}], "env", "crm") as (read, _):
        assert (type(read), read.environment_name, read.env_provider_type) == (
            MCPServerEnv, stored.environment_name, stored.env_provider_type)
        assert read.docker_image_artifact.dockerfile_path == "Dockerfile"
        written, original = _context(read.docker_image_artifact), _context(stored.docker_image_artifact)
        assert written == {name: data for name, data in original.items() if name != "unread.md"}


def test_a_multi_env_reads_back_naming_its_envs_in_order(home):
    children = [_env(home, "slack"), _env(home, "gmail")]
    MultiEnv.put(id="suite", mcp_server_envs=children, name="suite-gateway")

    with _read_back(home, [{"id": "env", "type": "deploy_env", "env_id": "suite"}], "env", "suite") as (read, imported):
        assert (type(read), read.name, read.env_provider_type) == (MultiEnv, "suite-gateway", "gateway")
        assert [env.environment_name for env in read.mcp_server_envs] == ["slack", "gmail"]
        assert sorted(entity.folder for entity in imported.written) == ["envs/gmail", "envs/slack", "envs/suite"]


def test_a_website_env_reads_back_with_both_images_built_from_its_one_folder(home):
    files = {"Dockerfile.backend": "FROM python:3.12-slim\nCOPY api.py /api.py\n",
             "Dockerfile.frontend": "FROM nginx\nCOPY site/ /usr/share/nginx/html/\n", "api.py": "api\n",
             "site/index.html": "<p>hi</p>\n"}
    backend = context_image(home, "shop-backend", files, "Dockerfile.backend")
    frontend = context_image(home, "shop-frontend", files, "Dockerfile.frontend")
    WebsiteEnv.put(id="shop", backend_docker_image_artifact=backend, frontend_docker_image_artifact=frontend,
                   environment_name="shop")

    with _read_back(home, [{"id": "env", "type": "deploy_env", "env_id": "shop"}], "env", "shop") as (read, _):
        assert (type(read), read.environment_name) == (WebsiteEnv, "shop")
        assert (read.backend_docker_image_artifact.dockerfile_path,
                read.frontend_docker_image_artifact.dockerfile_path) == ("Dockerfile.backend", "Dockerfile.frontend")
        assert set(_context(read.frontend_docker_image_artifact)) >= {"site/index.html", "Dockerfile.frontend"}


def test_an_agent_reads_back_with_its_env_vars_and_the_metadata_its_toml_carries(home):
    A2AAgent.put(id="solver", docker_image_artifact=context_image(home, "solver-image"),
                 default_env_vars={"MODE": "fast", "OPENAI_API_KEY": "sk-not-real"},
                 metadata={"default_model": "claude", "min_disk_size_gb": 20, "takes_credentials": False,
                           "agent_card": {"name": "x"}})
    _env(home, "crm")
    steps = [{"id": "env", "type": "deploy_env", "env_id": "crm"},
             {"id": "agent", "type": "deploy_agent", "env_ids": ["crm"], "a2a_agent_id": "solver"}]

    with _read_back(home, steps, "agent", "solver", user=True) as (read, imported):
        assert read.default_env_vars == {"MODE": "fast"}
        assert read.metadata == {"default_model": "claude", "min_disk_size_gb": 20, "takes_credentials": False}
        whys = {drop.where: drop.why for drop in imported.dropped}
        assert whys["agents/solver: metadata"] == (
            "an agent.toml's [metadata] takes only default_model, min_disk_size_gb and takes_credentials, not agent_card")
        assert whys["agents/solver/agent.toml: default_env_vars.OPENAI_API_KEY"].startswith(
            "dropped: its name looks like a secret")


@pytest.mark.parametrize("description", ["rows.csv", "the crm's rows"])
def test_a_file_reads_back_with_its_bytes_name_and_description(home, description):
    _env(home, "crm")
    (home / "rows.csv").write_text("a,b\n1,2\n")
    stored = FileArtifact.put("rows", description=description, file_path=str(home / "rows.csv"))

    with _read_back(home, [{**LOAD, "artifacts": [{"id": "rows"}]}], "artifact", "rows") as (read, _):
        assert (type(read), read.filename, read.description, _bytes(read)) == (
            FileArtifact, stored.filename, description, b"a,b\n1,2\n")


@pytest.mark.parametrize("files", [{"only.txt": "one\n"}, {"a.txt": "a\n", "deep/er/b.txt": "b\n"}],
                         ids=["one-file", "nested"])
def test_a_file_universe_reads_back_with_each_file_at_its_path(home, files):
    _env(home, "crm")
    paths = {name: path for name in files for path in [layout(home / "fau", files) / name]}
    FileArtifactUniverse.put_bundled("docs", files=paths)

    with _read_back(home, [{**LOAD, "artifacts": [{"id": "docs"}]}], "artifact", "docs") as (read, _):
        assert type(read) is FileArtifactUniverse
        assert {path: _bytes(file).decode() for path, file in read.get_file_artifacts().items()} == files


def test_an_environment_reads_back_wrapping_its_file(home):
    _env(home, "crm")
    (home / "seed.json").write_text('{"rows": 1}')
    file = FileArtifact.put("seed-file", description="the seed", file_path=str(home / "seed.json"))
    EnvironmentArtifact.put("crm-seed", environment_name="crm", file_artifact=file)

    with _read_back(home, [{**LOAD, "artifacts": [{"id": "crm-seed"}]}], "artifact", "crm-seed") as (read, _):
        seeded = read.get_file_artifact()
        assert (type(read), read.environment_name) == (EnvironmentArtifact, "crm")
        assert (seeded.filename, seeded.description, _bytes(seeded)) == ("seed.json", "the seed", b'{"rows": 1}')


def test_an_environment_of_several_files_reads_back_with_each_file_at_its_path(home):
    _env(home, "crm")
    texts = {"rows.json": "rows", "deep/er/notes.txt": "notes"}
    files = {path: FileArtifact.put_bytes(f"crm-{index}", description=path, filename=path.rsplit("/", 1)[-1],
                                          content=text.encode(), content_type="text/plain")
             for index, (path, text) in enumerate(texts.items())}
    EnvironmentArtifact.put("crm-seed", environment_name="crm", file_artifacts=files)

    with _read_back(home, [{**LOAD, "artifacts": [{"id": "crm-seed"}]}], "artifact", "crm-seed") as (read, _):
        assert (type(read), read.environment_name) == (EnvironmentArtifact, "crm")
        assert {path: _bytes(file).decode() for path, file in read.get_file_artifacts().items()} == texts


def test_an_environment_universe_reads_back_with_its_environments_and_metadata(home):
    _env(home, "crm")
    environments = []
    for name, text in (("crm", "crm rows"), ("team/slack", "slack rows")):
        path = layout(home / "u" / name, {"data.json": text}) / "data.json"
        file = FileArtifact.put(f"u-{name.replace('/', '-')}-file", description="data.json", file_path=str(path))
        environments.append(EnvironmentArtifact.put(f"u-{name.replace('/', '-')}", environment_name=name,
                                                    file_artifact=file))
    notes = layout(home / "u" / "meta", {"notes.md": "who's who"}) / "notes.md"
    EnvironmentUniverseArtifact.put("world", environment_artifacts=environments, metadata={
        "people/roster": FileArtifact.put("u-roster", description="notes.md", file_path=str(notes))})
    steps = [{"id": "env", "type": "deploy_env", "env_id": "crm", "artifact_id": "world"}]

    with _read_back(home, steps, "artifact", "world") as (read, _):
        assert type(read) is EnvironmentUniverseArtifact
        assert {env.environment_name: _bytes(env.get_file_artifact()).decode()
                for env in read.get_environment_artifacts()} == {"crm": "crm rows", "team/slack": "slack rows"}
        assert {key: _bytes(file).decode() for key, file in read.get_metadata().items()} == {"people/roster": "who's who"}


def test_a_cli_reads_back_with_the_tree_of_the_version_the_task_names(home):
    old = layout(home / "cli-v1", {"bin/crm": "#!/bin/sh\necho v1\n", "lib/util.sh": "v1\n"})
    new = layout(home / "cli-v2", {"bin/crm": "#!/bin/sh\necho v2\n"})
    CliArtifact.put("crm-cli", command_name="crm", entrypoint="bin/crm", cli_dir=old)
    CliArtifact.put("crm-cli", command_name="crm", entrypoint="bin/crm", cli_dir=new)
    _env(home, "crm")
    steps = [{**LOAD, "agent_name": "default-agent", "artifacts": [{"id": "crm-cli", "version": 1}]}]

    with _read_back(home, steps, "artifact", "crm-cli") as (read, _):
        files = read.get_cli_files().get_file_artifacts()
        assert (type(read), read.command_name, read.entrypoint) == (CliArtifact, "crm", "bin/crm")
        assert {path: _bytes(file).decode() for path, file in files.items()} == {
            "bin/crm": "#!/bin/sh\necho v1\n", "lib/util.sh": "v1\n"}


def test_a_skill_reads_back_in_a_folder_named_for_it(home):
    skill = layout(home / "skill", {"SKILL.md": "---\nname: triage-notes\ndescription: how triage is noted\n---\n"
                                                "Write it down.\n", "examples/one.md": "one\n"})
    SkillArtifact.put("team/triage-notes", skill_dir=skill)
    steps = [{"id": "skills", "type": "add_skills", "agent_name": "default-agent",
              "skills": [{"skill_artifact_id": "team/triage-notes"}]}]

    with _read_back(home, steps, "artifact", "triage-notes") as (read, imported):
        files = read.get_skill_files().get_file_artifacts()
        assert (type(read), read.skill_name, read.description) == (SkillArtifact, "triage-notes", "how triage is noted")
        assert sorted(files) == ["SKILL.md", "examples/one.md"]
        assert [entity.folder for entity in imported.written] == ["skills/triage-notes"]
