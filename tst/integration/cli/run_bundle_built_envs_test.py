"""``agent-env run`` on bundles whose envs are written from their folders, with real docker builds and the envs
deployed on the local sandbox. An MCP server on the server provider is named by the environment card in its source,
reused while its folder is unchanged, and rebuilt when it changes. Through the gateway, each way of writing an env
deploys and answers a verifier calling it: an MCP server built from its folder, one built from a Dockerfile elsewhere
under a name of its own, one over a store image, a website, and a multi of bundle and store envs, which is rewritten
when a child is. Environments and a universe the bundle writes load into an MCP server deployed through the gateway,
and an edit to their files loads anew. Needs a Docker daemon and the local registry."""

import hashlib
import json
import logging
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.artifact import EnvironmentUniverseArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle import parse_bundle
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.env import Env
from agent_env.store.routing import namespace_routing

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "tst" / "data"
ITEMS = DATA / "agentenv_mcp"  # an in-memory MCP server whose card names it 'items'
PROTOCOL = REPO / "packages" / "agentenv-protocol" / "src" / "agentenv_protocol"
IMAGE = "envs/items (Dockerfile image)"

# Checks what an env serves through its gateway, by calling it: each items server adds an item under its own
# environment_name's tool and lists it back from the build it was given, the slack server lists its channels, and each
# website serves its frontend and its backend's health. CONFIG, naming what to check, goes above it.
VERIFY = '''
import json

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def verify(mcp_url):
    results = []

    def check(id, passed, description):
        results.append({"id": id, "result": bool(passed), "description": description})

    if CONFIG["items"] or CONFIG["slack"]:
        async with streamablehttp_client(mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = [tool.name for tool in (await session.list_tools()).tools]

                async def call(suffix, arguments):
                    name = next((name for name in names if name.endswith(suffix)), None)
                    if name is None:
                        return None
                    result = await session.call_tool(name, arguments)
                    return None if result.isError else " ".join(getattr(part, "text", "") for part in result.content)

                for name in CONFIG["items"]:
                    added = await call(f"{name}_add_item", {"item": f"from-{name}"})
                    listed = json.loads(await call("list_items", {}) or "{}")
                    check(f"{name}-items", added and f"from-{name}" in listed.get("items", []),
                          f"{name}_add_item adds an item list_items lists, among {names}")
                    check(f"{name}-build", listed.get("build") == CONFIG["build"],
                          f"list_items answers from build {CONFIG['build']}: {listed}")
                if CONFIG["slack"]:
                    channels = await call("channels_list", {"channel_types": "public_channel"})
                    check("slack", channels and json.loads(channels).get("ok"), f"channels_list answers: {channels}")
    gateway = mcp_url.rsplit("/mcp", 1)[0]
    async with httpx.AsyncClient(timeout=30) as client:
        for name in CONFIG["websites"]:
            page = await client.get(f"{gateway}/website/{name}/")
            check(f"{name}-frontend", page.status_code == 200 and "Slack Workspace" in page.text,
                  f"the frontend serves its page: {page.status_code}")
            health = await client.get(f"{gateway}/svc/{name}/api/health")
            check(f"{name}-backend", health.status_code == 200, f"the backend is healthy: {health.status_code}")
    return results
'''


@pytest.fixture
def state(monkeypatch, tmp_path):
    """Local stores under this test's folder. Whatever ran, no sandbox work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    # HOME stays: docker's credential helper (the macOS keychain, for one) can hang a build's base-image lookup
    # when HOME moves.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure()
    reset_artifact_store()
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        reset_artifact_store()
        reset_config()


def _bundle(root):
    """The items server as an env folder, deployed by the server provider, which runs it with no gateway."""
    folder = root / "envs" / "items"
    shutil.copytree(PROTOCOL, folder / "agentenv_protocol", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("server.py", "Dockerfile", "seed.json"):
        shutil.copy(ITEMS / name, folder / name)
    (folder / "env.toml").write_text('env_provider_type = "server"\n')
    (root / "tasks").mkdir()
    (root / "tasks/items.json").write_text(json.dumps(
        [{"id": "env", "type": "deploy_env", "env_id": "items", "sandbox_type": "local"}]))
    return root


def _run(root, *args):
    return CliRunner().invoke(cli, ["run", str(root), *args])


def test_an_env_folder_is_built_named_by_its_card_deployed_and_rebuilt_only_when_it_changes(state):
    root = _bundle(state / "envs-bundle")

    first = _run(root)

    assert first.exit_code == 0, first.output
    assert f"{IMAGE}: building with docker" in first.output, first.output
    assert f"{IMAGE}: v1 (new)" in first.output and "envs/items: v1 (new)" in first.output, first.output
    assert "tasks/items.json v1: unscored" in first.output, first.output
    with namespace_routing():
        env = Env.get(next(entry.id for entry in parse_bundle(root).entries if entry.name == "items"))
    assert (env.environment_name, env.env_provider_type) == ("items", "server")

    again = _run(root)

    assert again.exit_code == 0, again.output
    assert "building with docker" not in again.output, again.output
    assert f"{IMAGE}: v1, unchanged" in again.output and "envs/items: v1, unchanged" in again.output, again.output

    (root / "envs/items/seed.json").write_text('{"items": ["edited"]}\n')
    edited = _run(root)

    assert edited.exit_code == 0, edited.output
    assert f"{IMAGE}: v2 (files changed: seed.json)" in edited.output, edited.output
    assert "envs/items: v2 (" in edited.output and "tasks/items.json v1: unscored" in edited.output, edited.output
    assert edited.output.count("building with docker") == 1, edited.output


def _server(build):
    """The items server's source, its list_items answering with ``build``."""
    source = (ITEMS / "server.py").read_text()
    server = source.replace('json.dumps({"items": self.store})',
                            f'json.dumps({{"items": self.store, "build": "{build}"}})')
    assert server != source
    return server


def _items(folder, build, dockerfile="Dockerfile"):
    """The items server of ``build``, built from ``dockerfile`` in its folder."""
    shutil.copytree(PROTOCOL, folder / "agentenv_protocol", ignore=shutil.ignore_patterns("__pycache__"))
    (folder / "server.py").write_text(_server(build))
    shutil.copy(ITEMS / "seed.json", folder / "seed.json")
    (folder / dockerfile).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ITEMS / "Dockerfile", folder / dockerfile)


def _shop(folder):
    """The slack website as an env folder: its backend and frontend built from Dockerfile.backend and
    Dockerfile.frontend, with the folder as their context."""
    no_dockerfiles = shutil.ignore_patterns("Dockerfile", "__pycache__")
    for part in ("backend", "frontend"):
        shutil.copytree(DATA / "slack_website" / part, folder / part, ignore=no_dockerfiles)
        dockerfile = (DATA / "slack_website" / part / "Dockerfile").read_text()
        (folder / f"Dockerfile.{part}").write_text(dockerfile.replace("slack_website/", ""))
    for package in ("base_mcp", "slack_mcp"):
        shutil.copytree(DATA / package, folder / package, ignore=no_dockerfiles)
    (folder / "env.toml").write_text('type = "website"\nenvironment_name = "shop"\n')


def _check(root, env, *, items=(), slack=False, websites=(), build="v1"):
    """A task deploying ``env`` and verifying what it serves, its check a file artifact of the bundle."""
    config = {"items": list(items), "slack": slack, "websites": list(websites), "build": build}
    (root / f"artifacts/check-{env}").mkdir(parents=True, exist_ok=True)
    (root / f"artifacts/check-{env}/verify.py").write_text(f"CONFIG = {config!r}\n{VERIFY}")
    (root / "tasks").mkdir(exist_ok=True)
    (root / f"tasks/{env}.json").write_text(json.dumps([
        {"id": "env", "type": "deploy_env", "env_id": env, "sandbox_type": "local"},
        {"id": "check", "type": "env_outcome_verifier", "env_id": env, "file_artifact_id": f"check-{env}",
         "verifier_id": env, "depends_on": ["env"]},
    ]))


def _gateway_bundle(root):
    """An env of each kind a bundle writes, all deployed through the gateway: items and stock are the items server,
    built from its folder and named by its card, and built from docker/Dockerfile under a name of its own; relay is an
    MCP server over the store image the CLI put for mail; shop is a website; suite is a multi of items, mail and
    shop."""
    envs = root / "envs"
    _items(envs / "items", "v1")
    _items(envs / "stock", "v1", dockerfile="docker/Dockerfile")
    (envs / "stock/env.toml").write_text('image = { dockerfile = "docker/Dockerfile" }\nenvironment_name = "stock"\n')
    (envs / "relay").mkdir()
    (envs / "relay/env.toml").write_text('image = "mail__env_image"\nenvironment_name = "relay"\n')
    _shop(envs / "shop")
    (envs / "suite").mkdir()
    (envs / "suite/env.toml").write_text(
        'type = "multi"\nmcp_server_envs = ["items", "mail"]\nwebsite_envs = ["shop"]\nname = "suite"\n')
    _check(root, "items", items=["items"])
    _check(root, "stock", items=["stock"])
    _check(root, "relay", slack=True)
    _check(root, "shop", websites=["shop"])
    _check(root, "suite", items=["items"], slack=True, websites=["shop"])
    return root


def _put_mail():
    """The slack server as a store env, mail, put by the CLI as a user would before a bundle names it."""
    put = CliRunner().invoke(cli, ["env", "mcp-server", "put", "--id", "mail", "--environment-name", "mail",
                                   "--dockerfile", str(DATA / "slack_mcp/Dockerfile"), "--context", str(DATA),
                                   "--platform", ""])
    assert put.exit_code == 0, put.output


GATEWAY_IMAGES = ("envs/items (Dockerfile image)", "envs/stock (docker/Dockerfile image)",
                  "envs/shop (Dockerfile.backend image)", "envs/shop (Dockerfile.frontend image)")
GATEWAY_ENVS = ("items", "stock", "relay", "shop", "suite")


def test_each_kind_of_bundle_env_deploys_through_the_gateway_and_a_child_edit_rewrites_its_multi(state):
    _put_mail()
    root = _gateway_bundle(state / "gateway-bundle")

    first = _run(root)

    assert first.exit_code == 0, first.output
    for image in GATEWAY_IMAGES:
        assert f"{image}: building with docker" in first.output and f"{image}: v1 (new)" in first.output, first.output
    for env in GATEWAY_ENVS:
        assert f"envs/{env}: v1 (new)" in first.output, first.output
        assert f"tasks/{env}.json v1: passed (check: 1)" in first.output, first.output
    with namespace_routing():
        ids = {entry.name: entry.id for entry in parse_bundle(root).entries}
        written = {name: Env.get(ids[name]) for name in GATEWAY_ENVS}
        mail = Env.get("mail")
    assert {name: env.environment_name for name, env in written.items() if name != "suite"} == {
        "items": "items", "stock": "stock", "relay": "relay", "shop": "shop"}
    assert written["relay"].docker_image_artifact.id == mail.docker_image_artifact.id == "mail__env_image"
    assert [child.id for child in written["suite"].mcp_server_envs] == [ids["items"], "mail"]
    assert [child.id for child in written["suite"].website_envs] == [ids["shop"]]
    assert written["suite"].name == "suite"

    # The checks now expect build v2 from items, alone and inside suite.
    (root / "envs/items/server.py").write_text(_server("v2"))
    _check(root, "items", items=["items"], build="v2")
    _check(root, "suite", items=["items"], slack=True, websites=["shop"], build="v2")
    edited = _run(root, "--task", "items", "--task", "suite")

    assert edited.exit_code == 0, edited.output
    assert f"{IMAGE}: v2 (files changed: server.py)" in edited.output, edited.output
    assert edited.output.count("building with docker") == 1, edited.output
    assert "envs/items: v2 (" in edited.output and "envs/suite: v2 (" in edited.output, edited.output
    assert "envs/shop: v1, unchanged" in edited.output, edited.output
    for env in ("items", "suite"):
        assert f"artifacts/check-{env}: v2 (files changed: verify.py)" in edited.output, edited.output
        assert f"tasks/{env}.json v1: passed (check: 1)" in edited.output, edited.output

    planned = _run(root, "--dry-run")

    assert planned.exit_code == 0, planned.output
    versions = {"items": 2, "suite": 2}
    for image in GATEWAY_IMAGES:
        assert f"{image}: v{2 if image == IMAGE else 1}, unchanged" in planned.output, planned.output
    for env in GATEWAY_ENVS:
        assert f"envs/{env}: v{versions.get(env, 1)}, unchanged" in planned.output, planned.output


# Checks the items server lists exactly the items CONFIG names, the ones the task loaded. CONFIG goes above it.
VERIFY_LOADED = '''
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def verify(mcp_url):
    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            name = next(tool.name for tool in (await session.list_tools()).tools if tool.name.endswith("list_items"))
            result = await session.call_tool(name, {})
            listed = json.loads(" ".join(getattr(part, "text", "") for part in result.content))
    return [{"id": "loaded", "result": listed.get("items") == CONFIG["items"],
             "description": f"list_items lists what the task loaded, {CONFIG['items']}: {listed}"}]
'''


def _seed(items):
    return json.dumps({"items": items}) + "\n"


def _write(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)


def _load_task(root, name, artifact, items):
    """A task deploying the items server, loading ``artifact`` into it and checking it lists ``items``."""
    (root / f"artifacts/check-{name}").mkdir(parents=True, exist_ok=True)
    (root / f"artifacts/check-{name}/verify.py").write_text(f"CONFIG = {{'items': {items!r}}}\n{VERIFY_LOADED}")
    (root / "tasks").mkdir(exist_ok=True)
    (root / f"tasks/{name}.json").write_text(json.dumps([
        {"id": "env", "type": "deploy_env", "env_id": "items", "sandbox_type": "local"},
        {"id": "load", "type": "load_artifact", "env_id": "items", "artifact_id": artifact, "depends_on": ["env"]},
        {"id": "check", "type": "env_outcome_verifier", "env_id": "items", "file_artifact_id": f"check-{name}",
         "verifier_id": name, "depends_on": ["load"]},
    ]))


def _data_bundle(root):
    """The items server, and data for it in each shape a bundle writes: an environment over its folder's file
    (seed), one naming a file artifact (by-ref over items-file), and a universe laid out as `environment-universe get
    --output-dir` writes one, with the items server's data in its items/ folder, a metadata file, and an environment
    for another server (mail-data), which the items server skips."""
    _items(root / "envs/items", "v1")
    _write(root / "artifacts", {
        "seed/artifact.toml": 'type = "environment"\nenvironment_name = "items"\n',
        "seed/items.json": _seed(["folder"]),
        "items-file/items.json": _seed(["by-ref"]),
        "by-ref/artifact.toml": 'type = "environment"\nenvironment_name = "items"\nfile = "items-file"\n',
        "mail-data/artifact.toml": 'type = "environment"\nenvironment_name = "mail"\n', "mail-data/mail.json": "{}\n",
        "world/artifact.toml": 'type = "environment_universe"\nenvironment_artifacts = ["mail-data"]\n',
        "world/items/items.json": _seed(["universe"]), "world/metadata/manifest/manifest.json": '{"m": 1}\n',
    })
    _load_task(root, "folder", "seed", ["folder"])
    _load_task(root, "ref", "by-ref", ["by-ref"])
    _load_task(root, "universe", "world", ["universe"])
    return root


WRITTEN = ("seed", "by-ref", "items-file", "world")  # the artifacts the data bundle writes


def test_environments_and_a_universe_written_from_a_bundle_load_into_an_env_through_the_gateway(state):
    root = _data_bundle(state / "data-bundle")

    first = _run(root)

    assert first.exit_code == 0, first.output
    for name in WRITTEN:
        assert f"artifacts/{name}: v1 (new)" in first.output, first.output
    for task in ("folder", "ref", "universe"):
        assert f"tasks/{task}.json v1: passed (check: 1)" in first.output, first.output
    with namespace_routing():
        ids = {entry.name: entry.id for entry in parse_bundle(root).entries}
        world = EnvironmentUniverseArtifact.get(ids["world"])
        assert [env.environment_name for env in world.get_environment_artifacts()] == ["items", "mail"]
        assert list(world.get_metadata()) == ["manifest"]

    (root / "artifacts/seed/items.json").write_text(_seed(["folder", "edited"]))
    (root / "artifacts/world/items/items.json").write_text(_seed(["universe", "edited"]))
    _load_task(root, "folder", "seed", ["folder", "edited"])
    _load_task(root, "universe", "world", ["universe", "edited"])
    edited = _run(root, "--task", "folder", "--task", "universe")

    assert edited.exit_code == 0, edited.output
    assert "artifacts/seed: v2 (files changed: items.json)" in edited.output, edited.output
    assert "artifacts/world: v2 (files changed: items/items.json)" in edited.output, edited.output
    assert "artifacts/mail-data: v1, unchanged" in edited.output and "envs/items: v1, unchanged" in edited.output
    for task in ("folder", "universe"):
        assert f"tasks/{task}.json v1: passed (check: 1)" in edited.output, edited.output


def _files(root):
    """A digest of every file under ``root`` but the locks a run takes."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and "locks" not in path.parts:
            digest.update(str(path.relative_to(root)).encode() + path.read_bytes())
    return digest.hexdigest()


def test_a_multi_whose_children_name_one_container_is_refused_before_anything_is_built_or_written(state):
    root = state / "clash-bundle"
    envs = root / "envs"
    _items(envs / "items", "v1")
    _items(envs / "stock", "v1")
    (envs / "stock/env.toml").write_text('environment_name = "items"\n')
    (envs / "suite").mkdir()
    (envs / "suite/env.toml").write_text('type = "multi"\nmcp_server_envs = ["items", "stock"]\n')
    (root / "tasks").mkdir()
    (root / "tasks/suite.json").write_text(json.dumps(
        [{"id": "env", "type": "deploy_env", "env_id": "suite", "sandbox_type": "local"}]))
    before = _files(state)

    result = _run(root)

    assert result.exit_code == 1, result.output
    assert ("mcp_server_envs[1]: environment_name 'items' names the container 'items', as mcp_server_envs[0]'s does"
            in result.output), result.output
    assert "building with docker" not in result.output and "Traceback" not in result.output, result.output
    assert _files(state) == before
