"""snapshot_computer_env: a computer env's files kept by their paths in its environment artifact, in a new environment
universe beside each of its apps' exports, returned inline or uploaded under the grant and written where they were put,
the computer asked again without the grant when it fails under it, file_paths sent to its data/get alone, and an app
that keeps nothing yet left out; a multi env's computer, the one named or its only one, kept so beside the multi env's
other environments; refused for an env that isn't a computer nor a multi env with the one named, for paths that aren't
relative, and with no files to keep."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
from agentenv_protocol import uploaded_file_part
from agentenv_protocol.computer import COMPUTER_EXEC

from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact
from agent_env.config import set_object_store
from agent_env.env.env import DeployedGatewayEnv, DeployedSandboxEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.computer.snapshot_computer_env import SnapshotComputerEnvTaskStep
from tst.unit.task_step.computer_exec_test import _stored_multi
from tst.unit.task_step.snapshot_env_without_gateway_test import (DATA_OBJECTS, FILE_TREE, UNREACHABLE, _computer,
                                                                   _inline, _record, _serve)
from tst.unit.task_step.test_snapshot_env import _ctx
from tst.util.granting_object_store import GrantingObjectStore

GATEWAY = "http://gateway.test"


class _Desk(ComputerEnv):
    """A stored computer env, as far as a snapshot reads one."""

    def __init__(self):
        self.id, self.version, self.environment_name = "desk", 1, "desktop"


class _Mail:
    id, version, type, environment_name = "mail", 1, "mcp_server", "mail"


class _Office(MultiEnv):
    """A stored multi env, as far as a snapshot reads one: the MCP server mail beside its computers."""

    def __init__(self):
        self.id, self.version = "office", 1
        self.mcp_server_envs, self.website_envs = [SimpleNamespace(environment_name="mail")], []


def _office(*computers: str) -> DeployedGatewayEnv:
    """The multi env deployed behind a gateway fronting mail, each of ``computers`` on a sandbox of its own, its data
    plane at ``/<name>/agentenv``."""
    def computer(name: str) -> dict:
        card = {**_computer(), "name": name, "url": f"/{name}/agentenv"}
        card["capabilities"] = {"operations": ["data/add", "data/get"], "extensions": [
            FILE_TREE, {"uri": COMPUTER_EXEC, "params": {"methods": {"run": {}}}}]}
        return dataclasses.asdict(DeployedSandboxEnv(
            env_id=name, env_version=1, sandbox_id=f"sb-{name}", instance_id=f"{name}-abc", sandbox_type="local",
            environment_card_url=f"http://{name}.test{WELL_KNOWN_PATH}", environment_card=card))

    card = {"name": "office", "children_environments": [
        {"name": "mail", "url": "/svc/mcp-mail/agentenv", "capabilities": {"operations": ["data/get"]}}]}
    return DeployedGatewayEnv(env_id="office", env_version=1, gateway_url=GATEWAY, sandbox_id="sb-gw",
                              instance_id="office-abc", environment_card_url=f"{GATEWAY}{WELL_KNOWN_PATH}",
                              environment_card=card, sandbox_type="local",
                              child_envs={name: computer(name) for name in computers})


async def _snapshot(monkeypatch, card: dict, env=None, **fields) -> EnvironmentArtifact:
    """The computer's environment artifact in the universe the step snapshots it into."""
    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: env or _Desk()))
    context = _ctx(_record(card))
    await SnapshotComputerEnvTaskStep(id="snap", version=None, env_id="desk", **fields).execute(context)
    snapshotted = context.metadata["env_snapshotted_universes"]["snap"]
    (desktop,) = EnvironmentUniverseArtifact.get(snapshotted["id"], version=snapshotted["version"]).get_environment_artifacts()
    assert desktop.environment_name == "desktop"
    return desktop


def _kept(desktop: EnvironmentArtifact) -> dict[str, bytes]:
    return {path: file.load() for path, file in desktop.get_file_artifacts().items()}


def test_its_fields_round_trip_and_its_type_is_registered():
    step = SnapshotComputerEnvTaskStep(id="snap", version=None, env_id="office", computer_name="desk",
                                       file_paths=["Documents", "Desktop/a.pdf"], export_timeout_seconds=60)

    data = step.to_dict()
    restored = get_task_step_registry()["snapshot_computer_env"].from_dict(data)

    assert isinstance(restored, SnapshotComputerEnvTaskStep) and data["type"] == "snapshot_computer_env"
    assert (restored.env_id, restored.computer_name, restored.file_paths, restored.export_timeout_seconds) == (
        "office", "desk", ["Documents", "Desktop/a.pdf"], 60)
    assert not {"gateway_url", "include_env_trajectory", "include_env_changes"} & set(data)


@pytest.mark.parametrize("paths", [["/etc/passwd"], ["../outside"], ["Documents/../.."]],
                         ids=["absolute", "parent", "escapes"])
def test_file_paths_not_relative_to_the_home_are_refused_before_a_run(paths):
    step = SnapshotComputerEnvTaskStep(id="snap", version=None, env_id="desk", file_paths=paths)

    assert [problem for problem in step.config_problems() if "file_paths are normalized paths" in problem]


@pytest.mark.parametrize("stored, computer, problem", [
    (_Mail(), None, "env 'mail' is no computer env, nor a multi env with one"),
    (_Desk(), None, None),
    (_Desk(), "desktop", "env 'desk' is a computer env, so it takes no computer_name"),
    (_stored_multi("desk"), None, None),
    (_stored_multi("desk", "laptop"), "laptop", None),
    (_stored_multi("desk", "laptop"), None,
     "env 'office' has 2 computers: name one as computer_name, of ['desk', 'laptop']"),
    (_stored_multi("desk"), "kiosk", "env 'office' has no computer named 'kiosk' (it has ['desk'])"),
], ids=["mcp-server", "computer", "computer-named", "multi-only-computer", "multi-named", "multi-two-unnamed",
        "multi-unknown-name"])
def test_an_env_that_isnt_a_computer_nor_a_multi_env_with_the_one_named_is_refused_before_a_run(
        monkeypatch, stored, computer, problem):
    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: stored))

    problems = SnapshotComputerEnvTaskStep(id="snap", version=None, env_id=stored.id,
                                           computer_name=computer).preflight()

    assert problems == ([] if problem is None else [f"snapshot_computer_env 'snap': {problem}"])


@pytest.mark.asyncio
async def test_a_computer_with_apps_keeps_its_files_beside_each_apps_export_sending_file_paths_to_itself_alone(
        local_stores, monkeypatch):
    card = {**_computer(), "children_environments": [
        {"name": name, "url": f"/svc/{name}/agentenv", "capabilities": {
            "operations": ["data/add", "data/get"], "extensions": [FILE_TREE]}} for name in ("mail", "notes")]}
    card["capabilities"] = {**card["capabilities"], "operations": ["data/add", "data/get"]}
    sent = _serve(monkeypatch, {"/agentenv": lambda params: [_inline("Documents/a.txt", b"a")],
                                "/svc/mail/agentenv": lambda params: [_inline("Mail/inbox.txt", b"m")],
                                "/svc/notes/agentenv": lambda params: []})
    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: _Desk()))
    context = _ctx(_record(card))

    await SnapshotComputerEnvTaskStep(id="snap", version=None, env_id="desk", file_paths=["Documents"]).execute(context)

    snapshotted = context.metadata["env_snapshotted_universes"]["snap"]
    universe = EnvironmentUniverseArtifact.get(snapshotted["id"], version=snapshotted["version"])
    assert {environment.environment_name: _kept(environment) for environment in universe.get_environment_artifacts()
            } == {"desktop": {"Documents/a.txt": b"a"}, "mail": {"Mail/inbox.txt": b"m"}}
    assert sorted((path, sorted(params)) for path, params in sent) == [("/agentenv", ["paths"]),
                                                                       ("/svc/mail/agentenv", []),
                                                                       ("/svc/notes/agentenv", [])]


@pytest.mark.asyncio
@pytest.mark.parametrize("computers, computer", [(("desk", "laptop"), "laptop"), (("laptop",), None)],
                         ids=["the-one-named", "its-only-one"])
async def test_a_multi_envs_computer_is_kept_beside_its_other_environments_with_file_paths_sent_to_it_alone(
        local_stores, monkeypatch, computers, computer):
    sent = _serve(monkeypatch, {"/svc/mcp-mail/agentenv": lambda params: [{"kind": "data", "data": {"inbox": 2}}],
                                "/desk/agentenv": lambda params: [_inline("Documents/d.txt", b"d")],
                                "/laptop/agentenv": lambda params: [_inline("notes.txt", b"n")]})
    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: _Office()))
    context = _ctx(_office(*computers))

    await SnapshotComputerEnvTaskStep(id="snap", version=None, env_id="office", computer_name=computer,
                                      file_paths=["notes.txt"]).execute(context)

    snapshotted = context.metadata["env_snapshotted_universes"]["snap"]
    kept = {environment.environment_name: environment for environment in EnvironmentUniverseArtifact.get(
        snapshotted["id"], version=snapshotted["version"]).get_environment_artifacts()}
    assert sorted(kept) == sorted(["mail", *computers]) and _kept(kept["laptop"]) == {"notes.txt": b"n"}
    assert sorted(sent) == sorted([("/svc/mcp-mail/agentenv", {}), ("/laptop/agentenv", {"paths": ["notes.txt"]}),
                                   *([("/desk/agentenv", {})] if "desk" in computers else [])])


@pytest.mark.asyncio
@pytest.mark.parametrize("computers, computer, refused", [
    (("desk", "laptop"), None, r"env 'office' has 2 computers serving .* run: name one as computer_name"),
    (("desk",), "kiosk", r"env 'office' has no computer named 'kiosk' \(it has \['desk'\]\)"),
], ids=["two-unnamed", "an-unknown-name"])
async def test_a_multi_env_without_the_computer_named_is_refused(local_stores, monkeypatch, computers, computer,
                                                                 refused):
    sent = _serve(monkeypatch, {})
    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: _Office()))

    with pytest.raises(RuntimeError, match=refused):
        await SnapshotComputerEnvTaskStep(id="snap", version=None, env_id="office", computer_name=computer,
                                          file_paths=["notes.txt"]).execute(_ctx(_office(*computers)))
    assert sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("file_paths, params", [(["Documents"], {"paths": ["Documents"]}), (None, {})],
                         ids=["file-paths", "the-cards-folders"])
async def test_each_file_returned_inline_is_kept_by_its_path(local_stores, monkeypatch, file_paths, params):
    sent = _serve(monkeypatch, {"/agentenv": lambda params: [_inline("Documents/notes.txt", b"hello"),
                                                             _inline("Documents/b/c.bin", b"\x00\x01")]})

    desktop = await _snapshot(monkeypatch, _computer(), file_paths=file_paths)

    assert _kept(desktop) == {"Documents/notes.txt": b"hello", "Documents/b/c.bin": b"\x00\x01"}
    assert desktop.get_file_artifacts()["Documents/b/c.bin"].filename == "c.bin"
    assert sent == [("/agentenv", params)]


@pytest.mark.asyncio
async def test_files_uploaded_under_the_grant_are_written_where_the_computer_put_them_without_a_download(
        local_stores, tmp_path, monkeypatch):
    store = GrantingObjectStore(str(tmp_path / "objects"))
    set_object_store(store)
    files = {"Documents/notes.txt": b"hello", "Downloads/c.bin": b"\x00"}

    def uploading(params):
        root = params["write_namespace"]["root_path"]
        for path, data in files.items():
            store.put(f"{root}/{path}", data, content_type="application/octet-stream")
        return [uploaded_file_part(path, name=Path(path).name, mime_type="text/plain").model_dump(
            mode="json", exclude_none=True) for path in files]

    sent = _serve(monkeypatch, {"/agentenv": uploading})

    desktop = await _snapshot(monkeypatch, _computer(DATA_OBJECTS))

    ((_, params),) = sent
    root = params["write_namespace"]["root_path"]
    assert params["write_namespace"]["max_objects"] == 100_000
    assert {path: (file.object_url, file.load()) for path, file in desktop.get_file_artifacts().items()} == {
        path: (store.object_url(f"{root}/{path}"), data) for path, data in files.items()}


@pytest.mark.asyncio
async def test_a_computer_that_fails_under_the_grant_is_asked_again_without_one(local_stores, tmp_path, monkeypatch):
    set_object_store(GrantingObjectStore(str(tmp_path / "objects")))
    sent = _serve(monkeypatch, {"/agentenv": lambda params: UNREACHABLE if "write_namespace" in params
                                else [_inline("Desktop/left.txt", b"x")]})

    desktop = await _snapshot(monkeypatch, _computer(DATA_OBJECTS))

    assert _kept(desktop) == {"Desktop/left.txt": b"x"}
    assert [sorted(params) for _, params in sent] == [["write_namespace"], []]


@pytest.mark.asyncio
@pytest.mark.parametrize("env, answers, refused", [
    (_Mail(), {}, "env 'mail' is no computer env, nor a multi env with one"),
    (None, {"/agentenv": lambda params: []}, "'desktop' returned no files to snapshot"),
], ids=["not-a-computer", "no-files"])
async def test_a_run_with_nothing_a_computer_snapshot_can_keep_is_refused(local_stores, monkeypatch, env, answers,
                                                                          refused):
    _serve(monkeypatch, answers)

    with pytest.raises(RuntimeError, match=refused):
        await _snapshot(monkeypatch, _computer(), env=env, file_paths=["Documents"])
