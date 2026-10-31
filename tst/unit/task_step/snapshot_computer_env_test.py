"""snapshot_computer_env: snapshot_env for a computer env, its file_paths, relative to the session user's home, the
data/get params of the computer's own environment and of nothing else; refused for an env that isn't a computer, and
for paths that aren't relative, before a run."""

from __future__ import annotations

import pytest

from agent_env.artifact import EnvironmentUniverseArtifact
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.computer.snapshot_computer_env import SnapshotComputerEnvTaskStep
from tst.unit.task_step.snapshot_env_without_gateway_test import _computer, _inline, _record, _serve
from tst.unit.task_step.test_snapshot_env import _ctx


class _Desk(ComputerEnv):
    """A stored computer env, as far as a snapshot reads one."""

    def __init__(self):
        self.id, self.version, self.environment_name = "desk", 1, "desktop"


class _Mail:
    id, version, type, environment_name = "mail", 1, "mcp_server", "mail"


def test_its_file_paths_round_trip_and_its_type_is_registered():
    step = SnapshotComputerEnvTaskStep(id="snap", version=None, env_id="desk", file_paths=["Documents", "Desktop/a.pdf"],
                                       include_env_trajectory=True)

    restored = get_task_step_registry()["snapshot_computer_env"].from_dict(step.to_dict())

    assert isinstance(restored, SnapshotComputerEnvTaskStep) and step.to_dict()["type"] == "snapshot_computer_env"
    assert (restored.env_id, restored.file_paths, restored.include_env_trajectory) == (
        "desk", ["Documents", "Desktop/a.pdf"], True)


@pytest.mark.parametrize("paths", [["/etc/passwd"], ["../outside"], ["Documents/../.."]],
                         ids=["absolute", "parent", "escapes"])
def test_file_paths_not_relative_to_the_home_are_refused_before_a_run(paths):
    step = SnapshotComputerEnvTaskStep(id="snap", version=None, env_id="desk", file_paths=paths)

    assert [problem for problem in step.config_problems() if "file_paths are normalized paths" in problem]


@pytest.mark.parametrize("stored, refused", [(_Mail(), True), (_Desk(), False)], ids=["mcp-server", "computer"])
def test_an_env_that_isnt_a_computer_is_refused_before_a_run(monkeypatch, stored, refused):
    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: stored))

    problems = SnapshotComputerEnvTaskStep(id="snap", version=None, env_id=stored.id).preflight()

    assert problems == (["snapshot_computer_env snapshots a computer env, and mail is a mcp_server env"]
                        if refused else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("file_paths, params", [(["Documents"], {"paths": ["Documents"]}), (None, {})],
                         ids=["file-paths", "the-cards-folders"])
async def test_its_file_paths_are_the_computers_own_data_get_params(local_stores, monkeypatch, file_paths, params):
    sent = _serve(monkeypatch, {"/agentenv": lambda params: [_inline("Documents/notes.txt", b"hello")]})
    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: _Desk()))
    context = _ctx(_record(_computer()))

    await SnapshotComputerEnvTaskStep(id="snap", version=None, env_id="desk", file_paths=file_paths).execute(context)

    snapshotted = context.metadata["env_snapshotted_universes"]["snap"]
    universe = EnvironmentUniverseArtifact.get(snapshotted["id"], version=snapshotted["version"])
    assert sorted(universe.get_file_artifacts()) == ["desktop/Documents/notes.txt"]
    assert sent == [("/agentenv", params)]


@pytest.mark.asyncio
async def test_a_run_on_an_env_that_isnt_a_computer_is_refused_before_any_export(local_stores, monkeypatch):
    sent = _serve(monkeypatch, {})
    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: _Mail()))

    with pytest.raises(RuntimeError, match="snapshots a computer env, and mail is a mcp_server env"):
        await SnapshotComputerEnvTaskStep(id="snap", version=None, env_id="desk").execute(_ctx(_record(_computer())))
    assert sent == []
