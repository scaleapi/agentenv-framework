"""load_artifact with ``computer``: a file artifact universe loaded into one of a multi env's computers, by its
environment name, through that computer's own driver (``tst.unit.computer_home``), instead of onto the env's host; and
what it refuses: a computer the env hasn't, anything but files, and a computer with no env."""

from __future__ import annotations

import dataclasses

import pytest

from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.envs.computer.utils import inside
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from tst.unit.computer_home import TOKEN, Computer


@pytest.fixture
def computer(tmp_path, monkeypatch):
    monkeypatch.setattr(inside, "issues_grants_to", lambda store, sandbox_type: False)
    (tmp_path / "home").mkdir()
    return Computer(tmp_path / "home").install(monkeypatch)


def _office(computer: Computer, *names: str) -> TaskStepContext:
    office = DeployedGatewayEnv(env_id="office", env_version=1, gateway_url="http://gw", sandbox_id="sb-gw",
                                environment_card={"name": "office"},
                                child_envs={name: dataclasses.asdict(computer.deployed(env_id=name)) for name in names})
    return TaskStepContext(deployed_agents=[], deployed_envs=[office], metadata={})


def _universe(tmp_path, **contents) -> FileArtifactUniverse:
    for name, data in contents.items():
        (tmp_path / "files" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "files" / name).write_bytes(data)
    return FileArtifactUniverse.put_bundled("kit-files", files={
        name: tmp_path / "files" / name for name in contents})


@pytest.mark.asyncio
async def test_files_load_into_the_named_computer_of_a_multi_env_relative_to_its_home(local_stores, computer, tmp_path):
    universe = _universe(tmp_path, **{"Desktop/a.xlsx": b"sheet", "notes/b.txt": b"note"})
    step = LoadArtifactTaskStep(id="files", version=None, env_id="office", computer="desk",
                                artifact_id=universe.id, artifact_version=universe.version, destination_path=".")

    ctx = await step.execute(_office(computer, "desk", "chromium"))

    assert (computer.home / "Desktop/a.xlsx").read_bytes() == b"sheet"
    assert (computer.home / "notes/b.txt").read_bytes() == b"note"
    assert computer.headers == [{"Authorization": f"Bearer {TOKEN}"}]
    [loaded] = ctx.metadata["loaded_file_artifact_universes"]
    assert loaded["env_id"] == "office" and loaded["computer"] == "desk" and loaded["destination_path"] == "."
    assert loaded["files"] == {"Desktop/a.xlsx": "Desktop/a.xlsx", "notes/b.txt": "notes/b.txt"}


@pytest.mark.asyncio
async def test_a_computer_the_env_hasnt_is_refused_before_anything_is_written(local_stores, computer, tmp_path):
    universe = _universe(tmp_path, **{"a.txt": b"a"})
    step = LoadArtifactTaskStep(id="files", version=None, env_id="office", computer="laptop",
                                artifact_id=universe.id, artifact_version=universe.version)

    with pytest.raises(RuntimeError, match=r"env 'office' has no computer named 'laptop' \(it has \['desk'\]\)"):
        await step.execute(_office(computer, "desk"))
    assert computer.calls == []


@pytest.mark.asyncio
async def test_of_two_deployments_of_the_env_the_files_go_to_the_computer_of_the_one_env_step_id_names(
        local_stores, computer, tmp_path):
    universe = _universe(tmp_path, **{"a.txt": b"a"})

    def office(step_id: str, port: int) -> DeployedGatewayEnv:
        desk = dataclasses.replace(computer.deployed(env_id="desk"),
                                   environment_card_url=f"http://127.0.0.1:{port}/.well-known/agent-env.json")
        return DeployedGatewayEnv(env_id="office", env_version=1, gateway_url=f"http://gw-{port}", sandbox_id="sb",
                                  environment_card={"name": "office"}, metadata={"deploy_step_id": step_id},
                                  child_envs={"desk": dataclasses.asdict(desk)})

    ctx = TaskStepContext(deployed_agents=[], deployed_envs=[office("first", 20001), office("second", 20002)],
                          metadata={})
    step = LoadArtifactTaskStep(id="files", version=None, env_id="office", env_step_id="second", computer="desk",
                                artifact_id=universe.id, artifact_version=universe.version)

    await step.execute(ctx)

    assert computer.added_at == ["http://127.0.0.1:20002"]


def test_a_computer_needs_an_env_and_must_be_a_name():
    with pytest.raises(ValueError, match="`computer` names one of the computers of the env `env_id`"):
        LoadArtifactTaskStep(id="files", version=None, agent_name="a", computer="desk", artifact_id="x")
    for target in ({"sandbox_name": "box"}, {"sandbox_name": "box", "container_name": "c"}, {"agent_name": "a"}):
        with pytest.raises(ValueError, match="takes no `sandbox_name`, `container_name` or `agent_name`"):
            LoadArtifactTaskStep(id="files", version=None, env_id="office", computer="desk", artifact_id="x",
                                 **target)
    step = LoadArtifactTaskStep(id="files", version=None, env_id="office", computer="", artifact_id="x")
    assert "computer, when given, must be a non-empty string" in step.config_problems()[-1]


def test_computer_round_trips_and_is_left_out_when_unset():
    step = LoadArtifactTaskStep(id="files", version=None, env_id="office", computer="desk", artifact_id="x",
                                artifact_version=2, destination_path=".")
    again = get_task_step_registry()["load_artifact"].from_dict(step.to_dict())
    assert again.computer == "desk" and again.to_dict() == step.to_dict()
    assert "computer" not in LoadArtifactTaskStep(id="f", version=None, env_id="e", artifact_id="x").to_dict()


def test_preflight_names_a_computer_the_stored_env_hasnt(monkeypatch):
    from tst.unit.task_step.computer_exec_test import _stored_multi
    monkeypatch.setattr("agent_env.task_step.task_steps.load_artifact.stored_env",
                        lambda id: _stored_multi("desk", browser="chromium"))

    def preflight(computer):
        return LoadArtifactTaskStep(id="files", version=None, env_id="office", computer=computer,
                                    artifact_id="x").preflight()

    assert preflight("desk") == [] and preflight("chromium") == []
    assert preflight("laptop") == [
        "load_artifact 'files': env 'office' has no computer named 'laptop' (it has ['desk', 'chromium'])"]
