"""A step's preflight reads its env at the version the first deploy_env before it in the task pins, else the latest:
through a task's preflight and a bundle's, for the browser steps, load_artifact and snapshot_computer_env."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_env.bundle import materialize
from agent_env.env.envs.computer.app_env import ComputerAppEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.task.task import Task
from agent_env.task_step import preflight
from agent_env.task_step.task_steps import load_artifact
from agent_env.task_step.task_steps.browser.tabs import BrowserActivateTabTaskStep
from agent_env.task_step.task_steps.computer.snapshot_computer_env import SnapshotComputerEnvTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep

_NO_BROWSER = ("step 'early': env 'desk' has no website browser: browser steps act on a computer env's browser app, "
               "or a browser deployed alone")


@pytest.fixture
def asked(monkeypatch) -> list:
    """The (env id, version) each preflight reads, from a store whose desk has chromium at version 1 only."""
    read = []

    def get(env_id, version=None):
        read.append((env_id, version))
        return _desk(browser=version == 1)

    monkeypatch.setattr(preflight, "Env", SimpleNamespace(get=get))
    monkeypatch.setattr(DeployEnvTaskStep, "preflight", lambda self: [])
    return read


def _steps() -> list:
    return [_activate("early"), DeployEnvTaskStep(id="deploy", version=None, env_id="desk", env_version=1),
            _activate("pinned"),
            DeployEnvTaskStep(id="again", version=None, env_id="desk", env_version=2), _activate("still")]


def test_a_tasks_preflight_reads_a_steps_env_at_the_version_the_first_deploy_before_it_pins(asked):
    problems = Task(id="t", version=None, steps=_steps()).preflight()

    assert problems == [_NO_BROWSER]
    assert asked == [("desk", None), ("desk", 1), ("desk", 1)]


def test_a_bundles_preflight_reads_them_so_too(asked):
    steps = _steps()
    write = SimpleNamespace(id="t", source=SimpleNamespace(config=[step.to_dict() for step in steps], outputs=[]))

    problems, unchecked = materialize._preflight(SimpleNamespace(agent=None), write, Task(id="t", version=None,
                                                                                          steps=steps), set())

    assert (problems, unchecked) == ([_NO_BROWSER], [])
    assert asked == [("desk", None), ("desk", 1), ("desk", 1)]


def test_load_artifact_and_snapshot_computer_env_read_the_pinned_version_too(asked, monkeypatch):
    monkeypatch.setattr(load_artifact.Artifact, "get", lambda id, version=None: SimpleNamespace(id=id))
    loading = LoadArtifactTaskStep(id="load", version=None, env_id="desk", artifact_id="seed")
    snapshot = SnapshotComputerEnvTaskStep(id="snap", version=None, env_id="desk")

    with preflight.pinned({"desk": 3}):
        assert loading.preflight() == [] and snapshot.preflight() == []
    assert snapshot.preflight() == []

    assert asked == [("desk", 3), ("desk", 3), ("desk", None)]


def _activate(step_id: str) -> BrowserActivateTabTaskStep:
    return BrowserActivateTabTaskStep(id=step_id, version=None, env_id="desk", url="https://shop.test/")


def _desk(browser: bool) -> ComputerEnv:
    """A stored desktop, without its constructor: chromium among its apps or not."""
    app = ComputerAppEnv.__new__(ComputerAppEnv)
    app.environment_name, app.declaration = "chromium", SimpleNamespace(browser=browser or None)
    desk = ComputerEnv.__new__(ComputerEnv)
    desk.apps = [app]
    return desk
