"""Prompt responses whose turns are atif/1 copies, stored as a prompt step that prescribes atif/1 stores them: the
conversation's document at one url its turns share, its media in the folder's ``media/``, and each turn's record naming
the turn's A2A task and the step it reached. The documents are the protocol's ATIF fixtures."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Optional

from agentenv_protocol.atif import media_paths, tasks

from agent_env.config import get_config
from agent_env.task_step.context import AtifRecord, TurnTrajectory

FIXTURES = Path(str(files("agentenv_protocol.atif").joinpath("fixtures")))


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def folder(run: str = "inst-1", agent: str = "agent") -> str:
    return f"{get_config().get_artifact_key_prefix()}atif/instance_id={run}/{agent}/"


def stored(name: str, *, document: Optional[dict[str, Any]] = None, media: bool = True, run: str = "inst-1") -> str:
    """The fixture ``name`` (or ``document``) stored as an agent's copy, ``<name>.json`` in the run's folder, with
    the media it names beside it unless ``media`` is off; its url."""
    store, where = get_config().get_object_store(), folder(run)
    document = fixture(name) if document is None else document
    for path in media_paths(document) if media else []:
        store.put(f"{where}{path}", (FIXTURES / path).read_bytes(), allow_overwrite=True)
    return store.put(f"{where}{name}.json", json.dumps(document).encode(), content_type="application/json",
                     allow_overwrite=True)


def through_steps(name: str) -> dict[str, int]:
    """Each of the fixture's tasks, by id, with the step its copy reached when that task ended."""
    return {task.task_id: task.steps[-1]["step_id"] for task in tasks(fixture(name))}


def record(url: str, task_id: str, through_step: Optional[int], *, status: str = "valid") -> TurnTrajectory:
    """An atif/1 turn's record: its trajectory is the copy at ``url``."""
    return TurnTrajectory(format="atif/1", atif=AtifRecord(
        url=url, task_id=task_id, through_step=through_step, status=status, rules=[], reason=None,
        validator_version="0.0.0", adapter_version="0.1.0", aux_url=None, turn_context_url=None))


def native(task_id: str, native_format: str = "claude-code-stream-json") -> TurnTrajectory:
    """A turn whose copy couldn't be had, its native trajectory read in its place."""
    return TurnTrajectory(format=native_format, atif=AtifRecord(
        url=None, task_id=task_id, through_step=None, status="unavailable", rules=[], reason="timeout",
        validator_version=None, adapter_version=None, aux_url=None, turn_context_url=None))
