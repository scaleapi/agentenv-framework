"""Tear down the sandboxes a finished run deployed.

``teardown_run`` terminates every sandbox a run's context records: its bare sandboxes', its envs' and its
agents'. It removes compute only; the instance, its outputs and the artifacts it collected stay. A local
sandbox's work folder goes too, once the sandbox is down. ``agent-env run`` and ``agent-env eval run`` call
it after each run. ``Task.run`` never does, so a caller that keeps a run's environment up, as the hub and
the worker do, owns its teardown.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import NamedTuple, Optional

from agent_env.a2a_agent.staging import drain_all, staged_changelogs
from agent_env.env.env import DeployedEnv
from agent_env.providers.sandbox_providers.local_sandbox import remove_local_work_dir
from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider
from agent_env.task.interrupts import tearing_down
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.teardown_sandboxes import _DEFAULT_PROVIDERS, TORN_DOWN_KEY, _env_sandbox_ids

logger = logging.getLogger(__name__)

TERMINATE_TIMEOUT_SECONDS = 120


class RecordedSandbox(NamedTuple):
    sandbox_id: str
    sandbox_type: Optional[str]  # the backend that made it; None leaves it to the slot's configured provider
    slot: str  # "sandbox", "env" or "agent"


@dataclass(frozen=True)
class TeardownReport:
    terminated: tuple[RecordedSandbox, ...] = ()
    failed: tuple[tuple[RecordedSandbox, str], ...] = ()  # each with why
    left: tuple[RecordedSandbox, ...] = ()  # not reached, because the teardown was stopped

    @property
    def still_up(self) -> tuple[RecordedSandbox, ...]:
        """What may still be running: what failed to terminate and what the teardown didn't reach."""
        return (*(sandbox for sandbox, _ in self.failed), *self.left)

    @classmethod
    def skipped(cls, context: TaskStepContext) -> TeardownReport:
        """A teardown that didn't start: everything ``context`` records that's still up is left."""
        return cls(left=_pending(context))


def recorded_sandboxes(context: TaskStepContext) -> tuple[RecordedSandbox, ...]:
    """Every sandbox ``context`` records, once each. An agent placed on another record's sandbox shares its
    id, so the sandbox is attributed to whichever made it."""
    found: dict[str, RecordedSandbox] = {}
    for sandbox in context.deployed_sandboxes:
        found.setdefault(sandbox.sandbox_id, RecordedSandbox(sandbox.sandbox_id, sandbox.sandbox_type, "sandbox"))
    for env in context.deployed_envs:
        for sandbox_id, sandbox_type in _env_sandbox_ids(env):
            found.setdefault(sandbox_id, RecordedSandbox(sandbox_id, sandbox_type, "env"))
    for agent in context.deployed_agents:
        if agent.sandbox_id:
            found.setdefault(agent.sandbox_id, RecordedSandbox(agent.sandbox_id, agent.sandbox_type, "agent"))
    return tuple(found.values())


def sandbox_ids(record) -> list[str]:
    """The ids of the sandboxes behind one deployed env, agent or bare sandbox record."""
    if isinstance(record, DeployedEnv):
        return [sandbox_id for sandbox_id, _ in _env_sandbox_ids(record)]
    return [record.sandbox_id] if record.sandbox_id else []


async def teardown_run(context: TaskStepContext, *, timeout: float = TERMINATE_TIMEOUT_SECONDS) -> TeardownReport:
    """Terminate every sandbox ``context`` records, all at once, each within ``timeout`` seconds, and remove a
    local one's work folder once it's down.

    Best-effort: a failure is logged and reported, never raised. A sandbox already torn down, by a
    ``teardown_sandboxes`` step or an earlier call, isn't terminated again, though a local one's folder
    still goes. Under ``Interrupts.gather`` the first signal spares it. Cancelling it stops it early: it
    returns what it reached, with the rest in ``left``, rather than raising."""
    tearing_down()
    pending = _pending(context)
    terminated, failed = [], []
    try:  # a changelog staged on an agent goes into the store before the agent does
        async with asyncio.timeout(timeout):
            await drain_all(staged_changelogs(context.metadata))
    except TimeoutError:
        logger.warning("teardown: staged changelog increments were not all drained within %gs", timeout)
    except asyncio.CancelledError:
        return TeardownReport(left=pending)

    async def take_down(sandbox: RecordedSandbox) -> None:
        if sandbox not in pending:  # torn down already: only a local one's folder can be left
            if why := await _remove_folder(sandbox):
                failed.append((sandbox, why))
            return
        try:
            async with asyncio.timeout(timeout):
                handle = await _terminate(sandbox)
        except Exception as e:
            reason = f"timed out after {timeout:g}s" if isinstance(e, TimeoutError) else _described(e)
            logger.warning("teardown: %s (%s) was not terminated: %s", sandbox.sandbox_id, kind(sandbox), reason)
            failed.append((sandbox, reason))
            return
        context.metadata[TORN_DOWN_KEY] = [*context.metadata.get(TORN_DOWN_KEY, []), sandbox.sandbox_id]
        if handle.ON_THIS_MACHINE and (why := await _remove_folder(sandbox)):
            failed.append((sandbox, why))
        else:
            terminated.append(sandbox)

    try:
        await asyncio.gather(*(take_down(sandbox) for sandbox in recorded_sandboxes(context)))
    except asyncio.CancelledError:
        pass
    reached = {sandbox.sandbox_id for sandbox in terminated} | {sandbox.sandbox_id for sandbox, _ in failed}
    return TeardownReport(tuple(terminated), tuple(failed), tuple(s for s in pending if s.sandbox_id not in reached))


def _pending(context: TaskStepContext) -> tuple[RecordedSandbox, ...]:
    torn_down = set(context.metadata.get(TORN_DOWN_KEY) or [])
    return tuple(sandbox for sandbox in recorded_sandboxes(context) if sandbox.sandbox_id not in torn_down)


async def _remove_folder(sandbox: RecordedSandbox) -> str | None:
    """Remove a local sandbox's work folder, and say why when it can't be."""
    try:
        await asyncio.to_thread(remove_local_work_dir, sandbox.sandbox_id)
    except OSError as e:
        logger.warning("teardown: %s's work folder was not removed: %s", sandbox.sandbox_id, e)
        return f"terminated, but its work folder wasn't removed: {e}"
    return None


async def _terminate(sandbox: RecordedSandbox):
    provider = (build_sandbox_provider(sandbox.sandbox_type) if sandbox.sandbox_type
                else _DEFAULT_PROVIDERS[sandbox.slot]())
    handle = await provider.get_sandbox(sandbox.sandbox_id)
    await handle.terminate()
    return handle


def _described(error: Exception) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


def kind(sandbox: RecordedSandbox) -> str:
    """The backend a sandbox is on, as the CLI prints it."""
    return sandbox.sandbox_type or f"the configured {sandbox.slot} provider"
