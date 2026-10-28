"""Tear down the sandboxes a finished run deployed.

``teardown_run`` takes down everything a run's context records: its bare sandboxes, its envs and its agents. It
removes compute only; the instance, its outputs and the artifacts it collected stay. What has to reach the stores goes
first: the changelogs staged on its agents (``before_teardown``), the recordings the run left open, ended with fresh
grants whatever env serves them, and what each env saves as it closes (``Env.close``); a recording still open after
that is registered from what its namespace holds. Then every sandbox is terminated by id, and a local one's work
folder goes once it's down. ``agent-env run`` and ``agent-env eval run`` call it after each run, and the
``teardown_sandboxes`` step takes down part of a run the same way (``tear_down``). ``Task.run`` never does, so a
caller that keeps a run's environment up, as the hub and the worker do, owns its teardown.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, Optional

from agent_env.a2a_agent.staging import drain_all, staged_changelogs
from agent_env.env.env import DeployedEnv, DeployedSandboxEnv, Env
from agent_env.env.envs._deployment import close_deployed
from agent_env.providers.sandbox_providers.local_sandbox import remove_local_work_dir
from agent_env.providers.sandbox_providers.sandbox_provider import (
    build_sandbox_provider,
    get_agent_sandbox_provider,
    get_env_sandbox_provider,
    get_sandbox_provider,
)
from agent_env.task.interrupts import tearing_down
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.computer_recording import end_open_recordings, register_open_recordings

logger = logging.getLogger(__name__)

PROVIDER_CLOSE_TIMEOUT_SECONDS = 5

TERMINATE_TIMEOUT_SECONDS = 120

#: ``context.metadata[TORN_DOWN_KEY]``: the ids of the sandboxes a run has terminated, since its ``deployed_*``
#: records stay (context list fields persist additions only).
TORN_DOWN_KEY = "torn_down_sandbox_ids"


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
        return cls(left=_pending(context, recorded_sandboxes(context)))


def recorded_sandboxes(context: TaskStepContext) -> tuple[RecordedSandbox, ...]:
    """Every sandbox ``context`` records, once each, including those a retry rolled back, which only its
    ``retry_resets`` marker still names. An agent placed on another record's sandbox shares its id, so the
    sandbox is attributed to whichever made it."""
    found: dict[str, RecordedSandbox] = {}
    for sandbox in context.deployed_sandboxes:
        found.setdefault(sandbox.sandbox_id, RecordedSandbox(sandbox.sandbox_id, sandbox.sandbox_type, "sandbox"))
    for env in context.deployed_envs:
        for sandbox_id, sandbox_type in env_sandbox_ids(env):
            found.setdefault(sandbox_id, RecordedSandbox(sandbox_id, sandbox_type, "env"))
    for agent in context.deployed_agents:
        if agent.sandbox_id:
            found.setdefault(agent.sandbox_id, RecordedSandbox(agent.sandbox_id, agent.sandbox_type, "agent"))
    for reset in context.metadata.get("retry_resets") or []:
        gone = reset.get("rolled_back_sandboxes") or {}
        for sandbox in gone.get("deployed_sandboxes") or []:
            found.setdefault(sandbox["sandbox_id"], RecordedSandbox(sandbox["sandbox_id"], sandbox["sandbox_type"], "sandbox"))
        for env in gone.get("deployed_envs") or []:
            for sandbox_id, sandbox_type in _sandbox_ids_of(env["sandbox_id"], env["sandbox_type"], env["sandbox_ids"]):
                found.setdefault(sandbox_id, RecordedSandbox(sandbox_id, sandbox_type, "env"))
        for agent in gone.get("deployed_agents") or []:
            found.setdefault(agent["sandbox_id"], RecordedSandbox(agent["sandbox_id"], agent["sandbox_type"], "agent"))
    return tuple(found.values())


def sandbox_ids(record) -> list[str]:
    """The ids of the sandboxes behind one deployed env, agent or bare sandbox record."""
    if isinstance(record, DeployedEnv):
        return [sandbox_id for sandbox_id, _ in env_sandbox_ids(record)]
    return [record.sandbox_id] if record.sandbox_id else []


async def teardown_run(context: TaskStepContext, *, timeout: float = TERMINATE_TIMEOUT_SECONDS) -> TeardownReport:
    """Take down everything ``context`` records, as ``tear_down`` does, once the changelogs staged on its agents have
    reached the stores.

    Best-effort: a failure is logged and reported, never raised. A sandbox already torn down, by a
    ``teardown_sandboxes`` step or an earlier call, isn't terminated again, though a local one's folder
    still goes. Under ``Interrupts.gather`` the first signal spares it. Cancelling it stops it early: it
    returns what it reached, with the rest in ``left``, rather than raising."""
    tearing_down()
    try:
        await before_teardown(context, timeout)
    except asyncio.CancelledError:
        return TeardownReport.skipped(context)
    return await tear_down(context, recorded_sandboxes(context), context.deployed_envs, timeout=timeout)


async def tear_down(context: TaskStepContext, sandboxes: Sequence[RecordedSandbox], envs: Sequence[DeployedEnv], *,
                    timeout: float) -> TeardownReport:
    """End the recordings the run left open on ``envs`` with fresh grants, close each env still up, register any
    recording still open from its namespace, then terminate each of ``sandboxes`` by id, all at once and each within
    ``timeout`` seconds, and remove a local one's work folder once it's down. A close that fails is logged: its env's sandboxes are terminated by id regardless, which leaves one the
    close took down as it is. Cancelling it returns what it reached, with the rest in ``left``."""
    pending = _pending(context, sandboxes)
    up = {sandbox.sandbox_id for sandbox in pending}
    env_ids = [env.env_id for env in envs]
    try:
        try:
            async with asyncio.timeout(timeout):
                await end_open_recordings(context, env_ids)
        except TimeoutError:
            logger.warning("teardown: the run's open recordings weren't all ended within %gs", timeout)
        await asyncio.gather(*(_close(env, timeout) for env in envs
                               if any(sandbox_id in up for sandbox_id, _ in env_sandbox_ids(env))))
        await asyncio.to_thread(register_open_recordings, context, env_ids)
    except asyncio.CancelledError:
        return TeardownReport(left=pending)
    terminated, failed = [], []

    async def take_down(sandbox: RecordedSandbox) -> None:
        if sandbox not in pending:  # torn down already: only a local one's folder can be left
            if why := await _remove_folder(sandbox):
                failed.append((sandbox, why))
            return
        try:
            async with asyncio.timeout(timeout):
                on_this_machine = await terminate(sandbox)
        except Exception as e:
            reason = f"timed out after {timeout:g}s" if isinstance(e, TimeoutError) else _described(e)
            logger.warning("teardown: %s (%s) was not terminated: %s", sandbox.sandbox_id, kind(sandbox), reason)
            failed.append((sandbox, reason))
            return
        context.metadata[TORN_DOWN_KEY] = [*context.metadata.get(TORN_DOWN_KEY, []), sandbox.sandbox_id]
        if on_this_machine and (why := await _remove_folder(sandbox)):
            failed.append((sandbox, why))
        else:
            terminated.append(sandbox)

    try:
        await asyncio.gather(*(take_down(sandbox) for sandbox in sandboxes))
    except asyncio.CancelledError:
        pass
    reached = {sandbox.sandbox_id for sandbox in terminated} | {sandbox.sandbox_id for sandbox, _ in failed}
    return TeardownReport(tuple(terminated), tuple(failed), tuple(s for s in pending if s.sandbox_id not in reached))


async def before_teardown(context: TaskStepContext, within: float, agent_names: Optional[list[str]] = None) -> None:
    """Drain the changelogs staged on ``agent_names``, or on every agent when None, within ``within`` seconds each."""
    for name in [None] if agent_names is None else agent_names:
        await drain_all(staged_changelogs(context.metadata, agent_name=name), within=within)


def env_sandbox_ids(env: DeployedEnv) -> list[tuple[str, Optional[str]]]:
    """Every (sandbox_id, sandbox_type) of a deployed env: its primary sandbox plus ``sandbox_ids``.

    ``sandbox_ids`` values are an id or a ``{service: id}`` dict and inherit the env's type, unless
    the key itself names a registered backend (a sandbox on another backend than the gateway's).
    """
    if not isinstance(env, DeployedSandboxEnv):  # an env outside our sandboxes owns its own lifetime
        return []
    return _sandbox_ids_of(env.sandbox_id, env.sandbox_type, env.sandbox_ids)


async def terminate(sandbox: RecordedSandbox) -> bool:
    """Terminate ``sandbox`` and say whether it ran on this machine: by id through the backend that made it, or, for
    a record that doesn't name one, through a live handle, since "not found" from a provider that didn't make the
    sandbox doesn't mean it's gone."""
    if sandbox.sandbox_type:
        provider = build_sandbox_provider(sandbox.sandbox_type)
        try:
            await provider.terminate_sandbox(sandbox.sandbox_id)
            return provider.ON_THIS_MACHINE
        finally:
            try:
                async with asyncio.timeout(PROVIDER_CLOSE_TIMEOUT_SECONDS):
                    await provider.close()
            except Exception as error:
                logger.warning("Closing teardown provider %s failed: %s: %s", sandbox.sandbox_type, type(error).__name__, error)
    handle = await _DEFAULT_PROVIDERS[sandbox.slot]().get_sandbox(sandbox.sandbox_id)
    await handle.terminate()
    return handle.ON_THIS_MACHINE


def kind(sandbox: RecordedSandbox) -> str:
    """The backend a sandbox is on, as the CLI prints it."""
    return sandbox.sandbox_type or f"the configured {sandbox.slot} provider"


async def _close(deployed: DeployedEnv, timeout: float) -> None:
    """Close one env a run deployed through its type, so it ends what it runs before its sandboxes go (a computer env,
    its recordings). A type that can't close a deployment leaves them to be terminated by id."""
    try:
        async with asyncio.timeout(timeout):
            await close_deployed(deployed, type(Env.get(deployed.env_id, deployed.env_version)))
    except NotImplementedError:
        return
    except Exception as e:
        reason = f"timed out after {timeout:g}s" if isinstance(e, TimeoutError) else _described(e)
        logger.warning("teardown: env %s wasn't closed (%s); its sandboxes are terminated by id", deployed.env_id, reason)


def _pending(context: TaskStepContext, sandboxes: Sequence[RecordedSandbox]) -> tuple[RecordedSandbox, ...]:
    torn_down = set(context.metadata.get(TORN_DOWN_KEY) or [])
    return tuple(sandbox for sandbox in sandboxes if sandbox.sandbox_id not in torn_down)


async def _remove_folder(sandbox: RecordedSandbox) -> str | None:
    """Remove a local sandbox's work folder, and say why when it can't be."""
    try:
        await asyncio.to_thread(remove_local_work_dir, sandbox.sandbox_id)
    except OSError as e:
        logger.warning("teardown: %s's work folder was not removed: %s", sandbox.sandbox_id, e)
        return f"terminated, but its work folder wasn't removed: {e}"
    return None


def _sandbox_ids_of(
    sandbox_id: Optional[str], sandbox_type: Optional[str], sandbox_ids: Optional[dict],
) -> list[tuple[str, Optional[str]]]:
    ids = [(sandbox_id, sandbox_type)] if sandbox_id else []
    for key, value in (sandbox_ids or {}).items():
        entry_type = key if _is_backend(key) else sandbox_type
        values = [value] if isinstance(value, str) else [v for v in value.values() if isinstance(v, str)]
        ids.extend((v, entry_type) for v in values)
    return ids


def _is_backend(name: str) -> bool:
    try:
        build_sandbox_provider(name)
    except Exception:
        return False
    return True


def _described(error: Exception) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__


_DEFAULT_PROVIDERS = {
    "agent": get_agent_sandbox_provider,
    "env": get_env_sandbox_provider,
    "sandbox": get_sandbox_provider,
}
