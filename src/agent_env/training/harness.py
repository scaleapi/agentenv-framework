"""Train a bundle's own agents with TRL's harness rollouts (``trl.experimental.async_grpo.HarnessRolloutWorker`` with
``harness_adapter=None``): each rollout runs a task as ``agent-env run`` does, its agents' model calls going through
OpenEnv's capture proxy to the trainer's vLLM, which records the tokens TRL trains on, and the task's verifiers score
it."""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import socket
import threading
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

try:
    from openenv.core.harness.capture import CaptureServer, require_llm, to_training_trace
    from openenv.core.harness.capture.export import export_session
    from openenv.core.harness.capture.sessions import Upstream
    from openenv.core.harness.rollout import ResourceSession, ResourceSessionFactory, ToolResult, VerifyResult
    from openenv.core.harness.training import TrainingTrace
except ModuleNotFoundError as exc:
    if exc.name is None or exc.name.split(".")[0] != "openenv":
        raise
    raise ImportError("agent_env.training.harness requires pip install 'agentenv-framework[harness]'") from exc

from agent_env.bundle.plan import solver_steps
from agent_env.bundle.resolve import build_step
from agent_env.bundle.run import planned_tasks, prepare_bundle, solver_params
from agent_env.store.routing import namespace_routing
from agent_env.openenv.reward import reward_of
from agent_env.task.task import Task, recorded_scores
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext, sandbox_overrides
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import is_prompt_step, user_sim_agents
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.training._openenv import on_session_loop

logger = logging.getLogger(__name__)
#: How a row's prompt names its task, since TRL hands the factory only the prompt.
_NAMED = re.compile(r"^AgentEnv task (\d+): (.+)$")
#: Where every deploy runs: local Docker alone reaches the capture proxy on this machine, a remote sandbox needing a
#: tunnel.
_SANDBOX = "local"


def prompt_rows(bundle: str | Path, *, tasks: Sequence[str] = (), evals: Sequence[str] = (),
                agent: str | None = None) -> list[dict[str, Any]]:
    """The trainer's dataset, a row per task ``AgentEnvSessionFactory`` plays with the same arguments: a system line
    naming the task, which is how the factory finds it, then the task's first prompt, for TRL's logs. The model never
    sees this prompt: the task's own agent builds its own."""
    return [{"prompt": [{"role": "system", "content": f"AgentEnv task {index}: {task.id}"},
                        {"role": "user", "content": _first_prompt(task)}], "task_id": task.id}
            for index, (task, _) in enumerate(_trained(bundle, tasks, evals, agent))]


class AgentEnvSessionFactory(ResourceSessionFactory):
    """OpenEnv's session factory for TRL's ``HarnessRolloutWorker``, built in its rollout process. It prepares the
    bundle to run on local Docker, checks that ``llm_url``'s engine returns token ids, and serves OpenEnv's capture
    proxy in front of it, where the task's solver reaches it: the agent a step prompts, which it deploys, its other
    agents and judges keeping the configured endpoint. TRL sets ``model`` to the policy it publishes; ``reward`` is as
    ``build_app``'s, and ``agent`` the agent the solver deploys, as ``agent-env run --agent``'s."""

    def __init__(self, bundle: str | Path, *, llm_url: str, model: str, sampling: dict[str, Any] | None = None,
                 tasks: Sequence[str] = (), evals: Sequence[str] = (),
                 reward: str | Callable[[dict[str, float]], float] | None = None, agent: str | None = None):
        self.model = model
        self._llm_url, self._sampling = llm_url, sampling
        self._trained = _trained(bundle, tasks, evals, agent)
        self._rewards = [_reward(task, reward) for task, _ in self._trained]
        require_llm(llm_url, model, require_tokens=True)
        self._proxy = CaptureServer(llm_url=llm_url, model=model, port=_free_port(),
                                    admin_key=secrets.token_urlsafe(32))
        self._proxy.start()

    def create(self, task: Any, seed: int | None = None, episode_id: str | None = None) -> AgentEnvSession:
        index = _index_named(task, [each.id for each, _ in self._trained])
        played, solver = self._trained[index]
        capture = self._proxy.registry.create(
            upstream=Upstream(llm_url=self._llm_url, model=self.model), capture_level="tokens", purpose="train",
            sampling=self._sampling, task=played.id, episode=episode_id or uuid.uuid4().hex)
        endpoint = {"litellm_base_url": f"http://127.0.0.1:{self._proxy.port}", "litellm_api_key": capture.session_id}
        overrides = {**sandbox_overrides(_SANDBOX),
                     "step_params": {step: {**params, **endpoint} for step, params in solver.items()}}
        return AgentEnvSession(task=played, overrides=overrides, reward=self._rewards[index],
                               registry=self._proxy.registry, capture=capture, model=self.model)


class AgentEnvSession(ResourceSession):
    """One run of a task with its own agents, their model calls captured under this session's key."""

    def __init__(self, *, task: Task, overrides: dict[str, Any], reward: Callable[[dict[str, float]], float | None],
                 registry: Any, capture: Any, model: str):
        self._task, self._overrides, self._reward, self._model = task, overrides, reward, model
        self._registry, self._capture = registry, capture
        self._scores: dict[str, float] | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._task_run: asyncio.Task | None = None  # Task.run, the one part of a run a stop cancels
        self._done: threading.Event | None = None  # set once the run is torn down

    def initial_messages(self) -> list[dict[str, Any]]:
        return [{"role": "user", "content": _first_prompt(self._task)}]

    def list_tools(self) -> list[Any]:
        return []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(error="the task's own agent calls its tools in its sandbox; none are offered here")

    def wait_for_completion(self, timeout_s: float | None = None) -> int:
        """Runs the task and tears it down: 0 when it ran to the end, 1 when it failed or ``close`` stopped it. Raises
        TimeoutError, once the run is torn down, when the task runs past ``timeout_s``."""
        with self._lock:
            if self._closed:
                return 1
            self._done = threading.Event()
        return on_session_loop(self._run(timeout_s), None)

    def fetch_training_trace(self) -> TrainingTrace:
        document = export_session(self._capture, include_messages=True, capture_level="tokens")
        return to_training_trace(self._capture.graph, document)

    def verify(self, transcript: list[dict[str, Any]], final_state: Any | None = None) -> VerifyResult:
        """The task's reward from its recorded scores; ``None``, which TRL leaves out of the update, when the run
        failed or recorded no score its reward reads."""
        try:
            reward = None if self._scores is None else self._reward(self._scores)
        except LookupError:
            reward = None
        return VerifyResult(env_reward=reward, done=True, metrics={f"score/{step}": score
                                                                    for step, score in (self._scores or {}).items()})

    def close(self) -> None:
        """Stops a run still going, as TRL does to the rollouts in flight when it stops, and returns once the run is
        torn down, so no sandbox outlives the trainer; then drops the session's captures."""
        with self._lock:
            stopping, self._closed = not self._closed, True
            task_run, done = self._task_run, self._done
        if stopping and task_run is not None:
            task_run.get_loop().call_soon_threadsafe(task_run.cancel)
        if done is not None:
            done.wait()
        self._registry.delete(self._capture.session_id)

    async def _run(self, timeout: float | None) -> int:
        """``Task.run`` in an asyncio task of its own, which a stop or the timeout cancels, then the teardown in this
        coroutine, which nothing cancels, so a stop can't cut it short."""
        context = TaskStepContext(metadata={"task_id": self._task.id, "user_overrides": self._overrides})
        try:
            with namespace_routing():
                with self._lock:
                    if self._closed:
                        return 1
                    task_run = self._task_run = asyncio.ensure_future(
                        self._task.run(agent_model=self._model, context=context))
                try:
                    await asyncio.wait_for(task_run, timeout)
                    self._scores = recorded_scores(self._task.steps, context)
                    return 0
                except TimeoutError:
                    raise
                except asyncio.CancelledError:
                    if not task_run.cancelled():
                        raise
                    return 1
                except Exception as e:
                    logger.warning("task %s failed, so its rollout is left unscored: %s", self._task.id, e)
                    return 1
                finally:
                    await teardown_run(context)
        finally:
            self._done.set()


def _trained(bundle: str | Path, tasks: Sequence[str], evals: Sequence[str],
             agent: str | None) -> list[tuple[Task, dict[str, dict[str, Any]]]]:
    """The selected tasks that can train their solver, prepared as ``agent-env run`` prepares them, each with its
    solver's deploys and their step params, ``agent``'s if it swaps the solver. The others are left out before they're
    checked, so what they alone need can't stop the rest."""
    with namespace_routing():
        kept = []
        for entry in planned_tasks(bundle, tasks=tasks, evals=evals, agent=agent):
            if why := _untrainable(entry.config):
                logger.warning("%s is left out: %s", entry.entry.name, why)
            else:
                kept.append(entry.entry.name)
        if not kept:
            raise ValueError("no task of the bundle has a solver to train: an agent a step prompts, deployed by a "
                             "deploy_agent step, that isn't also its judge")
        prepared = prepare_bundle(bundle, tasks=kept, sandbox=_SANDBOX, agent=agent, solver_endpoint=True)
    swapped = solver_params(prepared.plan, prepared.materialization)
    return [(task, {step: swapped.get(entry.id, {}).get(step, {})
                    for step in solver_steps([step.to_dict() for step in task.steps])})
            for entry, task in prepared.tasks]


def _untrainable(config: list[dict[str, Any]]) -> str | None:
    """Why a task's rollouts can't train its solver alone, or None."""
    solver = set(solver_steps(config))
    steps = [build_step(step) for step in config]
    deploys = [step for step in steps if step.id in solver]
    if not deploys or not all(isinstance(step, DeployAgentTaskStep) for step in deploys):
        return "training needs the agent a step prompts deployed by a deploy_agent step"
    agents = {step.agent_name for step in deploys}
    if any(isinstance(step, RubricsVerifierTaskStep) and step.use_agent_judge and step.agent_name in agents
           for step in steps):
        return "its rubric judge is its solver, whose grading would be trained on"
    return None


def _reward(task: Task, reward: str | Callable[[dict[str, float]], float] | None
            ) -> Callable[[dict[str, float]], float | None]:
    of, why = reward_of(task, reward)
    if of is None:
        raise ValueError(why)
    return of


def _index_named(prompt: Any, ids: list[str]) -> int:
    for message in prompt if isinstance(prompt, list) else []:
        if isinstance(message, dict) and (named := _NAMED.match(str(message.get("content") or ""))):
            index = int(named[1])
            if index < len(ids) and ids[index] == named[2]:
                return index
    raise KeyError("the row's prompt names no task of this factory's: build the dataset with prompt_rows, from the "
                   "same bundle, tasks and evals")


def _first_prompt(task: Task) -> str:
    """The user's first message to the agent under test: its step's prompt, else the opening of the user sim the step
    names (an always_on step's ``user_sim``, a prompt_agent step's ``user_agent_name``), as it was written. A step that
    prompts an agent another step talks to as its user (the user sim, say a briefing) isn't the agent under test's."""
    users = user_sim_agents(task.steps)
    step = next((each for each in task.steps
                 if is_prompt_step(each) and getattr(each, "agent_name", None) not in users), None)
    if step is None:
        return ""
    if getattr(step, "prompt", None):
        return step.prompt
    user = getattr(step, "user_sim", None) or getattr(step, "user_agent_name", None)
    sim = next((each for each in task.steps
                if user and getattr(each, "agent_name", None) == user and getattr(each, "opening_prompt", None)), None)
    return getattr(sim, "opening_prompt", None) or ""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]
