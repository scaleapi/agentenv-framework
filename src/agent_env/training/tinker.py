"""Tinker's RL on a bundle's tasks: tinker-cookbook's ``rl.train``, each rollout one episode of a task
(``agent_env.task.EpisodeSpec``) whose agent the model being trained plays. ``AgentEnvDatasetBuilder`` is the
``dataset_builder`` of ``rl.train.Config``: each batch is groups of episodes, a group playing one task, and every
episode is a ``Task.run`` of its own that the task's verifiers score as ``agent-env run`` would.

The model reads the turn's system prompt, its tools (the envs' tools, and ``bash`` when its agent works in a sandbox)
as the cookbook's renderer writes them, and the turn's prompt. Each sampled message is recorded on the turn before its
tool calls run; a message without tool calls ends the turn, and the run's next turn, if it has one, comes as the next
user message. Once the run ends, the rollout's reward is the task's reward of its scores by the rules ``build_app``
applies (``agent_env.openenv.reward``: the score at the step id ``reward`` names, or the task's one score), and each
score is a metric ``score/<step id>``.

The system prompt and the tools are written once, at the start of the conversation, as the cookbook's renderers write
them. A later turn that changes the system prompt has the new one written before its prompt, as every trainer's model
reads it (``agent_env.training._episodes.next_prompt``); a turn that offers other tools than the first fails its
rollout; and a task that gives the agent skills, which no trainer shows its model yet, is left out (``playable``).

An episode lives on the event loop that started it, so every rollout runs on ``rl.train``'s: set no rollout executor.
A rollout the cookbook ends itself (a structural parse failure, the context or token limit) is scored the moment it is
cut (``_ScoredWhenCut``): its turn ends with no reply, the verifiers score where the episode got to, that score is added
to the cookbook's own reward for the cut, ``failed_reward``, and the episode's slot under ``max_concurrent_episodes`` is
freed. The metric ``cut`` is 1 for such a rollout and 0 for the rest."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

try:
    import chz
    from tinker_cookbook import tokenizer_utils
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.renderers.base import Message, Renderer, ToolCall, ToolSpec, get_text_content
    from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
    from tinker_cookbook.rl.types import (Action, ActionExtra, Env, EnvGroupBuilder, InitialObservationOverflow,
                                          RLDataset, RLDatasetBuilder, StepResult)
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".")[0] not in ("chz", "tinker", "tinker_cookbook"):
        raise
    raise ImportError("agent_env.training.tinker runs in tinker-cookbook's environment: install tinker-cookbook and "
                      "'agentenv-framework[tinker]' there") from exc

from agent_env.bundle import BundleKind, parse_bundle
from agent_env.bundle.run import prepare_bundle
from agent_env.store.routing import namespace_routing
from agent_env.task import Episode, EpisodeError, EpisodeSpec, Task, Turn
from agent_env.openenv.reward import reward_of
from agent_env.training._episodes import RunGroup, TrainingSteps, next_prompt, typed_arguments

logger = logging.getLogger(__name__)

Reward = Callable[[dict[str, float]], float | None]

#: Per event loop, the episodes that may be under way at once, by limit: a semaphore can't be pickled with a builder.
_SLOTS: dict[tuple[int, int], asyncio.Semaphore] = {}


def _slots(limit: int) -> asyncio.Semaphore:
    key = (id(asyncio.get_running_loop()), limit)
    if key not in _SLOTS:
        _SLOTS[key] = asyncio.Semaphore(limit)
    return _SLOTS[key]


class EpisodeEnv(MessageEnv):
    """One episode of ``spec``'s task as a cookbook ``MessageEnv``: started at the first observation, played turn by
    turn, rewarded from its scores once the run ends. ``max_turns`` bounds the model's messages; at the bound the turn
    ends with what the model last said, and every later turn with no reply, so the verifiers score where it got to.

    The episode holds a slot under ``max_concurrent_episodes`` from its start until its run is scored or abandoned,
    since a run is containers of its own. The system prompt and the tools are the first turn's, written once; a later
    turn's new system prompt comes before its prompt (``next_prompt``)."""

    #: The most of an assistant message, a call's arguments or a tool's answer a step logs, in characters.
    LOGGED_CHARS: ClassVar[int] = 2000

    def __init__(self, name: str, spec: EpisodeSpec, *, renderer: Renderer, reward: Reward, model_name: str,
                 sandbox: str | None, max_turns: int, tool_timeout: float, max_concurrent_episodes: int,
                 run_group_id: str):
        self._name = name
        self._run_group_id = run_group_id
        self._spec = spec
        self._renderer = renderer
        self._reward = reward
        self._model_name = model_name
        self._sandbox = sandbox
        self._max_turns = max_turns
        self._tool_timeout = tool_timeout
        self._max_concurrent = max_concurrent_episodes
        self.history: list[Message] = []
        self._episode: Episode | None = None
        self._turn: Turn | None = None
        self._tools: Any = None
        self._specs: list[ToolSpec] | None = None
        self._system: str | None = None
        self._holding = False
        self._scored = False
        self._messages = 0
        self._tool_calls = 0
        self._tool_errors = 0

    @property
    def scored(self) -> bool:
        """Whether the episode's run has ended and been rewarded."""
        return self._scored

    async def initial_observation(self) -> list[Message]:
        await _slots(self._max_concurrent).acquire()
        self._holding = True
        try:
            self._episode = await self._spec.start(sandbox=self._sandbox, run_group_id=self._run_group_id)
            turn = await self._episode.next_turn()
            if turn is None:
                raise EpisodeError(f"episode {self._episode.id} ended before its first turn")
            await self._open(turn)
        except BaseException:
            await self.close()  # a failed episode gives its slot back now, not when its group is cleaned up
            raise
        self._system = turn.system_prompt
        prefix = self._renderer.create_conversation_prefix_with_tools(tools=self._specs or [],
                                                                      system_prompt=self._system or "")
        self.history = [*prefix, {"role": "user", "content": turn.prompt}]
        return self.history

    async def step(self, message: Message) -> MessageStepResult:
        try:
            return await self._step(message)
        except BaseException:
            await self.close()
            raise

    async def _step(self, message: Message) -> MessageStepResult:
        self.history.append(message)
        self._messages += 1
        text = get_text_content(message)
        calls = [(call, call.id or f"call_{self._messages}_{n}", typed_arguments(_arguments(call), self._schema(call)))
                 for n, call in enumerate(message.get("tool_calls") or [])]
        # A call whose arguments aren't an object can't run, so it isn't recorded: the turn runs every recorded call.
        await self._turn.record_model_call(
            _chat_message(text, [call for call in calls if isinstance(call[2], dict)]), model=self._model_name)
        logs: dict[str, str | int | float] = {"task": self._name, "episode": self._episode.id,
                                              "assistant": _clipped(text, self.LOGGED_CHARS)}
        for n, (call, call_id, arguments) in enumerate(calls):
            if isinstance(arguments, dict):
                result = await self._run(call, call_id, arguments)
            else:
                self._tool_errors += 1
                result = _tool_message(call, call_id, "error: the arguments aren't a JSON object")
            self.history.append(result)
            logs[f"tool_call_{n}"] = (f"{call.function.name}({_clipped(json.dumps(arguments), self.LOGGED_CHARS)}) -> "
                                      f"{_clipped(str(result['content']), self.LOGGED_CHARS)}")
        unparsed = message.get("unparsed_tool_calls") or []
        for each in unparsed:
            self._tool_errors += 1
            self.history.append({"role": "user", "content": f"Your tool call couldn't be parsed ({each.error}). "
                                                            "Call the tool again, with JSON arguments."})
        if (calls or unparsed) and self._messages < self._max_turns:
            return MessageStepResult(reward=0.0, episode_done=False, next_messages=self.history, logs=logs)
        result = await self._end_turn("" if calls else text)
        result.logs = {**logs, **result.logs}
        return result

    async def close(self) -> None:
        """Abandon the episode unless its run has ended, and give back its slot. Safe to call more than once."""
        try:
            if self._episode is not None and not self._episode.finished:
                await self._episode.abandon("its rollout ended before the episode did")
        finally:
            self._release()

    async def _open(self, turn: Turn) -> None:
        """Make ``turn`` the one being played, refusing what the conversation can't show the model: skills, and tools
        other than the first turn's, which the renderer wrote at its start."""
        if turn.skills:
            raise EpisodeError(f"turn {turn.step_id!r} gives the agent skills {[s.name for s in turn.skills]}, which "
                               "no trainer shows its model yet")
        self._turn = turn
        self._tools = await turn.tools(timeout=self._tool_timeout).__aenter__()
        specs: list[ToolSpec] = [{"name": tool.name, "description": tool.description or "",
                                  "parameters": tool.inputSchema or {"type": "object", "properties": {}}}
                                 for tool in self._tools.list()]
        if self._specs is not None and specs != self._specs:
            raise EpisodeError(f"turn {turn.step_id!r} offers other tools than the episode's first turn, and the "
                               "renderer writes the tools once, at the start of the conversation")
        self._specs = specs

    def _schema(self, call: ToolCall) -> dict:
        return next((spec["parameters"] for spec in self._specs or [] if spec["name"] == call.function.name), {})

    async def _run(self, call: ToolCall, call_id: str, arguments: dict) -> Message:
        self._tool_calls += 1
        name = call.function.name
        offered = [spec["name"] for spec in self._specs or []]
        if name not in offered:
            self._tool_errors += 1
            return _tool_message(call, call_id, f"error: there is no tool {name!r}; the tools are {offered}")
        result = await self._tools.call(name, arguments)
        self._tool_errors += bool(result.isError)
        text = "".join(_text(part) for part in result.content)
        return _tool_message(call, call_id, f"error: {text}" if result.isError else text)

    async def score_cut(self) -> tuple[float, dict[str, float]]:
        """The verifiers' reward for an episode whose rollout the cookbook cut, called the moment it is cut
        (``_ScoredWhenCut``): its open turn ends with no reply, its run is scored and its slot freed. Nothing for an
        episode ``step`` already rewarded."""
        if self._scored:
            return 0.0, {"cut": 0.0}
        try:
            if self._turn is not None:
                await self._turn.end("")
            reward, metrics, _ = await self._finish()
        except BaseException:
            await self.close()
            raise
        return reward, {**metrics, "cut": 1.0}

    async def _end_turn(self, reply: str) -> MessageStepResult:
        await self._turn.end(reply)
        while (turn := await self._episode.next_turn()) is not None:
            if self._messages < self._max_turns:
                await self._open(turn)
                prompt, self._system = next_prompt(turn.prompt, turn.system_prompt, self._system)
                self.history.append({"role": "user", "content": prompt})
                return MessageStepResult(reward=0.0, episode_done=False, next_messages=self.history)
            await turn.end("")
        reward, metrics, scores = await self._finish()
        return MessageStepResult(reward=reward, episode_done=True, next_messages=self.history, metrics=metrics,
                                 logs={"scores": json.dumps(scores)})

    async def _finish(self) -> tuple[float, dict[str, float], dict[str, float]]:
        """End every turn the run has left with no reply, and reward the scores the run records."""
        while (turn := await self._episode.next_turn()) is not None:
            await turn.end("")
        scores = await self._episode.result()
        self._release()
        self._scored = True
        reward = self._reward(scores)
        if reward is None:
            raise EpisodeError(f"episode {self._episode.id} recorded no score to make a reward of: {scores}")
        metrics = {f"score/{step}": score for step, score in scores.items()}
        metrics |= {"messages": self._messages, "tool_calls": self._tool_calls, "tool_errors": self._tool_errors,
                    "cut": 0.0}
        return reward, metrics, scores

    def _release(self) -> None:
        if self._holding:
            self._holding = False
            _slots(self._max_concurrent).release()


def _arguments(call: ToolCall) -> Any:
    try:
        return json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return None


def _clipped(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}… ({len(text) - limit} more characters)"


def _chat_message(text: str, calls: list[tuple[ToolCall, str, dict]]) -> dict[str, Any]:
    """The model's message as an OpenAI chat message, with the tool calls that will run."""
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [{"id": call_id, "type": "function",
                                  "function": {"name": call.function.name, "arguments": json.dumps(arguments)}}
                                 for call, call_id, arguments in calls]
    return message


def _tool_message(call: ToolCall, call_id: str, content: str) -> Message:
    return {"role": "tool", "content": content, "tool_call_id": call_id, "name": call.function.name}


def _text(part: Any) -> str:
    kind = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
    if kind == "text":
        return part.get("text", "") if isinstance(part, dict) else part.text
    return f"[{kind} content]"


class _ScoredWhenCut(EnvFromMessageEnv):
    """The cookbook's token-level ``Env`` over an ``EpisodeEnv``, scoring a rollout the moment it cuts it.

    ``EnvFromMessageEnv`` ends a rollout itself, without calling the ``EpisodeEnv`` behind it, when a message runs past
    the token limit, the response can't be parsed, or the conversation outgrows ``max_trajectory_tokens``. The episode
    is then left with its turn open, still holding its containers and its slot under ``max_concurrent_episodes``. Left
    until its group is done, such episodes can hold every slot while their groups' other rollouts wait for one, and
    training stops. So the cut is scored here, where it happens, which frees the slot at once."""

    async def initial_observation(self):
        observation = await super().initial_observation()
        if isinstance(observation, InitialObservationOverflow):  # the prompt alone is over the token budget
            reward, metrics = await self.message_env.score_cut()
            observation = dataclasses.replace(observation, reward=observation.reward + reward,
                                              metrics={**observation.metrics, **metrics})
        return observation

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        result = await super().step(action, extra=extra)
        if result.episode_done and not self.message_env.scored:  # the cookbook ended it, not the episode
            reward, metrics = await self.message_env.score_cut()
            result = dataclasses.replace(result, reward=result.reward + reward, metrics={**result.metrics, **metrics})
        return result


class EpisodeGroupBuilder(EnvGroupBuilder):
    """``group_size`` episodes of one task, each its own run, filed in ``run_group``, which the explorer shows
    together. ``make_envs`` may be called again for a retry, and ``cleanup`` abandons every episode still under way."""

    def __init__(self, name: str, spec: EpisodeSpec, reward: Reward, settings: _Settings, group_size: int,
                 run_group: RunGroup):
        self.name = name
        self.run_group = run_group
        self.spec = spec
        self.reward = reward
        self.settings = settings
        self.group_size = group_size
        self._made: list[EpisodeEnv] = []

    async def make_envs(self) -> Sequence[Env]:
        renderer = self.settings.renderer()
        run_group_id = self.run_group.id()
        envs = []
        for _ in range(self.group_size):
            episode_env = EpisodeEnv(self.name, self.spec, renderer=renderer, reward=self.reward,
                                     model_name=self.settings.model_name, sandbox=self.settings.sandbox,
                                     max_turns=self.settings.max_turns, tool_timeout=self.settings.tool_timeout,
                                     max_concurrent_episodes=self.settings.max_concurrent_episodes,
                                     run_group_id=run_group_id)
            self._made.append(episode_env)
            envs.append(_ScoredWhenCut(renderer, episode_env, failed_parse_reward=self.settings.failed_reward,
                                       context_overflow_reward=self.settings.failed_reward,
                                       max_trajectory_tokens=self.settings.max_trajectory_tokens))
        return envs

    async def cleanup(self) -> None:
        made, self._made = self._made, []
        for result in await asyncio.gather(*(env.close() for env in made), return_exceptions=True):
            if isinstance(result, BaseException):
                logger.warning("task %s: an episode didn't close: %s", self.name, result)

    def logging_tags(self) -> list[str]:
        return [self.settings.tag]


class _Settings:
    """What every group of a dataset shares."""

    def __init__(self, *, model_name: str, renderer_name: str, sandbox: str | None, max_turns: int,
                 tool_timeout: float, max_concurrent_episodes: int, max_trajectory_tokens: int | None,
                 failed_reward: float, tag: str):
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.sandbox = sandbox
        self.max_turns = max_turns
        self.tool_timeout = tool_timeout
        self.max_concurrent_episodes = max_concurrent_episodes
        self.max_trajectory_tokens = max_trajectory_tokens
        self.failed_reward = failed_reward
        self.tag = tag

    def renderer(self) -> Renderer:
        return get_renderer(self.renderer_name, tokenizer_utils.get_tokenizer(self.model_name),
                            model_name=self.model_name)


class EpisodeDataset(RLDataset):
    """A split's tasks in batches of ``groups_per_batch`` groups, shuffled anew each of ``epochs`` passes; a group's
    episodes are filed under the run, whether they train or evaluate, and the training step that plays them."""

    def __init__(self, tasks: list[tuple[str, EpisodeSpec, Reward]], settings: _Settings, *, group_size: int,
                 groups_per_batch: int, epochs: int, seed: int, shuffle: bool, run_name: str,
                 steps: TrainingSteps, trains: bool):
        self._run_name = run_name
        self._steps = steps
        self._trains = trains
        self._tasks = tasks
        self._settings = settings
        self._group_size = group_size
        self._groups_per_batch = min(groups_per_batch, len(tasks))
        self._epochs = epochs
        self._seed = seed
        self._shuffle = shuffle
        self._per_epoch = len(tasks) // self._groups_per_batch

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        epoch, at = divmod(index, self._per_epoch)
        order = list(range(len(self._tasks)))
        if self._shuffle:
            random.Random(self._seed + epoch).shuffle(order)
        picked = order[at * self._groups_per_batch:(at + 1) * self._groups_per_batch]
        group = RunGroup(self._run_name, self._steps, self._steps.trained(index) if self._trains else None)
        return [EpisodeGroupBuilder(name, spec, reward, self._settings, self._group_size, group)
                for name, spec, reward in (self._tasks[i] for i in picked)]

    def __len__(self) -> int:
        return self._per_epoch * self._epochs


@chz.chz
class AgentEnvDatasetBuilder(RLDatasetBuilder):
    """The tasks of the bundle folder ``bundle`` for ``rl.train``, prepared as ``agent-env run`` prepares them, their
    episodes filed in run groups named ``<run_name>/train|eval/step-<n>``. ``train_split`` and ``test_split`` are
    ``all`` or an eval of the bundle. ``reward`` is the step id whose score is the reward, needed
    when a task records more than one. ``sandbox`` is where episodes deploy, as ``agent-env run --sandbox`` sets it,
    and ``max_concurrent_episodes`` bounds the runs under way at once. A rollout the cookbook ends itself is scored
    as it is cut, and given ``failed_reward`` on top."""

    bundle: str
    run_name: str
    model_name: str
    renderer_name: str
    group_size: int
    groups_per_batch: int
    train_split: str = "all"
    test_split: str | None = None
    test_group_size: int = 1
    reward: str | None = None
    sandbox: str | None = None
    max_turns: int = 30
    tool_timeout: float = 60.0
    turn_timeout: float | None = None
    max_concurrent_episodes: int = 8
    max_trajectory_tokens: int | None = None
    failed_reward: float = 0.0
    epochs: int = 1
    seed: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        wanted = [split for split in (self.train_split, self.test_split) if split is not None]
        splits = await asyncio.to_thread(_splits, Path(self.bundle).expanduser(), wanted, self.sandbox)
        settings = _Settings(model_name=self.model_name, renderer_name=self.renderer_name, sandbox=self.sandbox,
                             max_turns=self.max_turns, tool_timeout=self.tool_timeout,
                             max_concurrent_episodes=self.max_concurrent_episodes,
                             max_trajectory_tokens=self.max_trajectory_tokens, failed_reward=self.failed_reward,
                             tag=Path(self.bundle).name)
        steps = TrainingSteps()
        train = EpisodeDataset(playable(splits[self.train_split], self.reward, self.turn_timeout), settings,
                               group_size=self.group_size, groups_per_batch=self.groups_per_batch, epochs=self.epochs,
                               seed=self.seed, shuffle=True, run_name=self.run_name, steps=steps, trains=True)
        test = None
        if self.test_split is not None:
            tasks = playable(splits[self.test_split], self.reward, self.turn_timeout)
            test = EpisodeDataset(tasks, settings, group_size=self.test_group_size, groups_per_batch=len(tasks),
                                  epochs=1, seed=self.seed, shuffle=False, run_name=self.run_name,
                                  steps=steps, trains=False)
        return train, test


def _splits(bundle: Path, splits: list[str], sandbox: str | None) -> dict[str, list[tuple[str, Task]]]:
    """Each split's tasks with their names, ``all`` or an eval's, prepared as ``agent-env run`` prepares them before
    its first task, without the agent the trainer plays (``played``), as the OpenEnv server prepares a bundle."""
    with namespace_routing():
        every = [entry.name for entry in parse_bundle(bundle).entries if entry.kind is BundleKind.TASK]
        prepared = prepare_bundle(bundle, tasks=every, evals=[split for split in splits if split != "all"],
                                  sandbox=sandbox, on_progress=logger.info, played=True)
    tasks = [(entry.name, task) for entry, task in prepared.tasks]
    by_id = {task.id: (name, task) for name, task in tasks}
    found = {"all": tasks, **{entry.entry.name: [by_id[ref.id] for ref in entry.references]
                              for entry in prepared.plan.evals}}
    return {split: found[split] for split in splits}


def _gives_skills(step: Any, agents: Sequence[str]) -> bool:
    return getattr(step, "agent_name", None) in agents and (
        step.type == "add_skills" or step.type == "deploy_agent" and bool(step.skills))


def playable(tasks: list[tuple[str, Task]], reward: str | None,
              turn_timeout: float | None) -> list[tuple[str, EpisodeSpec, Reward]]:
    """Each task that can run as an episode, with its reward; the rest are logged and left out. Raises ValueError when
    a task's scores make no reward, since a rollout must be rewarded, or when no task is left.

    A task that gives a played agent skills is left out, here rather than when its turn opens: no trainer shows its
    model skills yet, and a rollout refused at that turn would be retried and refused again."""
    episodes, refused, unscorable = [], {}, {}
    for name, task in tasks:
        try:
            spec = EpisodeSpec.from_task(task, turn_timeout=turn_timeout)
        except EpisodeError as e:
            refused[name] = str(e)
            continue
        if skilled := [step.id for step in task.steps if _gives_skills(step, spec.trainer_agents)]:
            refused[name] = (f"task {task.id!r} gives the agent it plays skills ({skilled}), which no trainer shows "
                             "its model yet")
            continue
        task_reward, problem = reward_of(task, reward)
        if problem or not any(getattr(step, "verifier_id", None) for step in task.steps):
            unscorable[name] = problem or f"task {task.id!r} records no score"
            continue
        episodes.append((name, spec, task_reward))
    for why in refused.values():
        logger.warning("left out: %s", why)
    if unscorable:
        raise ValueError("reward= can't make a reward of every task's scores:\n" + "\n".join(unscorable.values()))
    if not episodes:
        raise ValueError(f"no task can be played as an episode: {refused}")
    return episodes
