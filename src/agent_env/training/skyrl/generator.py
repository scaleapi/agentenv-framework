"""SkyRL's generator for the rows ``python -m agent_env.training.skyrl`` writes, which plays a rollout the env fails
again and, when every attempt fails, leaves its group out of training: SkyRL fails a whole batch on an env's exception,
and has no way to leave out one trajectory."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from typing import Any

try:
    from skyrl.train.generators.base import TRAINING_PHASE_EVAL, GeneratorInput, GeneratorOutput
    from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator, StepWiseOutput, TrajectoryOutput
except ModuleNotFoundError as exc:
    if exc.name is None or exc.name.split(".")[0] != "skyrl":
        raise
    raise ImportError("agent_env.training.skyrl.generator runs in SkyRL's trainer: install "
                      "'agentenv-framework[skyrl]' into SkyRL's environment") from exc

from agent_env.training.skyrl.env import EpisodeFailed
from agent_env.training.skyrl.prepare import AGENT

logger = logging.getLogger(__name__)


class AgentEnvGenerator(SkyRLGymGenerator):
    """SkyRL's ``SkyRLGymGenerator`` that plays a rollout of an ``agentenv`` row the env fails (EpisodeFailed) again
    from the start on a new episode, a fresh sample of the same policy: up to the row's ``attempts`` times, but not
    once its ``reset_timeout`` has passed since the rollout began. A rollout that fails every attempt, in training, has
    its whole group's loss mask and rewards zeroed, as SkyRL's Harbor generator does with a failed trial, so the group
    is out of the gradient and of its own baseline; in an eval it scores 0. Each rollout's env metrics hold its
    attempts and whether it ``failed``. At most ``environment.skyrl_gym.max_env_workers`` rollouts play at once, one
    per thread SkyRL calls envs from: a rollout waiting for room on the server holds its thread, so more rollouts than
    threads would starve the ones holding sessions of the threads that end them.

    It leans on SkyRL internals, which CI checks on SkyRL 0.4.0 and main: ``agent_loop`` builds a new env per call and
    raises what the env raises, its output's ``env_metrics`` reach ``generate``'s, and ``generate``'s ``loss_masks``,
    ``rewards`` and ``env_metrics`` are a list per trajectory, or per step, as its ``trajectory_ids`` or the input
    batch's are."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.batched:
            raise ValueError("agentenv rows play an episode over several turns, which SkyRL's batched generation "
                             "doesn't: set generator.batched=false")
        if self.skyrl_gym_cfg.max_env_workers < 1:
            raise ValueError("agentenv rows' envs wait on their server, which SkyRL's env threads let them do without "
                             "stalling generation: set environment.skyrl_gym.max_env_workers to at least 1")
        self._playing = asyncio.Semaphore(self.skyrl_gym_cfg.max_env_workers)

    async def agent_loop(self, prompt: Any, env_class: str, env_extras: dict[str, Any], *args: Any,
                         **kwargs: Any) -> TrajectoryOutput | StepWiseOutput:
        if env_class != AGENT:
            return await super().agent_loop(prompt, env_class, env_extras, *args, **kwargs)
        async with self._playing:
            return await self._played(prompt, env_class, env_extras, *args, **kwargs)

    async def _played(self, prompt: Any, env_class: str, env_extras: dict[str, Any], *args: Any,
                      **kwargs: Any) -> TrajectoryOutput | StepWiseOutput:
        server, episode = env_extras["server"], env_extras["episode"]
        began = time.monotonic()
        for attempt in range(1, server["attempts"] + 1):
            try:
                output = await super().agent_loop(prompt, env_class, dict(env_extras), *args, **kwargs)
            except EpisodeFailed as e:
                failure = e
                logger.warning("attempt %d of %d: %s", attempt, server["attempts"], e)
                if time.monotonic() - began >= server["reset_timeout"]:
                    break
                continue
            last = output.step_outputs[-1] if isinstance(output, StepWiseOutput) else output
            last.env_metrics.update(attempts=attempt, failed=0)
            return output
        logger.warning("task %s of split %s: the env failed it on %d attempt%s, so its group is left out of training: "
                       "%s", episode["task_id"], episode["split"], attempt, "s" * (attempt > 1), failure)
        return self._left_out(attempt, time.monotonic() - began, kwargs.get("sampling_params"))

    async def generate(self, input_batch: GeneratorInput, *args: Any, **kwargs: Any) -> GeneratorOutput:
        output = await super().generate(input_batch, *args, **kwargs)
        failed = [row for row, metrics in enumerate(output["env_metrics"]) if (metrics or {}).get("failed")]
        metadata = input_batch.get("batch_metadata")
        ids = output.get("trajectory_ids") or input_batch.get("trajectory_ids")
        if failed and ids and getattr(metadata, "training_phase", None) != TRAINING_PHASE_EVAL:
            groups = {ids[row].instance_id for row in failed}
            failed = [row for row, trajectory in enumerate(ids) if trajectory.instance_id in groups]
        for row in failed:
            output["loss_masks"][row] = [0] * len(output["loss_masks"][row])
            reward = output["rewards"][row]
            output["rewards"][row] = [0.0] * len(reward) if isinstance(reward, list) else 0.0
        return output

    def _left_out(self, attempts: int, seconds: float,
                  sampling_params: dict[str, Any] | None) -> TrajectoryOutput | StepWiseOutput:
        """A placeholder for a rollout every attempt of which the env failed: one token, masked, rewarded 0."""
        sampling = sampling_params if sampling_params is not None else asdict(self.generator_cfg.sampling_params)
        token = self.tokenizer.eos_token_id
        output = TrajectoryOutput(response_ids=[token], reward=[0.0], stop_reason="error", loss_mask=[0],
                                  prompt_ids=[token],
                                  rollout_logprobs=[0.0] if sampling.get("logprobs") is not None else None,
                                  env_metrics={"attempts": attempts, "failed": 1}, e2e_time=seconds,
                                  time_splits={"llm": 0.0, "env": 0.0})
        return StepWiseOutput(step_outputs=[output]) if self.generator_cfg.step_wise_trajectories else output
