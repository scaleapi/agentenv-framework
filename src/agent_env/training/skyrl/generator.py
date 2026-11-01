"""SkyRL's generator for the rows ``python -m agent_env.training.skyrl`` writes. SkyRL can't leave one trajectory out
of a step, so a rollout the env fails is played again, and one it fails every time leaves its whole group out."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

try:
    from skyrl.train.generators.base import TRAINING_PHASE_EVAL, GeneratorInput, GeneratorOutput
    from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator, StepWiseOutput, TrajectoryOutput
except ModuleNotFoundError as exc:
    if exc.name is None or exc.name.split(".")[0] != "skyrl":
        raise
    raise ImportError("agent_env.training.skyrl.generator runs in SkyRL's trainer: install "
                      "'agentenv-framework[skyrl]' into SkyRL's environment") from exc

from agent_env.training._openenv import attempt_numbers
from agent_env.training.skyrl.prepare import AGENT

logger = logging.getLogger(__name__)


class AgentEnvGenerator(SkyRLGymGenerator):
    """At most ``max_env_workers`` agentenv rollouts play at once: a rollout waiting for room on the server holds the
    thread it was called from, so more would starve the rollouts holding sessions of the threads that end them."""

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
        for attempt in attempt_numbers(server["attempts"], server["reset_timeout"]):
            output = await super().agent_loop(prompt, env_class, dict(env_extras), *args, **kwargs)
            metrics = (output.step_outputs[-1] if isinstance(output, StepWiseOutput) else output).env_metrics
            metrics["attempts"] = attempt
            if not metrics.get("failed"):
                return output
        logger.warning("task %s of split %s: the env failed it on %d attempt%s, so its group is left out of training",
                       episode["task_id"], episode["split"], attempt, "s" * (attempt > 1))
        return output

    async def generate(self, input_batch: GeneratorInput, *args: Any, **kwargs: Any) -> GeneratorOutput:
        output = await super().generate(input_batch, *args, **kwargs)
        failed = [row for row, metrics in enumerate(output["env_metrics"]) if (metrics or {}).get("failed")]
        metadata = input_batch.get("batch_metadata")
        ids = output.get("trajectory_ids") or input_batch.get("trajectory_ids")
        if not failed or not ids or getattr(metadata, "training_phase", None) == TRAINING_PHASE_EVAL:
            return output
        groups = {ids[row].instance_id for row in failed}
        for row, trajectory in enumerate(ids):
            if trajectory.instance_id in groups:
                output["loss_masks"][row] = [0] * len(output["loss_masks"][row])
                reward = output["rewards"][row]
                output["rewards"][row] = [0.0] * len(reward) if isinstance(reward, list) else 0.0
        return output
