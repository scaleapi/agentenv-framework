"""SkyRL's trainer with the env registered and AgentEnvGenerator generating: ``python -m agent_env.training.skyrl.train``,
or SkyRL's ``main_base`` with ``trainer.override_entrypoint=agent_env.training.skyrl.train``."""

from __future__ import annotations

import os
import sys
from typing import Any

try:
    import ray
    from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.entrypoints.main_base import BasePPOExp
    from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
    from skyrl.train.utils import initialize_ray, validate_cfg
    from skyrl_gym.envs.registration import register as register_env
    from skyrl_gym.envs.registration import registry
except ModuleNotFoundError as exc:
    if exc.name is None or exc.name.split(".")[0] not in ("skyrl", "skyrl_gym", "ray"):
        raise
    raise ImportError("agent_env.training.skyrl.train runs in SkyRL's environment: install "
                      "'agentenv-framework[skyrl]' into it") from exc

from agent_env.training._openenv import token_in
from agent_env.training.skyrl.generator import AgentEnvGenerator
from agent_env.training.skyrl.prepare import AGENT, token_vars


def register() -> None:
    """Registers ``AgentEnvTextEnv`` with skyrl-gym as ``agentenv``, the ``env_class`` of every row, in this process:
    a trainer of one's own calls it where SkyRL builds its envs, and generates with ``AgentEnvGenerator``."""
    if AGENT not in registry:
        register_env(id=AGENT, entry_point="agent_env.training.skyrl.env:AgentEnvTextEnv")


class AgentEnvExp(BasePPOExp):
    """SkyRL's experiment with ``AgentEnvGenerator``, and its fully async trainer when ``trainer.fully_async.enabled``
    is set."""

    def get_generator(self, cfg: SkyRLTrainConfig, tokenizer: Any, inference_engine_client: Any) -> AgentEnvGenerator:
        if cfg.generator.vision_language_generator:
            raise ValueError("agentenv rows are text, so set generator.vision_language_generator=false")
        return AgentEnvGenerator(generator_cfg=cfg.generator, skyrl_gym_cfg=cfg.environment.skyrl_gym,
                                 inference_engine_client=inference_engine_client, tokenizer=tokenizer,
                                 policy_model_name=resolve_policy_model_name(cfg))

    def get_trainer(self, cfg: SkyRLTrainConfig, **parts: Any) -> Any:
        """SkyRL's trainer, or its fully async one, built from the parts SkyRL's experiment hands its own."""
        if not cfg.trainer.fully_async.enabled:
            return super().get_trainer(cfg=cfg, **parts)
        return FullyAsyncRayPPOTrainer(cfg=cfg, **parts)


@ray.remote(num_cpus=1)
def _run(cfg: SkyRLTrainConfig, environ: dict[str, str]) -> None:
    os.environ.update(environ)
    register()
    AgentEnvExp(cfg).run()


def main() -> None:
    """Reads the tokens the rows' servers ask for here, so a trainer on any node of the Ray cluster has them."""
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    environ = {var: token_in(var) for var in token_vars([*cfg.data.train_data, *cfg.data.val_data])}
    initialize_ray(cfg)
    ray.get(_run.remote(cfg, environ))


if __name__ == "__main__":
    main()
