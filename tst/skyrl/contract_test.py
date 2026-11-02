"""The parts of SkyRL that ``AgentEnvGenerator`` and ``AgentEnvExp`` build on, so a SkyRL that changes one fails here,
naming it, before a rollout does."""

import dataclasses
import inspect
import typing

import pytest

from tst.util.capabilities import SKYRL, missing_capability_reason

train = pytest.importorskip("agent_env.training.skyrl.train", reason=missing_capability_reason(SKYRL),
                            exc_type=ImportError)
base = pytest.importorskip("skyrl.train.generators.base", reason=missing_capability_reason(SKYRL))
gym_generator = pytest.importorskip("skyrl.train.generators.skyrl_gym_generator",
                                    reason=missing_capability_reason(SKYRL))
config = pytest.importorskip("skyrl.train.config", reason=missing_capability_reason(SKYRL))
trainer = pytest.importorskip("skyrl.train.trainer", reason=missing_capability_reason(SKYRL))


def test_the_agent_loop_takes_a_rows_prompt_env_class_and_extras_first():
    parameters = list(inspect.signature(gym_generator.SkyRLGymGenerator.agent_loop).parameters)

    assert parameters[1:4] == ["prompt", "env_class", "env_extras"]


def test_the_generator_finds_env_metrics_loss_masks_rewards_and_phases_where_it_reads_them():
    assert "env_metrics" in _fields(gym_generator.TrajectoryOutput)
    assert "step_outputs" in _fields(gym_generator.StepWiseOutput)
    assert {"env_metrics", "loss_masks", "rewards", "trajectory_ids"} <= set(typing.get_type_hints(base.GeneratorOutput))
    assert {"batch_metadata", "trajectory_ids"} <= set(typing.get_type_hints(base.GeneratorInput))
    assert "training_phase" in _fields(base.BatchMetadata)


def test_the_experiment_builds_its_generator_from_what_skyrls_own_experiment_passes(monkeypatch):
    built = []

    def recorded(self, *args, **kwargs):
        built.append((args, kwargs))
        raise _Built

    monkeypatch.setattr(gym_generator.SkyRLGymGenerator, "__init__", recorded)
    cfg, experiment = config.SkyRLTrainConfig(), object.__new__(train.AgentEnvExp)
    for build in (train.BasePPOExp.get_generator, train.AgentEnvExp.get_generator):
        with pytest.raises(_Built):
            build(experiment, cfg, "tokenizer", "engine")

    assert built[0] == built[1]


def test_skyrls_trainers_take_the_parts_its_experiment_hands_get_trainer():
    parts = set(inspect.signature(train.BasePPOExp.get_trainer).parameters) - {"self"}

    assert parts <= set(inspect.signature(trainer.RayPPOTrainer.__init__).parameters)
    assert issubclass(train.FullyAsyncRayPPOTrainer, trainer.RayPPOTrainer)


class _Built(Exception):
    pass


def _fields(cls):
    return {field.name for field in dataclasses.fields(cls)}
