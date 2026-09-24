"""Unit tests for DeployEnvTaskStep cpu / memory_mb resource overrides."""

from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep


class TestDeployEnvResources:
    def test_cpu_memory_roundtrip(self):
        step = DeployEnvTaskStep(
            id="test-resources",
            version=None,
            env_id="env-1",
            cpu=2.0,
            memory_mb=4096,
        )

        d = step.to_dict()
        assert d["cpu"] == 2.0
        assert d["memory_mb"] == 4096

        restored = DeployEnvTaskStep.from_dict(d)
        assert restored.cpu == 2.0
        assert restored.memory_mb == 4096
        assert restored.env_id == "env-1"

    def test_defaults_to_none(self):
        step = DeployEnvTaskStep(id="test-defaults", version=None, env_id="env-1")
        assert step.cpu is None
        assert step.memory_mb is None

    def test_from_dict_backward_compat(self):
        """Legacy task definitions without cpu/memory_mb should still load."""
        data = {
            "id": "test-compat",
            "type": "deploy_env",
            "version": 1,
            "env_id": "env-1",
        }
        step = DeployEnvTaskStep.from_dict(data)
        assert step.cpu is None
        assert step.memory_mb is None

    def test_to_dict_always_includes_keys(self):
        step = DeployEnvTaskStep(id="test", version=None, env_id="env-1")
        d = step.to_dict()
        assert "cpu" in d
        assert "memory_mb" in d
        assert d["cpu"] is None
        assert d["memory_mb"] is None

    def test_cpu_only_override(self):
        step = DeployEnvTaskStep(id="test", version=None, env_id="env-1", cpu=8.0)
        restored = DeployEnvTaskStep.from_dict(step.to_dict())
        assert restored.cpu == 8.0
        assert restored.memory_mb is None

    def test_memory_only_override(self):
        step = DeployEnvTaskStep(id="test", version=None, env_id="env-1", memory_mb=16384)
        restored = DeployEnvTaskStep.from_dict(step.to_dict())
        assert restored.cpu is None
        assert restored.memory_mb == 16384
