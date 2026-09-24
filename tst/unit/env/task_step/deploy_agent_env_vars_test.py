"""Unit tests for DeployAgentTaskStep env_vars and cpu/memory resources."""

from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep


class TestDeployAgentEnvVars:
    def test_env_vars_roundtrip(self):
        """env_vars should survive to_dict -> from_dict roundtrip."""
        env_vars = {"ENABLED_TOOLS": '["tool1","tool2"]', "FOO": "bar"}
        step = DeployAgentTaskStep(
            id="test-env-vars",
            version=None,
            env_ids=["env-1"],
            env_vars=env_vars,
        )

        d = step.to_dict()
        assert d["env_vars"] == env_vars

        restored = DeployAgentTaskStep.from_dict(d)
        assert restored.env_vars == env_vars
        assert restored.env_ids == ["env-1"]

    def test_env_vars_default_empty(self):
        """env_vars should default to {} when not provided."""
        step = DeployAgentTaskStep(
            id="test-no-env-vars",
            version=None,
        )
        assert step.env_vars == {}

    def test_from_dict_without_env_vars(self):
        """from_dict should handle missing env_vars (backward compat)."""
        data = {
            "id": "test-compat",
            "type": "deploy_agent",
            "version": 1,
            "env_ids": ["env-1"],
            "artifact_id": "art-1",
            "agent_name": "default-agent",
        }
        step = DeployAgentTaskStep.from_dict(data)
        assert step.env_vars == {}

    def test_env_vars_in_to_dict(self):
        """to_dict should always include env_vars key."""
        step = DeployAgentTaskStep(id="test", version=None)
        d = step.to_dict()
        assert "env_vars" in d
        assert d["env_vars"] == {}


class TestDeployAgentResources:
    def test_cpu_memory_roundtrip(self):
        step = DeployAgentTaskStep(
            id="test-resources",
            version=None,
            cpu=2.0,
            memory_mb=4096,
        )

        d = step.to_dict()
        assert d["cpu"] == 2.0
        assert d["memory_mb"] == 4096

        restored = DeployAgentTaskStep.from_dict(d)
        assert restored.cpu == 2.0
        assert restored.memory_mb == 4096

    def test_defaults_to_none(self):
        step = DeployAgentTaskStep(id="test-defaults", version=None)
        assert step.cpu is None
        assert step.memory_mb is None

    def test_from_dict_backward_compat(self):
        """Legacy step definitions without cpu/memory_mb should still load."""
        data = {
            "id": "test-compat",
            "type": "deploy_agent",
            "version": 1,
            "env_ids": ["env-1"],
            "agent_name": "default-agent",
        }
        step = DeployAgentTaskStep.from_dict(data)
        assert step.cpu is None
        assert step.memory_mb is None

    def test_to_dict_always_includes_keys(self):
        step = DeployAgentTaskStep(id="test", version=None)
        d = step.to_dict()
        assert "cpu" in d
        assert "memory_mb" in d
        assert d["cpu"] is None
        assert d["memory_mb"] is None

    def test_cpu_only_override(self):
        step = DeployAgentTaskStep(id="test", version=None, cpu=8.0)
        restored = DeployAgentTaskStep.from_dict(step.to_dict())
        assert restored.cpu == 8.0
        assert restored.memory_mb is None

    def test_memory_only_override(self):
        step = DeployAgentTaskStep(id="test", version=None, memory_mb=16384)
        restored = DeployAgentTaskStep.from_dict(step.to_dict())
        assert restored.cpu is None
        assert restored.memory_mb == 16384
