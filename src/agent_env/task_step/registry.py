"""TaskStep registry for type-based deserialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_env.plugins import _registration

if TYPE_CHECKING:
    from agent_env.task_step.task_step import TaskStep
    from agent_env.config.runtime import Config


def _get_type(cls: type["TaskStep"]) -> str:
    return cls.type




def get_task_step_registry() -> dict[str, type["TaskStep"]]:
    from agent_env.config import runtime

    return runtime.get_config().task_step_registry()


def _builtin_registry() -> dict[str, type["TaskStep"]]:
    from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep
    from agent_env.task_step.task_steps.mcp_cli_builder.build_mcp_cli import BuildMcpCliTaskStep
    from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
    from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
    from agent_env.task_step.task_steps.deploy_human_agent import DeployHumanAgentTaskStep
    from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
    from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
    from agent_env.task_step.task_steps.verifiers.env_outcome_verifier import EnvOutcomeVerifierTaskStep
    from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
    from agent_env.task_step.task_steps.modify_env_tool_access import ModifyEnvToolAccessStep
    from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
    from agent_env.task_step.task_steps.register_env_triggers import RegisterEnvTriggersStep
    from agent_env.task_step.task_steps.register_agent_triggers import RegisterAgentTriggersStep
    from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep
    from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep
    from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
    from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import RunContainerUnitTestsVerifierTaskStep
    from agent_env.task_step.task_steps.run_docker_container import RunDockerContainerTaskStep
    from agent_env.task_step.task_steps.mcp_env_validator.verify_mcp_tool_schema import VerifyMCPToolSchemaTaskStep
    from agent_env.task_step.task_steps.mcp_env_validator.verify_spec_conformance import VerifySpecConformanceTaskStep
    from agent_env.task_step.task_steps.mcp_env_validator.verify_mcp_env_assessment import VerifyMCPEnvAssessmentStep
    from agent_env.task_step.task_steps.mcp_env_validator.validation_gate_aggregator import ValidationGateAggregatorStep
    from agent_env.task_step.task_steps.multienv_validator.verify_universe_roundtrip import VerifyUniverseLoadExportRoundtripStep
    from agent_env.task_step.task_steps.multienv_validator.combine_universe_verdicts import CombineUniverseVerdictsStep
    from agent_env.task_step.task_steps.env_card_validator.verify_env_card import VerifyEnvironmentCardStep
    from agent_env.task_step.task_steps.env_card_validator.verify_env_core_protocol import VerifyCoreEnvironmentProtocolStep
    from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_agent_card import VerifyA2AAgentCardStep
    from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_agent_config_identity import VerifyA2AAgentConfigIdentityStep
    from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_agent_mcp import VerifyA2AAgentMCPStep
    from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_core_protocol import VerifyCoreA2AProtocolStep
    from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_modalities import VerifyA2AModalitiesStep
    from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_peer_agents import VerifyA2APeerAgentsStep
    from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_install import VerifyA2AInstallStep
    from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_skill_config import VerifyA2ASkillConfigStep
    from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_snapshot import VerifyA2ASnapshotStep
    from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_trajectory import VerifyA2ATrajectoryStep
    from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_role import VerifyA2ARoleStep
    from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_system_prompt import VerifyA2ASystemPromptStep
    from agent_env.task_step.task_steps.add_skills import AddSkillsTaskStep
    from agent_env.task_step.task_steps.apply_server_config import ApplyServerConfigStep
    from agent_env.task_step.task_steps.verifiers.agent_prompt_response_verifier import AgentPromptResponseVerifierTaskStep
    from agent_env.task_step.task_steps.snapshot_agent_state import SnapshotAgentStateTaskStep
    from agent_env.task_step.task_steps.snapshot_env import SnapshotEnvTaskStep
    from agent_env.task_step.task_steps.peer_agents import PeerAgentsTaskStep
    from agent_env.task_step.task_steps.verifiers.verify_sandbox import VerifySandboxTaskStep
    from agent_env.task_step.task_steps.verifiers.aggregate_verifiers import AggregateVerifiersTaskStep
    from agent_env.task_step.task_steps.run_code import RunCodeTaskStep
    from agent_env.task_step.task_steps.review import ReviewTaskStep

    return {
        _get_type(ReviewTaskStep): ReviewTaskStep,
        _get_type(BuildMcpCliTaskStep): BuildMcpCliTaskStep,
        _get_type(CollectArtifactsTaskStep): CollectArtifactsTaskStep,
        _get_type(DeployAgentTaskStep): DeployAgentTaskStep,
        _get_type(DeployEnvTaskStep): DeployEnvTaskStep,
        _get_type(DeployHumanAgentTaskStep): DeployHumanAgentTaskStep,
        _get_type(DeploySandboxTaskStep): DeploySandboxTaskStep,
        _get_type(ResetEnvTaskStep): ResetEnvTaskStep,
        _get_type(EnvOutcomeVerifierTaskStep): EnvOutcomeVerifierTaskStep,
        _get_type(LoadArtifactTaskStep): LoadArtifactTaskStep,
        _get_type(ModifyEnvToolAccessStep): ModifyEnvToolAccessStep,
        _get_type(PromptAgentTaskStep): PromptAgentTaskStep,
        _get_type(RegisterEnvTriggersStep): RegisterEnvTriggersStep,
        _get_type(RegisterAgentTriggersStep): RegisterAgentTriggersStep,
        _get_type(SyncEnvClockTaskStep): SyncEnvClockTaskStep,
        _get_type(InstallAgentTaskStep): InstallAgentTaskStep,
        _get_type(RubricsVerifierTaskStep): RubricsVerifierTaskStep,
        _get_type(RunContainerUnitTestsVerifierTaskStep): RunContainerUnitTestsVerifierTaskStep,
        _get_type(RunDockerContainerTaskStep): RunDockerContainerTaskStep,
        _get_type(VerifyMCPToolSchemaTaskStep): VerifyMCPToolSchemaTaskStep,
        _get_type(VerifySpecConformanceTaskStep): VerifySpecConformanceTaskStep,
        _get_type(VerifyMCPEnvAssessmentStep): VerifyMCPEnvAssessmentStep,
        _get_type(ValidationGateAggregatorStep): ValidationGateAggregatorStep,
        _get_type(VerifyUniverseLoadExportRoundtripStep): VerifyUniverseLoadExportRoundtripStep,
        _get_type(CombineUniverseVerdictsStep): CombineUniverseVerdictsStep,
        _get_type(VerifyEnvironmentCardStep): VerifyEnvironmentCardStep,
        _get_type(VerifyCoreEnvironmentProtocolStep): VerifyCoreEnvironmentProtocolStep,
        _get_type(VerifyA2AAgentCardStep): VerifyA2AAgentCardStep,
        _get_type(VerifyA2AAgentConfigIdentityStep): VerifyA2AAgentConfigIdentityStep,
        _get_type(VerifyA2AAgentMCPStep): VerifyA2AAgentMCPStep,
        _get_type(VerifyA2ATrajectoryStep): VerifyA2ATrajectoryStep,
        _get_type(VerifyA2AInstallStep): VerifyA2AInstallStep,
        _get_type(VerifyA2ASkillConfigStep): VerifyA2ASkillConfigStep,
        _get_type(AddSkillsTaskStep): AddSkillsTaskStep,
        _get_type(ApplyServerConfigStep): ApplyServerConfigStep,
        _get_type(AgentPromptResponseVerifierTaskStep): AgentPromptResponseVerifierTaskStep,
        _get_type(VerifyCoreA2AProtocolStep): VerifyCoreA2AProtocolStep,
        _get_type(SnapshotAgentStateTaskStep): SnapshotAgentStateTaskStep,
        _get_type(SnapshotEnvTaskStep): SnapshotEnvTaskStep,
        _get_type(RunCodeTaskStep): RunCodeTaskStep,
        _get_type(PeerAgentsTaskStep): PeerAgentsTaskStep,
        _get_type(VerifyA2ASnapshotStep): VerifyA2ASnapshotStep,
        _get_type(VerifyA2APeerAgentsStep): VerifyA2APeerAgentsStep,
        _get_type(VerifyA2AModalitiesStep): VerifyA2AModalitiesStep,
        _get_type(VerifySandboxTaskStep): VerifySandboxTaskStep,
        _get_type(AggregateVerifiersTaskStep): AggregateVerifiersTaskStep,
        _get_type(VerifyA2ARoleStep): VerifyA2ARoleStep,
        _get_type(VerifyA2ASystemPromptStep): VerifyA2ASystemPromptStep,
    }


def _build_registry(source: Config | None = None) -> dict[str, type["TaskStep"]]:
    """The built-in task steps, then ``agent_env.task_steps`` plugins, then ``[task_steps] impls``
    in ``source``."""
    from agent_env.task_step.task_step import TaskStep

    registry = _builtin_registry()
    validate = _registration.typed_validator(TaskStep, builtins=frozenset(registry))
    from_plugins = _registration.merge(registry, _registration.TASK_STEPS, validate, source=source)
    _merge_config_toml_steps(registry, source=source, from_plugins=from_plugins)
    return registry


def _merge_config_toml_steps(
    registry: dict[str, type["TaskStep"]],
    *,
    source: Config | None = None,
    from_plugins: _registration.Registrations | None = None,
) -> None:
    from agent_env.config import runtime
    from agent_env.config import ConfigError, load_impl
    from agent_env.task_step.task_step import TaskStep

    section = (source or runtime.get_config()).section("task_steps")
    impls = section.get("impls", [])
    if not isinstance(impls, list):
        raise ConfigError(
            f"[task_steps] impls must be a list of 'module:Class' strings, got {type(impls).__name__}"
        )
    from_plugins = from_plugins if from_plugins is not None else _registration.Registrations.empty()
    for impl in impls:
        if not isinstance(impl, str):
            raise ConfigError(
                f"[task_steps] impl must be a 'module:Class' string, got {type(impl).__name__}: {impl!r}"
            )
        cls = load_impl(impl, TaskStep)
        if cls.type == TaskStep.type:
            raise ConfigError(
                f"[task_steps] impl {impl!r} does not define its own 'type' "
                f"(inherits the base default {TaskStep.type!r}); set a unique 'type' ClassVar"
            )
        if not from_plugins.release(cls.type, f"[task_steps] impl {impl!r}", cls) and cls.type in registry:
            raise ConfigError(
                f"[task_steps] impl {impl!r} type {cls.type!r} is already registered "
                f"(conflicts with a built-in or another custom step)"
            )
        registry[cls.type] = cls
