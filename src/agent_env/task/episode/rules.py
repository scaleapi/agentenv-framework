"""Which steps a task may have when some of its agents are played from outside AgentEnv, and why the others are refused.

An outside agent is played through a proxy agent deployed in its place (the ``agents`` bundle's ``proxy``), so steps
that act on the agent, or in its sandbox, act on the proxy. Every registered step type is either checked
here (``CHECKS``) or listed as untouched by an outside agent (``UNAFFECTED``). A test holds the registry to the two, so
a new step type is classified before an episode runs it. Plugin step types aren't inspected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_env.task_step.task_step import TaskStep
    from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep
    from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
    from agent_env.task_step.task_steps.peer_agents import PeerAgentsTaskStep
    from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
    from agent_env.task_step.task_steps.register_env_triggers import RegisterEnvTriggersStep
    from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep


@dataclass(frozen=True)
class Outside:
    """The agents played from outside, and their prompts by step id, each with its prompt id."""

    agents: frozenset[str]
    prompts: dict[str, str]


def problems(steps: list[TaskStep], outside: list[str]) -> list[str]:
    """Why each outside agent, and each step, can't run with those agents played from outside; empty when it can."""
    found = []
    for name in outside:
        deploys = [step for step in steps if step.type in DEPLOYS and step.agent_name == name]
        if len(deploys) != 1:
            found.append(f"agent {name!r} is played from outside, so it needs exactly one deploy_agent or "
                         f"install_agent step, not {len(deploys)}")
    scope = Outside(frozenset(outside), {
        step.id: step.prompt_id for step in steps if step.type == "prompt_agent" and step.agent_name in outside
    })
    for step in steps:
        why = ["retry_config can replay an outside turn"] if step.retry_config is not None else []
        if check := CHECKS.get(step.type):
            why += check(step, scope)
        found.extend(f"step {step.id!r}: {reason}" for reason in why)
    return found


def works_in_sandbox(steps: list[TaskStep], agent: str) -> bool:
    """Whether ``steps`` use ``agent``'s sandbox, so its proxy is deployed there rather than served in this
    process."""
    return any(step.agent_name == agent and (
        step.type in ("install_agent", "load_artifact", "run_code", "add_skills")
        or step.type == "deploy_agent" and step.skills
        or step.type == "verify_sandbox" and not step.sandbox_name
        or step.type == "collect_artifacts" and not (step.env_id or step.container_name or step.sandbox_name)
    ) for step in steps if getattr(step, "agent_name", None))


def _prompt_agent(step: PromptAgentTaskStep, outside: Outside) -> list[str]:
    if step.agent_name in outside.agents:
        return [
            f"an outside agent's prompt can't have {what}" for what, present in (
                ("parts, only a text prompt", step.parts is not None),
                ("output_format", step.output_format),
                ("agentenv_tools", step.agentenv_tools),
                ("snapshot_config", step.snapshot_config),
            ) if present
        ]
    if step.max_conversation_turns > 1 and step.user_agent_name in outside.agents:
        return [f"its user simulator {step.user_agent_name!r} is played from outside"]
    return []


def _deploy_agent(step: DeployAgentTaskStep, outside: Outside) -> list[str]:
    if step.agent_name not in outside.agents:
        return []
    return [
        f"an outside agent can't have {what}" for what, present in (
            ("agent_snapshot_files_artifact_id", step.agent_snapshot_files_artifact_id),
            ("agent_changelog_object_url", step.agent_changelog_object_url),
            ("enable_agent_changelog", step.enable_agent_changelog),
        ) if present
    ]


def _deploys_an_agent(step: TaskStep, outside: Outside) -> list[str]:
    if step.agent_name not in outside.agents:
        return []
    return [f"{step.type} deploys {step.agent_name!r}, which is played from outside"]


def _needs_the_agent(step: TaskStep, outside: Outside) -> list[str]:
    if step.agent_name not in outside.agents:
        return []
    return [f"{step.type} needs {step.agent_name!r} as a real agent, and it is played from outside"]


def _validates_an_agent(step: TaskStep, outside: Outside) -> list[str]:
    return ["validates the task's deployed A2A agent, and an episode plays one from outside"]


def _collect_artifacts(step: CollectArtifactsTaskStep, outside: Outside) -> list[str]:
    if step.manifest_step_id in outside.prompts:
        return [f"its manifest is outside prompt {step.manifest_step_id!r}"]
    return []


def _register_env_triggers(step: RegisterEnvTriggersStep, outside: Outside) -> list[str]:
    if step.executor_agent_name not in outside.agents:
        return []
    return [f"its executor {step.executor_agent_name!r} is played from outside"]


def _peer_agents(step: PeerAgentsTaskStep, outside: Outside) -> list[str]:
    named = dict.fromkeys(
        name for peering in step.peerings for name in (peering.source_agent_name, *peering.peer_agent_names)
    )
    return [f"peers {name!r}, which is played from outside" for name in named if name in outside.agents]


def _rubrics_verifier(step: RubricsVerifierTaskStep, outside: Outside) -> list[str]:
    if step.use_agent_judge and step.agent_name in outside.agents:
        return [f"judges in {step.agent_name!r}'s sandbox, and it is played from outside"]
    return []


CHECKS: dict[str, Callable[..., list[str]]] = {
    "prompt_agent": _prompt_agent,
    "deploy_agent": _deploy_agent,
    "deploy_human_agent": _deploys_an_agent,
    "register_agent_triggers": _needs_the_agent,
    "snapshot_agent_state": _needs_the_agent,
    "collect_artifacts": _collect_artifacts,
    "register_env_triggers": _register_env_triggers,
    "peer_agents": _peer_agents,
    "rubrics_verifier": _rubrics_verifier,
    **dict.fromkeys((
        "verify_a2a_agent_card", "verify_a2a_agent_config_identity", "verify_a2a_agent_mcp", "verify_a2a_core_protocol",
        "verify_a2a_install", "verify_a2a_modalities", "verify_a2a_peer_agents", "verify_a2a_role",
        "verify_a2a_skill_config", "verify_a2a_snapshot", "verify_a2a_system_prompt", "verify_a2a_trajectory",
    ), _validates_an_agent),
}

UNAFFECTED = frozenset({
    # They name no agent.
    "aggregate_verifiers", "apply_server_config", "build_mcp_cli", "combine_universe_verdicts", "configure_scheduled_checks",
    "deploy_env", "deploy_sandbox", "end_computer_recording", "start_computer_recording", "env_outcome_verifier",
    "modify_env_tool_access", "reset_env", "review",
    "run_container_unit_tests_verifier", "run_docker_container", "snapshot_env", "snapshot_computer_env", "sync_env_clock",
    "validation_gate_aggregator", "verify_env_card", "verify_env_core_protocol", "verify_mcp_tool_schema",
    "verify_spec_conformance", "verify_universe_load_export_roundtrip",
    # They read only what an outside turn leaves: its reply.
    "agent_prompt_response_verifier", "verify_mcp_env_assessment",
    # It skips an agent that isn't deployed.
    "teardown_sandboxes",
    # They act on an outside agent's proxy, in its sandbox, as on the agent; install_agent installs the proxy in its place.
    "add_skills", "install_agent", "load_artifact", "run_code", "verify_sandbox",
})

DEPLOYS = ("deploy_agent", "install_agent")
