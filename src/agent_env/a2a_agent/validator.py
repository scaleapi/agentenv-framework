"""A2A agent validator — orchestrates the full validation flow.

Extracted from `A2AAgent.validate()` so the data model (`A2AAgent`) and the
validation behavior live in separate modules.

Adding a new modality probe is one new `ModalityProbe` entry in
`MODALITY_PROBES`. The step ID, prompt ID, `PromptAgentTaskStep`
instantiation, and grading-list entry are all derived from that single
tuple — no need to coordinate three places.

Use as `await A2AAgentValidator.validate(agent)`. `A2AAgent.validate()` is
a thin wrapper that delegates here.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING, Union

from agentenv_protocol.a2a_agent import ObjectChangelogApplyResponse

from agent_env.a2a_agent.object_transfer import (
    TRANSFER_TIMEOUT_SECONDS,
    changelog_apply_call,
    check_changelog_applied,
    invoke_transfer,
)
from agent_env.a2a_agent.staging import transfer_store
from agent_env.config import get_config
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import get_agent_sandbox_provider
from agent_env.store.routing import refuse_local_derivation
from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_modalities import (
    AUDIO_M4A_EXPECTED,
    AUDIO_M4A_PROBE_PARTS,
    AUDIO_MP3_EXPECTED,
    AUDIO_MP3_PROBE_PARTS,
    AUDIO_OGG_EXPECTED,
    AUDIO_OGG_PROBE_PARTS,
    AUDIO_WAV_EXPECTED,
    AUDIO_WAV_PROBE_PARTS,
    IMAGE_GIF_PROBE_PARTS,
    IMAGE_JPEG_PROBE_PARTS,
    IMAGE_PROBE_EXPECTED,
    IMAGE_PROBE_PARTS,
    IMAGE_PROBE_PNG_B64,
    IMAGE_PROBE_PROMPT,
    PDF_PROBE_EXPECTED,
    PDF_PROBE_PARTS,
    TEXT_PROBE_EXPECTED,
    TEXT_PROBE_PARTS,
    VIDEO_PROBE_EXPECTED,
    VIDEO_PROBE_MP4_B64,
    VIDEO_PROBE_PROMPT,
)

if TYPE_CHECKING:
    from agent_env.a2a_agent.a2a_agent import A2AAgent

logger = logging.getLogger(__name__)


def _probe_agent_sandbox_type() -> str | None:
    """The sandbox type the validation agents will deploy on, as far as it is known before they do: the local
    provider's when it is the agent provider, or first in its chain, which runs the agent unless it fails."""
    provider = get_agent_sandbox_provider()
    if isinstance(provider, ChainedSandboxProvider):
        provider = provider.providers[0]
    return LocalSandbox.type if isinstance(provider, LocalSandboxProvider) else None


@dataclass
class _UploadedFixtures:
    """Object-store-hosted probe fixtures created at validation runtime."""
    skill_object_url: str
    png_object_uri: str
    png_signed_url: str
    mp4_object_uri: str


_PartsFactory = Callable[[_UploadedFixtures], list[dict]]


@dataclass
class ModalityProbe:
    """One modality probe in the validator flow.

    `parts` is either a static list of A2A parts (for inline probes whose
    bytes are known at module load) or a callable that takes the runtime
    `_UploadedFixtures` and returns parts (for URI probes whose s3:// or
    presigned https:// URIs only exist after upload).

    Each `ModalityProbe` produces exactly one `PromptAgentTaskStep` and
    one entry in `VerifyA2AModalitiesStep.probes`. Step ID, prompt ID,
    and grading entry all derive from `prompt_id_suffix`.
    """
    modality: str
    prompt_id_suffix: str
    parts: Union[list[dict], _PartsFactory]
    expected: str
    timeout_s: int = 120


MODALITY_PROBES: list[ModalityProbe] = [
    ModalityProbe("text",            "modality-text",            TEXT_PROBE_PARTS,       TEXT_PROBE_EXPECTED),
    ModalityProbe("image/png",       "modality-image",           IMAGE_PROBE_PARTS,      IMAGE_PROBE_EXPECTED),
    ModalityProbe("image/jpeg",      "modality-image-jpeg",      IMAGE_JPEG_PROBE_PARTS, IMAGE_PROBE_EXPECTED),
    ModalityProbe("image/gif",       "modality-image-gif",       IMAGE_GIF_PROBE_PARTS,  IMAGE_PROBE_EXPECTED),
    ModalityProbe("audio/wav",       "modality-audio-wav",       AUDIO_WAV_PROBE_PARTS,  AUDIO_WAV_EXPECTED),
    ModalityProbe("audio/mpeg",      "modality-audio-mp3",       AUDIO_MP3_PROBE_PARTS,  AUDIO_MP3_EXPECTED),
    ModalityProbe("audio/mp4",       "modality-audio-m4a",       AUDIO_M4A_PROBE_PARTS,  AUDIO_M4A_EXPECTED),
    ModalityProbe("audio/ogg",       "modality-audio-ogg",       AUDIO_OGG_PROBE_PARTS,  AUDIO_OGG_EXPECTED),
    ModalityProbe("application/pdf", "modality-pdf",             PDF_PROBE_PARTS,        PDF_PROBE_EXPECTED),
    ModalityProbe(
        "image/png+uri-s3", "modality-image-uri-s3",
        lambda f: [
            {"kind": "text", "text": IMAGE_PROBE_PROMPT},
            {"kind": "file", "file": {"uri": f.png_object_uri, "mimeType": "image/png", "name": "red.png"}},
        ],
        IMAGE_PROBE_EXPECTED,
    ),
    ModalityProbe(
        "image/png+uri-https", "modality-image-uri-https",
        lambda f: [
            {"kind": "text", "text": IMAGE_PROBE_PROMPT},
            {"kind": "file", "file": {"uri": f.png_signed_url, "mimeType": "image/png", "name": "red.png"}},
        ],
        IMAGE_PROBE_EXPECTED,
    ),
    ModalityProbe(
        "video/mp4", "modality-video",
        lambda f: [
            {"kind": "text", "text": VIDEO_PROBE_PROMPT},
            {"kind": "file", "file": {"uri": f.mp4_object_uri, "mimeType": "video/mp4", "name": "clip.mp4"}},
        ],
        VIDEO_PROBE_EXPECTED,
        timeout_s=180,
    ),
]


class A2AAgentValidator:
    """Static namespace for running the full A2A agent validation flow."""

    @staticmethod
    async def validate(
        agent: "A2AAgent",
        on_progress: Optional[Callable[[str], None]] = None,
        litellm_api_key: Optional[str] = None,
    ) -> dict:
        """Run the A2A agent validator task: deploy env + agent, verify card,
        modality probes, MCP tools, skills, snapshot extension, etc.

        Returns the merged `verifications` dict from the run's task context.
        """
        refuse_local_derivation(agent.id, "agent", "validating")
        from agent_env.task import Task
        from agent_env.task_step import (
            AddSkillsTaskStep,
            DeployAgentTaskStep,
            DeployEnvTaskStep,
            PromptAgentTaskStep,
            RubricsVerifierTaskStep,
        )
        from agent_env.task_step.context import TaskStepContext
        from agent_env.task_step.task_steps.a2a_agent_validator import (
            VerifyA2AAgentCardStep,
            VerifyA2AAgentConfigIdentityStep,
            VerifyA2AAgentMCPStep,
            VerifyA2ALitellmAttributionRuntimeStep,
            VerifyA2ALitellmAttributionStep,
            VerifyA2AModalitiesStep,
            VerifyA2APeerAgentsStep,
            VerifyA2ARoleStep,
            VerifyA2ASkillConfigStep,
            VerifyA2ASnapshotStep,
            VerifyA2ASystemPromptStep,
            VerifyA2ATrajectoryStep,
            VerifyCoreA2AProtocolStep,
        )
        from agent_env.task_step.task_steps.add_skills import Skill
        from agent_env.task_step.task_steps.peer_agents import AgentPeers, PeerAgentsTaskStep
        from agent_env.task_step.task_steps.snapshot_agent_state import SnapshotAgentStateTaskStep
        from agent_env.task_step.task_step import TaskStep, TaskStepDependency
        from agent_env.a2a_agent.a2a_agent import A2AAgent

        task_id = f"validate-a2a-{agent.id}-v{agent.version}"

        # ── Upload object-store fixtures (skill + URI probe bytes) ───────────
        skill_object_url = await asyncio.to_thread(
            A2AAgentValidator._upload_skill_fixture,
            agent,
            name="validator-test-s3",
            description=(
                "Validator test skill with project codename. "
                "Use when asked about the project codename."
            ),
            body=(
                "When asked for the project codename, respond with: "
                "VALIDATOR-S3-99"
            ),
        )
        skill_bundle_probe_url = await asyncio.to_thread(
            A2AAgentValidator._upload_skill_fixture,
            agent,
            name="validator-probe-bundle",
            description="Validator probe for the portable skill bundle variant.",
            body="Portable skill bundle validation fixture.",
        )
        skill_s3_probe_url = await asyncio.to_thread(
            A2AAgentValidator._upload_skill_fixture,
            agent,
            name="validator-probe-s3",
            description="Validator probe for the legacy S3 skill variant.",
            body="Legacy S3 skill validation fixture.",
        )
        fixtures = await asyncio.to_thread(A2AAgentValidator._upload_probe_fixtures, agent, skill_object_url)

        # ── Build modality probe steps + matching grading list ───────────────
        deploy_agent_id = f"{task_id}-deploy-agent"
        modality_steps, grading_probes = A2AAgentValidator._build_modality_steps(task_id, fixtures, deploy_agent_id)
        modality_step_ids = [s.id for s in modality_steps]

        # ── Identifiers for non-modality steps ───────────────────────────────
        protocol_prompt_id = f"{task_id}-protocol-prompt"
        skill_prompt_id = f"{task_id}-skill-prompt"
        skill_verifier_id = f"{task_id}-skill-rubric"
        # LiteLLM attribution runtime probe: plant unique IDs on context.metadata
        # so prompt_agent forwards them via /ext/agent-config; the agent records
        # the attribution it emits on the resulting LLM call; the post-prompt
        # verify step queries /ext/attribution-probe and asserts the planted
        # values came through. Unique per validation run to avoid cross-run
        # bleed when /ext/attribution-probe is stateful.
        probe_project_id = f"probe-project-{uuid.uuid4().hex[:12]}"
        probe_task_id = f"probe-task-{uuid.uuid4().hex[:12]}"
        # Snapshot validation: plant a unique token on agent A, snapshot, deploy
        # agent B with the snapshot loaded into a known target context_id, prompt
        # agent B for recall, grade with a rubric.
        snapshot_token = f"SNAPSHOT-TOKEN-{uuid.uuid4().hex[:12]}"
        snapshot_plant_prompt_id = f"{task_id}-snapshot-plant-prompt"
        snapshot_artifact_id = f"{task_id}-snapshot"
        snapshot_target_agent_name = "snapshot-target-agent"
        snapshot_target_context_id = uuid.uuid4().hex
        snapshot_recall_prompt_id = f"{task_id}-snapshot-recall-prompt"
        snapshot_recall_verifier_id = f"{task_id}-snapshot-recall-rubric"
        snapshot_recall_criterion_id = "snapshot_recall_token"
        # Peer-agents validation: deploy a second agent, peer it with the main one,
        # have main relay a unique token from the peer, verify substring match.
        peer_target_agent_name = "peer-target-agent"
        peer_prompt_id = f"{task_id}-peer-prompt"
        peer_token = f"PEERTOKEN-{uuid.uuid4().hex[:12]}"

        deploy_agent_dep = [TaskStepDependency(task_step_id=deploy_agent_id)]
        mcp_step_id = f"{task_id}-mcp"
        skill_chain_id = f"{task_id}-add-skill"
        snapshot_capture_id = f"{task_id}-snapshot-capture"
        snapshot_target_deploy_id = f"{task_id}-deploy-snapshot-target"
        snapshot_recall_rubric_id = f"{task_id}-snapshot-recall-rubric-step"
        peer_target_deploy_id = f"{task_id}-deploy-peer-target"
        peer_setup_id = f"{task_id}-peer-setup"
        skill_rubric_id = f"{task_id}-skill-rubric-step"
        # agent-changelog validation: a dedicated capture agent (enable_agent_changelog)
        # writes a marker file via a tool call; the validator then drives apply
        # onto a fresh agent (post-DAG) and checks the marker was reconstructed.
        changelog_capture_agent_name = "changelog-capture-agent"
        changelog_apply_agent_name = "changelog-apply-agent"
        changelog_token = f"CHANGELOG-TOKEN-{uuid.uuid4().hex[:12]}"
        changelog_marker_path = "/app/changelog_marker.txt"
        changelog_capture_deploy_id = f"{task_id}-changelog-capture-deploy"
        changelog_plant_prompt_id = f"{task_id}-changelog-plant"
        changelog_apply_deploy_id = f"{task_id}-changelog-apply-deploy"
        # install/v1 is validated post-DAG in _validate_install(), gated on the
        # LIVE agent card (context.deployed_agents), so it never depends on a
        # prior validation run having populated agent.metadata["agent_card"].

        task = Task.put(id=task_id, steps=[
            DeployEnvTaskStep(id=f"{task_id}-deploy-env", version=None, env_id=agent.VALIDATION_ENV_ID),
            DeployAgentTaskStep(id=deploy_agent_id, version=None, env_ids=[agent.VALIDATION_ENV_ID], a2a_agent_id=agent.id, a2a_agent_version=agent.version),
            VerifyA2AAgentCardStep(id=f"{task_id}-card", version=None, a2a_agent_id=agent.id, a2a_agent_version=agent.version, depends_on=deploy_agent_dep, fail_task_on_error=False),
            VerifyA2ALitellmAttributionStep(id=f"{task_id}-litellm-attribution", version=None, a2a_agent_id=agent.id, a2a_agent_version=agent.version, depends_on=deploy_agent_dep, fail_task_on_error=False),
            VerifyA2AAgentConfigIdentityStep(id=f"{task_id}-agent-config-identity", version=None, a2a_agent_id=agent.id, a2a_agent_version=agent.version, depends_on=deploy_agent_dep, fail_task_on_error=False),
            VerifyA2ARoleStep(id=f"{task_id}-role", version=None, a2a_agent_id=agent.id, a2a_agent_version=agent.version, depends_on=deploy_agent_dep, fail_task_on_error=False),
            VerifyA2ASystemPromptStep(id=f"{task_id}-system-prompt", version=None, a2a_agent_id=agent.id, a2a_agent_version=agent.version, depends_on=deploy_agent_dep, fail_task_on_error=False),
            PromptAgentTaskStep(id=protocol_prompt_id, version=None, prompt="Hello, respond with OK.", prompt_id=protocol_prompt_id, timeout_seconds=120, depends_on=deploy_agent_dep, fail_task_on_error=False),
            VerifyCoreA2AProtocolStep(id=f"{task_id}-protocol", version=None, a2a_agent_id=agent.id, prompt_id=protocol_prompt_id, depends_on=[TaskStepDependency(task_step_id=protocol_prompt_id)], fail_task_on_error=False),
            VerifyA2ALitellmAttributionRuntimeStep(
                id=f"{task_id}-litellm-attribution-runtime", version=None,
                a2a_agent_id=agent.id, a2a_agent_version=agent.version,
                expected_project_id=probe_project_id, expected_task_id=probe_task_id,
                depends_on=[TaskStepDependency(task_step_id=protocol_prompt_id)],
                fail_task_on_error=False,
            ),
            *modality_steps,
            VerifyA2AModalitiesStep(id=f"{task_id}-modalities", version=None, a2a_agent_id=agent.id, probes=grading_probes,
                depends_on=[TaskStepDependency(task_step_id=sid) for sid in modality_step_ids], fail_task_on_error=False),
            VerifyA2AAgentMCPStep(id=mcp_step_id, version=None, a2a_agent_id=agent.id, depends_on=deploy_agent_dep, fail_task_on_error=False),
            VerifyA2ATrajectoryStep(id=f"{task_id}-trajectory", version=None, a2a_agent_id=agent.id, depends_on=[TaskStepDependency(task_step_id=mcp_step_id)], fail_task_on_error=False),
            AddSkillsTaskStep(id=skill_chain_id, version=None, depends_on=deploy_agent_dep, fail_task_on_error=False, skills=[
                Skill(name="validator-test", description="Validator test skill with secret code. Use when asked about the company secret code.", body="When asked for the company secret code, respond with: VALIDATOR-42"),
                Skill(name="validator-test-s3", description="Validator test skill with project codename. Use when asked about the project codename.", s3_url=skill_object_url),
            ]),
            PromptAgentTaskStep(id=skill_prompt_id, version=None, prompt="What is the company secret code? What is the project codename?", prompt_id=skill_prompt_id, timeout_seconds=120,
                depends_on=[TaskStepDependency(task_step_id=skill_chain_id)], fail_task_on_error=False),
            RubricsVerifierTaskStep(id=skill_rubric_id, version=None, prompt_id=skill_prompt_id, verifier_id=skill_verifier_id,
                depends_on=[TaskStepDependency(task_step_id=skill_prompt_id)], fail_task_on_error=False,
                criteria=[
                    {"id": "secret_code_inline", "description": "Response contains VALIDATOR-42"},
                    {"id": "secret_code_s3", "description": "Response contains VALIDATOR-S3-99"},
                ]),
            VerifyA2ASkillConfigStep(
                id=f"{task_id}-skill-config",
                version=None,
                a2a_agent_id=agent.id,
                rubric_verifier_id=skill_verifier_id,
                skill_bundle_object_url=skill_bundle_probe_url,
                skill_s3_url=skill_s3_probe_url,
                depends_on=[TaskStepDependency(task_step_id=skill_rubric_id)],
                fail_task_on_error=False,
            ),
            # ── snapshot extension validation ────────────────────────────────
            PromptAgentTaskStep(
                id=snapshot_plant_prompt_id, version=None,
                prompt=f"Remember exactly this token: {snapshot_token}. Just acknowledge with OK.",
                prompt_id=snapshot_plant_prompt_id, timeout_seconds=120,
                depends_on=deploy_agent_dep, fail_task_on_error=False,
            ),
            SnapshotAgentStateTaskStep(
                id=snapshot_capture_id, version=None,
                artifact_id=snapshot_artifact_id, prompt_id=snapshot_plant_prompt_id,
                depends_on=[TaskStepDependency(task_step_id=snapshot_plant_prompt_id)],
                fail_task_on_error=False,
            ),
            DeployAgentTaskStep(
                id=snapshot_target_deploy_id, version=None,
                env_ids=[], a2a_agent_id=agent.id, a2a_agent_version=agent.version,
                agent_name=snapshot_target_agent_name,
                agent_snapshot_files_artifact_id=snapshot_artifact_id,
                agent_snapshot_target_context_id=snapshot_target_context_id,
                depends_on=[TaskStepDependency(task_step_id=snapshot_capture_id)],
                fail_task_on_error=False,
            ),
            PromptAgentTaskStep(
                id=snapshot_recall_prompt_id, version=None,
                prompt="What was the exact token I told you to remember earlier? Output only the token, nothing else.",
                prompt_id=snapshot_recall_prompt_id,
                agent_name=snapshot_target_agent_name, context_id=snapshot_target_context_id,
                timeout_seconds=120,
                depends_on=[TaskStepDependency(task_step_id=snapshot_target_deploy_id)],
                fail_task_on_error=False,
            ),
            RubricsVerifierTaskStep(
                id=snapshot_recall_rubric_id, version=None,
                prompt_id=snapshot_recall_prompt_id, verifier_id=snapshot_recall_verifier_id,
                depends_on=[TaskStepDependency(task_step_id=snapshot_recall_prompt_id)],
                fail_task_on_error=False,
                criteria=[{"id": snapshot_recall_criterion_id, "description": f"Response contains the exact token {snapshot_token}"}],
            ),
            VerifyA2ASnapshotStep(
                id=f"{task_id}-snapshot", version=None,
                a2a_agent_id=agent.id,
                rubric_verifier_id=snapshot_recall_verifier_id,
                rubric_criterion_id=snapshot_recall_criterion_id,
                depends_on=[TaskStepDependency(task_step_id=snapshot_recall_rubric_id)],
                fail_task_on_error=False,
            ),
            # ── peer-agents extension validation ─────────────────────────────
            DeployAgentTaskStep(
                id=peer_target_deploy_id, version=None,
                env_ids=[], a2a_agent_id=agent.id, a2a_agent_version=agent.version,
                agent_name=peer_target_agent_name,
                depends_on=deploy_agent_dep,
                fail_task_on_error=False,
            ),
            PeerAgentsTaskStep(
                id=peer_setup_id, version=None,
                peerings=[AgentPeers(source_agent_name=TaskStep.DEFAULT_AGENT_NAME, peer_agent_names=[peer_target_agent_name])],
                depends_on=[TaskStepDependency(task_step_id=peer_target_deploy_id)],
                fail_task_on_error=False,
            ),
            PromptAgentTaskStep(
                id=peer_prompt_id, version=None,
                agent_name=TaskStep.DEFAULT_AGENT_NAME,
                prompt_id=peer_prompt_id, timeout_seconds=300,
                prompt=(
                    f"Use the peer_send_message MCP tool to send the peer named "
                    f"'{peer_target_agent_name}' the exact prompt: 'Reply with only "
                    f"the single token {peer_token} and nothing else.' Then report "
                    f"the peer's full reply verbatim."
                ),
                depends_on=[TaskStepDependency(task_step_id=peer_setup_id)],
                fail_task_on_error=False,
            ),
            VerifyA2APeerAgentsStep(
                id=f"{task_id}-peer-agents", version=None,
                a2a_agent_id=agent.id, a2a_agent_version=agent.version,
                peer_prompt_id=peer_prompt_id, expected_token=peer_token,
                depends_on=[TaskStepDependency(task_step_id=peer_prompt_id)],
                fail_task_on_error=False,
            ),
            # ── agent-changelog validation (capture in DAG; apply driven
            #    by the validator post-run) ─────────────────────────────────────
            DeployAgentTaskStep(
                id=changelog_capture_deploy_id, version=None,
                env_ids=[], a2a_agent_id=agent.id, a2a_agent_version=agent.version,
                agent_name=changelog_capture_agent_name, enable_agent_changelog=True,
                depends_on=deploy_agent_dep, fail_task_on_error=False,
            ),
            PromptAgentTaskStep(
                id=changelog_plant_prompt_id, version=None,
                agent_name=changelog_capture_agent_name,
                prompt=(
                    f"Run exactly one bash command and nothing else: write the exact text "
                    f"'{changelog_token}' (no trailing newline) to the file {changelog_marker_path}, "
                    f"e.g. printf %s '{changelog_token}' > {changelog_marker_path}"
                ),
                prompt_id=changelog_plant_prompt_id, timeout_seconds=180,
                depends_on=[TaskStepDependency(task_step_id=changelog_capture_deploy_id)],
                fail_task_on_error=False,
            ),
            DeployAgentTaskStep(
                id=changelog_apply_deploy_id, version=None,
                env_ids=[], a2a_agent_id=agent.id, a2a_agent_version=agent.version,
                agent_name=changelog_apply_agent_name,
                depends_on=deploy_agent_dep, fail_task_on_error=False,
            ),
        ])
        logger.info(f"Created A2A validation task: {task.id} version={task.version}")

        # ── Execute task steps with progress callbacks ───────────────────────
        log = on_progress or (lambda msg: None)
        on_start = lambda i, total, step, ctx: log(f"Running step [{i+1}/{total}]: {step.type}...")
        on_complete = lambda i, total, step, ctx, dur: log(f"Completed step [{i+1}/{total}]: {step.type} [{dur:.1f}s]")
        initial_context = TaskStepContext()
        if litellm_api_key:
            initial_context.metadata["user_overrides"] = {"litellm_api_key": litellm_api_key}
        # Plant LiteLLM attribution probe values so prompt_agent forwards them
        # via /ext/agent-config — the runtime-attribution verify step reads them
        # back from /ext/attribution-probe and asserts the agent's bridge
        # actually wired them through to outbound LLM calls.
        initial_context.metadata["project_id"] = probe_project_id
        initial_context.metadata["task_id"] = probe_task_id
        num_deploy_steps = 2
        run_kwargs = {"on_step_start": on_start, "on_step_complete": on_complete}

        # Phase 1: deploy steps (must succeed)
        context = await task.run(start_step=0, end_step=num_deploy_steps, context=initial_context, **run_kwargs)

        # Phase 2: validation DAG — every step opts into fail_task_on_error=False
        # at construction so independent chains keep running on per-step failures.
        context = await task.run(start_step=num_deploy_steps, context=context, **run_kwargs)

        # agent-changelog: the validator drives apply directly — it discovers the
        # captured prefix from context, applies it onto the fresh apply agent,
        # and checks the marker file was reconstructed.
        await A2AAgentValidator._validate_agent_changelog(
            agent, context,
            capture_agent_name=changelog_capture_agent_name,
            apply_agent_name=changelog_apply_agent_name,
            marker_path=changelog_marker_path, token=changelog_token,
            on_progress=on_progress,
        )

        # install/v1: gated on the LIVE deployed card (not the stored one), so it
        # runs identically regardless of whether this agent was validated before.
        await A2AAgentValidator._validate_install(
            agent, context, task_id=task_id, on_progress=on_progress,
        )

        await A2AAgentValidator._cleanup_sandboxes(context)
        return context.metadata.get("verifications", {})

    @staticmethod
    async def _validate_install(
        agent: "A2AAgent",
        context,
        task_id: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Validate the install/v1 extension, gated on the LIVE deployed card.

        Runs after the main DAG (mirrors _validate_agent_changelog). The
        install-support decision reads the freshly-deployed agent's card off
        context.deployed_agents — NOT the stored agent.metadata["agent_card"] —
        so the result is identical whether or not the agent was validated before
        (fixes the --skip-validation first-run "upstream chain failure"). The
        sub-chain runs on the MAIN context so its host sandbox is reaped by
        _cleanup_sandboxes and its result lands in the returned verifications dict.
        """
        from agent_env.a2a_agent import A2AAgent
        from agent_env.task import Task
        from agent_env.task_step.task_step import TaskStep, TaskStepDependency
        from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_install import VerifyA2AInstallStep
        from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
        from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep
        from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
        from agent_env.task_step.task_steps.run_docker_container import RunDockerContainerTaskStep
        from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

        log = on_progress or (lambda _msg: None)

        def record(result: dict) -> None:
            try:
                fresh = A2AAgent.get(agent.id)
                validated = dict(fresh.metadata.get("validated_a2a_extensions", {}))
                validated[A2AAgent.EXT_INSTALL] = result
                fresh.update_metadata({**fresh.metadata, "validated_a2a_extensions": validated})
            except Exception as e:  # noqa: BLE001
                logger.warning(f"install/v1: failed to persist validated_a2a_extensions: {e}")
            context.metadata.setdefault("verifications", {})["a2a_install"] = result

        main = next((a for a in context.deployed_agents if a.agent_name == TaskStep.DEFAULT_AGENT_NAME), None)
        # deploy_agent (Phase 1, fail_task_on_error=True) normally guarantees a
        # fetched card, so a missing card here is an anomaly (e.g. a transient
        # /.well-known/agent.json failure), NOT a genuine "not advertised". Log it
        # so the failure mode is observable, and record a distinct reason rather
        # than silently masquerading as absence of the extension.
        card_unavailable = main is None or main.a2a_card is None
        if card_unavailable:
            logger.warning(
                "install/v1: no live a2a_card for deployed agent %r (main_present=%s) — "
                "possible transient card-fetch failure; recording install/v1 as unavailable",
                TaskStep.DEFAULT_AGENT_NAME, main is not None,
            )
        live_card = (main.a2a_card if main else None) or {}
        if A2AAgent.find_extension(live_card, A2AAgent.EXT_INSTALL) is None:
            reason = (
                "live agent card unavailable — deployed agent card was None"
                if card_unavailable
                else "install/v1 not advertised on live agent card"
            )
            log(f"install/v1 skipped: {reason}")
            record({
                "supported": False, "advertised": False,
                "installed_successfully": False, "responded_to_prompt": False,
                "skipped_reason": reason,
            })
            return

        sandbox_name = "install-test-host"
        container_name = "install-test-container"
        install_agent_name = "install-test-agent"
        prompt_id = f"{task_id}-install-test-prompt"
        verifier_id = f"{task_id}-install-test-rubric"
        deploy_sandbox_id = f"{task_id}-install-test-deploy-sandbox"
        run_container_id = f"{task_id}-install-test-run-container"
        install_id = f"{task_id}-install-test-install"
        rubric_step_id = f"{task_id}-install-test-rubric-step"
        validator_id = f"{task_id}-install-validator"

        try:
            install_test_image = await asyncio.to_thread(A2AAgentValidator._upload_install_test_image_fixture, agent)
            install_steps = [
                DeploySandboxTaskStep(
                    id=deploy_sandbox_id, version=None,
                    sandbox_name=sandbox_name, sandbox_mode="vm",
                    cpu=2, memory_mb=4096, disk_size_gb=10,
                    exposed_ports=[8000], ttl_seconds=1800,
                    fail_task_on_error=False,
                ),
                RunDockerContainerTaskStep(
                    id=run_container_id, version=None,
                    sandbox_name=sandbox_name,
                    docker_context_artifact_id=install_test_image.id,
                    docker_context_artifact_version=install_test_image.version,
                    container_name=container_name,
                    image_tag=f"{container_name}:latest",
                    ports=[8000],
                    command_override="sleep infinity",
                    depends_on=[TaskStepDependency(task_step_id=deploy_sandbox_id)],
                    fail_task_on_error=False,
                ),
                InstallAgentTaskStep(
                    id=install_id, version=None,
                    sandbox_name=sandbox_name,
                    container_name=container_name,
                    a2a_agent_id=agent.id, a2a_agent_version=agent.version,
                    agent_name=install_agent_name,
                    depends_on=[TaskStepDependency(task_step_id=run_container_id)],
                    fail_task_on_error=False,
                ),
                PromptAgentTaskStep(
                    id=prompt_id, version=None,
                    prompt="Respond with OK.",
                    prompt_id=prompt_id,
                    agent_name=install_agent_name,
                    timeout_seconds=120,
                    depends_on=[TaskStepDependency(task_step_id=install_id)],
                    fail_task_on_error=False,
                ),
                RubricsVerifierTaskStep(
                    id=rubric_step_id, version=None,
                    prompt_id=prompt_id,
                    verifier_id=verifier_id,
                    # Trivial "response is non-empty" check — use the direct-LLM
                    # judge instead of auto-deploying an ephemeral judge agent.
                    use_agent_judge=False,
                    criteria=[
                        {"id": "responded", "description": "Response is non-empty (the installed agent actually responded)"},
                    ],
                    depends_on=[TaskStepDependency(task_step_id=prompt_id)],
                    fail_task_on_error=False,
                ),
                VerifyA2AInstallStep(
                    id=validator_id, version=None,
                    a2a_agent_id=agent.id, a2a_agent_version=agent.version,
                    agent_name=install_agent_name,
                    rubric_verifier_id=verifier_id,
                    depends_on=[TaskStepDependency(task_step_id=rubric_step_id)],
                    fail_task_on_error=False,
                ),
            ]
            on_start = lambda i, total, step, ctx: log(f"Running install step [{i+1}/{total}]: {step.type}...")
            on_complete = lambda i, total, step, ctx, dur: log(f"Completed install step [{i+1}/{total}]: {step.type} [{dur:.1f}s]")
            sub_task = Task(id=f"{task_id}-install", version=None, steps=install_steps)
            # Run on the MAIN context: the install host lands in context.deployed_agents
            # (cleaned up by _cleanup_sandboxes) and VerifyA2AInstallStep writes the
            # result into context.metadata["verifications"]["a2a_install"] itself.
            await sub_task.run(context=context, on_step_start=on_start, on_step_complete=on_complete)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"install/v1 validation raised: {e}")
            record({
                "supported": False, "advertised": True,
                "installed_successfully": False, "responded_to_prompt": False,
                "error": f"install validation raised: {e}",
            })

    @staticmethod
    async def _validate_agent_changelog(
        agent: "A2AAgent",
        context,
        capture_agent_name: str,
        apply_agent_name: str,
        marker_path: str,
        token: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Drive the agent-changelog enable→apply round-trip directly (no task step).

        Discovers the captured changelog prefix from context.metadata['agent_changelog']
        (recorded by the capture agent's deploy), PUTs apply onto the fresh apply
        agent, then deterministically checks the marker file was reconstructed by
        exec-ing into the apply agent's sandbox. Records the outcome on the agent's
        validated_a2a_extensions and in context.metadata['verifications']."""
        from agent_env.a2a_agent import A2AAgent
        from agent_env.providers.sandbox_providers.sandbox_provider import (
            SANDBOX_MODE_VM, build_sandbox_provider, get_agent_sandbox_provider,
        )

        log = on_progress or (lambda _msg: None)

        def record(*, supported, advertised, save_ok, apply_ok, roundtrip_ok, note=""):
            try:
                fresh = A2AAgent.get(agent.id)
                validated = fresh.metadata.get("validated_a2a_extensions", {})
                snap = validated.setdefault(A2AAgent.EXT_SNAPSHOT, {})
                methods = snap.setdefault("methods", {})
                methods[A2AAgent.SNAPSHOT_METHOD_ENABLE_CHANGELOG] = {"supported": save_ok}
                methods[A2AAgent.SNAPSHOT_METHOD_APPLY_CHANGELOG] = {"supported": apply_ok and roundtrip_ok}
                snap["changelog_roundtrip"] = roundtrip_ok
                fresh.update_metadata({**fresh.metadata, "validated_a2a_extensions": validated})
            except Exception as e:  # noqa: BLE001
                logger.warning(f"agent-changelog: failed to persist validated_a2a_extensions: {e}")
            context.metadata.setdefault("verifications", {})["a2a_agent_changelog"] = {
                "methods_advertised": advertised, "save": save_ok,
                "apply": apply_ok, "roundtrip": roundtrip_ok, "note": note,
            }
            logger.info(
                f"agent-changelog validation: advertised={advertised} save={save_ok} "
                f"apply={apply_ok} roundtrip={roundtrip_ok} {note}"
            )

        capture = next(
            (e for e in (context.metadata.get("agent_changelog") or [])
             if e.get("agent_name") == capture_agent_name), None)
        capture_source = capture.get("object_url") if capture else None
        apply_agent = next(
            (d for d in context.deployed_agents if d.agent_name == apply_agent_name), None)
        advertised = apply_agent is not None and A2AAgent.extension_method(
            apply_agent.a2a_card or {}, A2AAgent.EXT_SNAPSHOT,
            A2AAgent.SNAPSHOT_METHOD_APPLY_CHANGELOG) is not None
        save_ok = capture_source is not None
        if not save_ok or apply_agent is None or not advertised:
            record(supported=False, advertised=advertised, save_ok=save_ok,
                   apply_ok=False, roundtrip_ok=False,
                   note="capture prefix or apply agent missing / extension not advertised")
            return

        # Apply the captured changelog onto the fresh apply agent.
        method, path = A2AAgent.operation(
            A2AAgent.find_extension(apply_agent.a2a_card, A2AAgent.EXT_SNAPSHOT),
            A2AAgent.SNAPSHOT_METHOD_APPLY_CHANGELOG)
        a2a_url = apply_agent.a2a_url or apply_agent.api_url
        store = transfer_store(
            get_config().get_object_store(), a2a_url, apply_agent.a2a_card, sandbox_type=apply_agent.sandbox_type
        )
        try:
            call = await asyncio.to_thread(
                changelog_apply_call,
                method,
                store,
                agent_name=apply_agent_name,
                source_url=capture_source,
                portable=capture.get("transfer_mode") == "objects",
                sandbox_type=apply_agent.sandbox_type,
            )
        except RuntimeError as e:
            record(supported=False, advertised=advertised, save_ok=save_ok,
                   apply_ok=False, roundtrip_ok=False, note=str(e))
            return
        except Exception as e:  # noqa: BLE001
            record(supported=False, advertised=advertised, save_ok=save_ok,
                   apply_ok=False, roundtrip_ok=False, note=f"apply request failed: {e}")
            return
        try:
            log("Applying agent-changelog onto apply agent...")
            answer = await invoke_transfer(
                a2a_url + path,
                call,
                verb="PUT",
                operation="changelog apply",
                timeout=TRANSFER_TIMEOUT_SECONDS,
                response_model=ObjectChangelogApplyResponse,
                store=store,
            )
            if call.mode == "objects":
                check_changelog_applied(answer, call, agent_name=apply_agent_name)
        except Exception as e:  # noqa: BLE001
            record(supported=False, advertised=advertised, save_ok=save_ok,
                   apply_ok=False, roundtrip_ok=False, note=f"apply request failed: {e}")
            return

        # Deterministically verify the marker file was reconstructed.
        roundtrip_ok = False
        provider = None
        try:
            provider = (build_sandbox_provider(apply_agent.sandbox_type)
                        if apply_agent.sandbox_type else get_agent_sandbox_provider())
            sandbox = await provider.get_sandbox(apply_agent.sandbox_id)
            if sandbox.mode == SANDBOX_MODE_VM:
                container = await A2AAgentValidator._discover_agent_container(sandbox)
                args = ("sudo", "docker", "exec", container, "cat", marker_path)
            else:
                args = ("cat", marker_path)
            exit_code, stdout, _ = await sandbox.exec_with_output(*args)
            roundtrip_ok = exit_code == 0 and token in (stdout or "")
        except Exception as e:  # noqa: BLE001
            record(supported=False, advertised=advertised, save_ok=save_ok,
                   apply_ok=True, roundtrip_ok=False, note=f"marker verification failed: {e}")
            return
        finally:
            if provider is not None:
                try:
                    await provider.close()
                except Exception:  # noqa: BLE001
                    pass

        record(supported=roundtrip_ok, advertised=advertised, save_ok=save_ok,
               apply_ok=True, roundtrip_ok=roundtrip_ok)

    @staticmethod
    async def _discover_agent_container(sandbox) -> str:
        """Find the agent container on a VM sandbox (mirrors collect_artifacts /
        verify_sandbox): prefer 'agent-api', else the first 'a2a-agent-*'."""
        exit_code, stdout, stderr = await sandbox.exec_with_output(
            "sudo", "docker", "ps", "--format", "{{.Names}}")
        if exit_code != 0:
            raise RuntimeError(f"docker ps failed: {stderr[:200]}")
        running = [n.strip() for n in stdout.splitlines() if n.strip()]
        if sandbox.container_name in running:
            return sandbox.container_name
        fallback = [n for n in running if n.startswith("a2a-agent-")]
        if not fallback:
            raise RuntimeError(f"no agent container found; running: {running}")
        return fallback[0]

    @staticmethod
    def _upload_install_test_image_fixture(agent: "A2AAgent"):
        """Upload a minimal Dockerfile as a FileArtifactUniverse so the install
        validator chain has a base task image to layer the agent onto.

        The Dockerfile is intentionally minimal: `FROM ubuntu:24.04`, no CMD.
        The validator pairs it with `command_override="sleep infinity"` on
        RunDockerContainerTaskStep so the bare container stays alive long
        enough for the install to finish.
        """
        from pathlib import Path
        import os
        import tempfile
        import time

        from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse

        ts = int(time.time())
        universe_id = f"validate-install-image-{agent.id}-v{agent.version}-{ts}"
        config = get_config()
        s3_url = config.get_object_store().object_url(f"{config.get_artifact_key_prefix()}a2a_validator/install_test_image/{ts}/")

        with tempfile.NamedTemporaryFile("w", suffix=".Dockerfile", delete=False) as f:
            f.write("FROM ubuntu:24.04\n")
            tmp_path = f.name
        try:
            universe = FileArtifactUniverse.put_bundled(
                id=universe_id,
                files={"Dockerfile": Path(tmp_path)},
                s3_url=s3_url,
            )
            logger.info(f"Uploaded install-test image universe '{universe.id}' v{universe.version} to {s3_url}")
            return universe
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _upload_skill_fixture(
        agent: "A2AAgent", *, name: str, description: str, body: str
    ) -> str:
        from agent_env.task_step.task_steps.add_skills import Skill

        config = get_config()
        store = config.get_object_store()
        prefix = (
            f"{config.get_artifact_key_prefix()}a2a_validator/validator_skill/{agent.id}-v{agent.version}/{name}/"
        )
        skill = Skill(
            name=name,
            description=description,
            body=body,
        )
        store.put(f"{prefix}SKILL.md", skill.to_skill_md().encode(), content_type="text/markdown", allow_overwrite=True)
        object_url = store.object_url(prefix)
        logger.info(f"Uploaded test skill to {object_url}")
        return object_url

    @staticmethod
    def _upload_probe_fixtures(agent: "A2AAgent", skill_object_url: str) -> _UploadedFixtures:
        """Upload `red.png` and `clip.mp4` for the URI probes.

        The PNG bytes are also sent inline via the existing PNG probe — comparing
        the two outcomes verifies both FilePart delivery modes work for this
        wrapper. The MP4 is too big for practical inline use and the URI path is
        the point.
        """

        config = get_config()
        store = config.get_object_store()
        prefix = f"{config.get_artifact_key_prefix()}a2a_validator/probe_fixtures/{agent.id}-v{agent.version}/"

        png_object_uri = store.put(f"{prefix}red.png", base64.b64decode(IMAGE_PROBE_PNG_B64), content_type="image/png", allow_overwrite=True)
        png_signed_url = store.signed_get_url(png_object_uri)
        if png_signed_url is None and store.supports_transfer_grants and store.grants_reach(_probe_agent_sandbox_type()):
            png_signed_url = store.issue_read_grant(png_object_uri).url
        if png_signed_url is None:
            raise RuntimeError("A2A validation requires an object store that signs URLs or issues grants, for the presigned-URI probe.")
        logger.info(f"Uploaded probe fixture to {png_object_uri}")

        mp4_object_uri = store.put(f"{prefix}clip.mp4", base64.b64decode(VIDEO_PROBE_MP4_B64), content_type="video/mp4", allow_overwrite=True)
        logger.info(f"Uploaded video probe fixture to {mp4_object_uri}")

        return _UploadedFixtures(
            skill_object_url=skill_object_url,
            png_object_uri=png_object_uri,
            png_signed_url=png_signed_url,
            mp4_object_uri=mp4_object_uri,
        )

    @staticmethod
    def _build_modality_steps(task_id: str, fixtures: _UploadedFixtures, deploy_agent_id: str):
        """Build modality probe `PromptAgentTaskStep`s and the matching grading list.

        All 11 prompts depend only on deploy-agent so they fan out in parallel
        once the agent is deployed. Adding a new probe is one new entry in
        MODALITY_PROBES, nothing else.
        """
        from agent_env.task_step import PromptAgentTaskStep
        from agent_env.task_step.task_step import TaskStepDependency
        prompt_steps: list[PromptAgentTaskStep] = []
        grading_probes: list[dict] = []
        for probe in MODALITY_PROBES:
            prompt_id = f"{task_id}-{probe.prompt_id_suffix}-prompt"
            parts = probe.parts(fixtures) if callable(probe.parts) else probe.parts
            prompt_steps.append(PromptAgentTaskStep(
                id=f"{task_id}-{probe.prompt_id_suffix}-prompt-step",
                version=None, parts=parts, prompt_id=prompt_id, timeout_seconds=probe.timeout_s,
                depends_on=[TaskStepDependency(task_step_id=deploy_agent_id)],
                fail_task_on_error=False,
            ))
            grading_probes.append({
                "modality": probe.modality,
                "prompt_id": prompt_id,
                "expected": probe.expected,
            })
        return prompt_steps, grading_probes

    @staticmethod
    async def _cleanup_sandboxes(context) -> None:
        from agent_env.env.env import DeployedSandboxEnv
        from agent_env.providers import build_sandbox_provider, get_agent_sandbox_provider, get_env_sandbox_provider
        for deployed_env in context.deployed_envs:
            if not isinstance(deployed_env, DeployedSandboxEnv):  # an env outside our sandboxes owns its own lifetime
                continue
            try:
                provider = build_sandbox_provider(deployed_env.sandbox_type) if deployed_env.sandbox_type else get_env_sandbox_provider()
                sandbox = await provider.get_sandbox(deployed_env.sandbox_id)
                await sandbox.terminate()
            except Exception as e:
                logger.warning(f"Failed to clean up env sandbox {deployed_env.sandbox_id}: {e}")
        for deployed_agent in context.deployed_agents:
            if not deployed_agent.sandbox_id:
                continue
            try:
                provider = build_sandbox_provider(deployed_agent.sandbox_type) if deployed_agent.sandbox_type else get_agent_sandbox_provider()
                sandbox = await provider.get_sandbox(deployed_agent.sandbox_id)
                await sandbox.terminate()
            except Exception as e:
                logger.warning(f"Failed to clean up agent sandbox {deployed_agent.sandbox_id}: {e}")
