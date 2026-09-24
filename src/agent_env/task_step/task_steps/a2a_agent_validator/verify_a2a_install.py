"""Verify A2A agent install/v1 extension end-to-end.

Pure recorder step. The actual install exercise (deploy sandbox -> run bare task
container -> InstallAgentTaskStep -> PromptAgent -> RubricsVerifier) lives in the
chained steps before this one in the validator DAG. This step reads the rubric
verdict + checks the installed agent's card, then records the outcome on
`agent.metadata["validated_a2a_extensions"]` and `context.metadata["verifications"]`.
"""

from __future__ import annotations

import logging
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


class VerifyA2AInstallStep(TaskStep):
    type: ClassVar[str] = "verify_a2a_install"
    entity_refs = (EntityRef.agent("a2a_agent_id", version_field="a2a_agent_version"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        a2a_agent_id: str,
        a2a_agent_version: Optional[int] = None,
        agent_name: str = "install-test-agent",
        rubric_verifier_id: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.a2a_agent_version = a2a_agent_version
        self.agent_name = agent_name
        self.rubric_verifier_id = rubric_verifier_id

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["a2a_agent_id"] = self.a2a_agent_id
        base["a2a_agent_version"] = self.a2a_agent_version
        base["agent_name"] = self.agent_name
        base["rubric_verifier_id"] = self.rubric_verifier_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyA2AInstallStep:
        return cls(
            **cls._base_from_dict(data),
            a2a_agent_id=data["a2a_agent_id"],
            a2a_agent_version=data.get("a2a_agent_version"),
            agent_name=data.get("agent_name", "install-test-agent"),
            rubric_verifier_id=data.get("rubric_verifier_id"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent

        deployed = next(
            (a for a in context.deployed_agents if a.agent_name == self.agent_name),
            None,
        )
        result: dict = {
            "supported": False,
            "advertised": False,
            "installed_successfully": deployed is not None,
            "responded_to_prompt": False,
        }

        if deployed is None:
            # Install never produced a DeployedAgent — upstream chain failed.
            result["error"] = (
                "InstallAgentTaskStep did not produce a deployed agent named "
                f"{self.agent_name!r} (upstream chain failure); check earlier validator step logs"
            )
            logger.warning(f"[verify_a2a_install] {result['error']}")
        else:
            card = deployed.a2a_card or {}
            result["advertised"] = A2AAgent.find_extension(card, A2AAgent.EXT_INSTALL) is not None

            if self.rubric_verifier_id:
                rubric = context.metadata.get("verifications", {}).get(self.rubric_verifier_id, {})
                rubric_results = rubric.get("results") or []
                responded = any(
                    r.get("id") == "responded" and r.get("result") for r in rubric_results
                )
                result["responded_to_prompt"] = responded
                result["rubric_score"] = rubric.get("score")
                logger.info(
                    f"[verify_a2a_install] rubric {self.rubric_verifier_id!r}: "
                    f"responded={responded} score={rubric.get('score')}"
                )

            result["supported"] = result["installed_successfully"] and result["responded_to_prompt"]

        # Persist on the A2AAgent doc so the hub / next consumers can query it.
        # Guard the DB write: this step runs with fail_task_on_error=False, so a
        # raised exception would be swallowed by the task runner and the result
        # lost. Catch it and always fall through to the in-context write.
        try:
            agent = A2AAgent.get(self.a2a_agent_id, self.a2a_agent_version)
            validated_ext = dict(agent.metadata.get("validated_a2a_extensions", {}))
            validated_ext[A2AAgent.EXT_INSTALL] = result
            agent.update_metadata({**agent.metadata, "validated_a2a_extensions": validated_ext})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[verify_a2a_install] failed to persist validated_a2a_extensions: {e}")

        context.metadata.setdefault("verifications", {})["a2a_install"] = result
        logger.info(f"[verify_a2a_install] result: {result}")
        return context
