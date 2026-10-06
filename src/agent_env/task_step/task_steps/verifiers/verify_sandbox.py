"""Verify filesystem-side rubric criteria against a live sandbox.

Handles the four "probe" criterion types from the MM universal-rubric format:
  - probe_file_exists      — every path in `paths` exists at <base_dir>/<path>
  - probe_dir_exists       — same, but checks they're directories
  - probe_file_contains    — first path's contents match `expected` as a regex (re.search)
  - bash_cmd_succeeds      — `bash -c <bash_cmd>` with cwd=<base_dir> exits 0

Other criterion types in the input list are silently skipped (counted in
results but excluded from the aggregator) so one criterion set can be split
across this step and a response-side verifier (e.g. `rubrics_verifier`).

Criteria use the same shape as `RubricsVerifierTaskStep.criteria` — a list of
dicts with a `criterion` (text), optional `weight` (defaults 1.0), and the
per-type fields documented above. Result rows merge `{**criterion, **outcome}`
so every input field flows through to the verification output.

The sandbox is a deployed agent's (`agent_name`, the default; on a VM the probes
run inside its container) or one a `deploy_sandbox` step created (`sandbox_name`;
the probes run directly on it).
"""
from __future__ import annotations

import asyncio
import logging
import re
import shlex
import uuid
from typing import ClassVar, Optional

from agent_env.providers.sandbox_providers.sandbox_provider import (
    SANDBOX_MODE_VM,
    build_sandbox_provider,
    get_agent_sandbox_provider,
    get_sandbox_provider,
)
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.sandbox_utils.sandbox_utils import find_agent_container
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score

logger = logging.getLogger(__name__)

_HANDLED_TYPES = {
    "probe_file_exists",
    "probe_dir_exists",
    "probe_file_contains",
    "bash_cmd_succeeds",
}
_DEFAULT_BASE_DIR = "/app"
_DEFAULT_SHELL_TIMEOUT_S = 120


def _outcome(passed: bool, justification: str) -> dict:
    return {"score": 1.0 if passed else 0.0, "passed": passed, "justification": justification}


class VerifySandboxTaskStep(TaskStep):
    type: ClassVar[str] = "verify_sandbox"
    entity_refs = ()

    def __init__(
        self,
        id: str,
        version: Optional[int],
        criteria: Optional[list[dict]] = None,
        agent_name: Optional[str] = None,
        sandbox_name: Optional[str] = None,
        base_dir: str = _DEFAULT_BASE_DIR,
        shell_timeout_seconds: int = _DEFAULT_SHELL_TIMEOUT_S,
        score_aggregator: Optional[ScoreAggregator] = None,
        verifier_id: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        # A blank sandbox_name counts as unset, like a blank agent_name: forms and TOML can't send null.
        if isinstance(sandbox_name, str) and not sandbox_name.strip():
            sandbox_name = None
        # A defaulted agent_name is tolerated: every step serialized before `sandbox_name` carries one.
        if sandbox_name is not None and agent_name and agent_name != TaskStep.DEFAULT_AGENT_NAME:
            raise ValueError("verify_sandbox: set either `agent_name` or `sandbox_name`, not both")
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.criteria = list(criteria or [])
        self.sandbox_name = sandbox_name
        self.agent_name = None if sandbox_name is not None else agent_name or TaskStep.DEFAULT_AGENT_NAME
        self.base_dir = base_dir
        self.shell_timeout_seconds = shell_timeout_seconds
        if isinstance(score_aggregator, str):
            score_aggregator = ScoreAggregator(score_aggregator)
        self.score_aggregator = score_aggregator or ScoreAggregator.ALL_PASS
        self.verifier_id = verifier_id or uuid.uuid4().hex

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["criteria"] = self.criteria
        base["agent_name"] = self.agent_name
        if self.sandbox_name is not None:
            base["sandbox_name"] = self.sandbox_name
        base["base_dir"] = self.base_dir
        base["shell_timeout_seconds"] = self.shell_timeout_seconds
        base["score_aggregator"] = self.score_aggregator.value
        base["verifier_id"] = self.verifier_id
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "VerifySandboxTaskStep":
        raw_agg = data.get("score_aggregator")
        return cls(
            **cls._base_from_dict(data),
            criteria=data.get("criteria"),
            agent_name=data.get("agent_name"),
            sandbox_name=data.get("sandbox_name"),
            base_dir=data.get("base_dir", _DEFAULT_BASE_DIR),
            shell_timeout_seconds=data.get("shell_timeout_seconds", _DEFAULT_SHELL_TIMEOUT_S),
            score_aggregator=ScoreAggregator(raw_agg) if raw_agg else None,
            verifier_id=data.get("verifier_id"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        on_agent = self.sandbox_name is None
        sandbox = await (self._agent_sandbox(context) if on_agent else self._deployed_sandbox(context))
        logger.info(f"Connected to sandbox {sandbox.sandbox_id} (mode={sandbox.mode})")
        container = (
            await find_agent_container(sandbox)
            if on_agent and sandbox.mode == SANDBOX_MODE_VM
            else None
        )
        if container:
            logger.info(f"Using agent container: {container}")

        results: list[dict] = []
        for idx, criterion in enumerate(self.criteria):
            rtype = criterion.get("type")
            if rtype not in _HANDLED_TYPES:
                results.append({
                    **criterion,
                    "criterion_index": idx,
                    "skipped": True,
                    "justification": f"type {rtype!r} not handled by verify_sandbox",
                })
                continue

            try:
                outcome = await self._eval_criterion(sandbox, container, criterion)
            except TimeoutError as e:
                outcome = _outcome(False, str(e))
            except Exception as e:
                logger.warning(f"criterion #{idx} ({rtype}) evaluator raised: {e}")
                outcome = _outcome(False, f"evaluator error: {e!s}"[:300])

            results.append({
                **criterion,
                "criterion_index": idx,
                "score": float(outcome["score"]),
                "result": bool(outcome["passed"]),
                "justification": outcome["justification"],
                "skipped": False,
            })

        # Aggregate over only the rows we actually evaluated.
        non_skipped = [r for r in results if not r.get("skipped")]
        score = aggregate_score(non_skipped, self.score_aggregator)

        if "verifications" not in context.metadata:
            context.metadata["verifications"] = {}
        context.metadata["verifications"][self.verifier_id] = {
            "results": results,
            "score": score,
        }
        logger.info(
            f"Verification '{self.verifier_id}': "
            f"{len(non_skipped)}/{len(self.criteria)} criteria evaluated, score={score}"
        )
        return context

    async def _agent_sandbox(self, context: TaskStepContext):
        agent = next(
            (a for a in context.deployed_agents if a.agent_name == self.agent_name),
            None,
        )
        if agent is None:
            raise RuntimeError(f"Agent '{self.agent_name}' not found in context.deployed_agents")
        if not agent.sandbox_id:
            raise RuntimeError(f"Agent '{self.agent_name}' has no sandbox_id")
        provider = (
            build_sandbox_provider(agent.sandbox_type)
            if agent.sandbox_type
            else get_agent_sandbox_provider()
        )
        return await provider.get_sandbox(agent.sandbox_id)

    async def _deployed_sandbox(self, context: TaskStepContext):
        deployed = next(
            (s for s in context.deployed_sandboxes if s.sandbox_name == self.sandbox_name),
            None,
        )
        if deployed is None:
            raise RuntimeError(
                f"Sandbox '{self.sandbox_name}' not found in context.deployed_sandboxes "
                f"(deployed: {[s.sandbox_name for s in context.deployed_sandboxes]})"
            )
        provider = (
            build_sandbox_provider(deployed.sandbox_type)
            if deployed.sandbox_type
            else get_sandbox_provider()
        )
        return await provider.get_sandbox(deployed.sandbox_id)

    async def _eval_criterion(self, sandbox, container: Optional[str], criterion: dict) -> dict:
        rtype = criterion["type"]
        if rtype == "probe_file_exists":
            return await self._eval_paths_exist(sandbox, container, criterion.get("paths") or [], "-e")
        if rtype == "probe_dir_exists":
            return await self._eval_paths_exist(sandbox, container, criterion.get("paths") or [], "-d")
        if rtype == "probe_file_contains":
            paths = criterion.get("paths") or []
            needle = criterion.get("expected") or criterion.get("value") or ""
            return await self._eval_file_contains(
                sandbox, container, paths[0] if paths else "", needle,
            )
        if rtype == "bash_cmd_succeeds":
            return await self._eval_shell(sandbox, container, criterion.get("bash_cmd") or "")
        raise RuntimeError(f"unhandled criterion type {rtype}")

    @staticmethod
    def _probe_args(sandbox, container: Optional[str], cmd: tuple[str, ...]) -> tuple[str, ...]:
        if container is not None:
            return ("sudo", "docker", "exec", container, *cmd)
        if sandbox.mode == SANDBOX_MODE_VM:
            return ("sudo", *cmd)
        return cmd

    def _resolve_path(self, path: str) -> str:
        return path if path.startswith("/") else f"{self.base_dir.rstrip('/')}/{path}"

    async def _exec(self, sandbox, args: tuple[str, ...], doing: str = "") -> tuple[int, str, str]:
        """Wrap exec_with_output with the per-rubric shell timeout."""
        try:
            return await asyncio.wait_for(sandbox.exec_with_output(*args), timeout=self.shell_timeout_seconds)
        except TimeoutError:
            what = f"timed out {doing}" if doing else "timed out"
            raise TimeoutError(f"{what} after {self.shell_timeout_seconds}s") from None

    async def _eval_paths_exist(
        self, sandbox, container: Optional[str], paths: list[str], test_flag: str,
    ) -> dict:
        if not paths:
            return _outcome(False, "no paths specified")
        missing: list[str] = []
        for p in paths:
            args = self._probe_args(sandbox, container, ("test", test_flag, self._resolve_path(p)))
            exit_code, _, _ = await self._exec(sandbox, args, doing=f"checking {p}")
            if exit_code != 0:
                missing.append(p)
        if missing:
            return _outcome(False, f"Missing: {missing}")
        return _outcome(True, "All paths exist")

    async def _eval_file_contains(
        self, sandbox, container: Optional[str], path: str, needle: str,
    ) -> dict:
        if not path:
            return _outcome(False, "no path")
        args = self._probe_args(sandbox, container, ("cat", self._resolve_path(path)))
        exit_code, stdout, stderr = await self._exec(sandbox, args, doing=f"reading {path}")
        if exit_code != 0:
            return _outcome(False, f"could not read {path}: {stderr[:100]}")
        try:
            found = bool(re.search(needle, stdout))
        except re.error as e:
            return _outcome(False, f"invalid regex pattern {needle!r}: {e}")
        return _outcome(found, ("contains" if found else "does not contain") + f" {needle!r}")

    async def _eval_shell(
        self, sandbox, container: Optional[str], bash_cmd: str,
    ) -> dict:
        if not bash_cmd:
            return _outcome(False, "no bash_cmd")
        if container is not None:
            args = (
                "sudo", "docker", "exec", "-w", self.base_dir, container,
                "bash", "-c", bash_cmd,
            )
        else:
            args = self._probe_args(
                sandbox, None, ("bash", "-c", f"cd {shlex.quote(self.base_dir)} && ({bash_cmd}\n)"),
            )
        exit_code, _, stderr = await self._exec(sandbox, args)
        return _outcome(exit_code == 0, f"exit={exit_code}; stderr={stderr[:200]}")

