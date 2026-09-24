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

For now only `agent_name` is supported as the sandbox source. An `env_id`
field can be added later without renaming the step.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shlex
import uuid
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
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


class VerifySandboxTaskStep(TaskStep):
    type: ClassVar[str] = "verify_sandbox"
    entity_refs = ()

    def __init__(
        self,
        id: str,
        version: Optional[int],
        criteria: Optional[list[dict]] = None,
        agent_name: Optional[str] = None,
        base_dir: str = _DEFAULT_BASE_DIR,
        shell_timeout_seconds: int = _DEFAULT_SHELL_TIMEOUT_S,
        score_aggregator: Optional[ScoreAggregator] = None,
        verifier_id: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.criteria = list(criteria or [])
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME
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
            base_dir=data.get("base_dir", _DEFAULT_BASE_DIR),
            shell_timeout_seconds=data.get("shell_timeout_seconds", _DEFAULT_SHELL_TIMEOUT_S),
            score_aggregator=ScoreAggregator(raw_agg) if raw_agg else None,
            verifier_id=data.get("verifier_id"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.providers.sandbox_provider import (
            SANDBOX_MODE_VM,
            build_sandbox_provider,
            get_agent_sandbox_provider,
        )

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
        sandbox = await provider.get_sandbox(agent.sandbox_id)
        logger.info(f"Connected to sandbox {agent.sandbox_id} (mode={sandbox.mode})")

        container = (
            await self._discover_container(sandbox)
            if sandbox.mode == SANDBOX_MODE_VM
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
            except Exception as e:
                logger.warning(f"criterion #{idx} ({rtype}) evaluator raised: {e}")
                outcome = {"score": 0.0, "passed": False, "justification": f"evaluator error: {e!s}"[:300]}

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

    def _resolve_path(self, path: str) -> str:
        return path if path.startswith("/") else f"{self.base_dir.rstrip('/')}/{path}"

    async def _exec(self, sandbox, args: tuple[str, ...]) -> tuple[int, str, str]:
        """Wrap exec_with_output with the per-rubric shell timeout."""
        return await asyncio.wait_for(
            sandbox.exec_with_output(*args),
            timeout=self.shell_timeout_seconds,
        )

    async def _eval_paths_exist(
        self, sandbox, container: Optional[str], paths: list[str], test_flag: str,
    ) -> dict:
        from agent_env.providers.sandbox_provider import SANDBOX_MODE_VM

        if not paths:
            return {"score": 0.0, "passed": False, "justification": "no paths specified"}
        missing: list[str] = []
        for p in paths:
            full = self._resolve_path(p)
            if sandbox.mode == SANDBOX_MODE_VM:
                args = ("sudo", "docker", "exec", container, "test", test_flag, full)
            else:
                args = ("test", test_flag, full)
            try:
                exit_code, _, _ = await self._exec(sandbox, args)
            except asyncio.TimeoutError:
                return {
                    "score": 0.0, "passed": False,
                    "justification": f"timed out checking {p} after {self.shell_timeout_seconds}s",
                }
            if exit_code != 0:
                missing.append(p)
        if missing:
            return {"score": 0.0, "passed": False, "justification": f"Missing: {missing}"}
        return {"score": 1.0, "passed": True, "justification": "All paths exist"}

    async def _eval_file_contains(
        self, sandbox, container: Optional[str], path: str, needle: str,
    ) -> dict:
        from agent_env.providers.sandbox_provider import SANDBOX_MODE_VM

        if not path:
            return {"score": 0.0, "passed": False, "justification": "no path"}
        full = self._resolve_path(path)
        if sandbox.mode == SANDBOX_MODE_VM:
            args = ("sudo", "docker", "exec", container, "cat", full)
        else:
            args = ("cat", full)
        try:
            exit_code, stdout, stderr = await self._exec(sandbox, args)
        except asyncio.TimeoutError:
            return {
                "score": 0.0, "passed": False,
                "justification": f"timed out reading {path} after {self.shell_timeout_seconds}s",
            }
        if exit_code != 0:
            return {
                "score": 0.0, "passed": False,
                "justification": f"could not read {path}: {stderr[:100]}",
            }
        try:
            found = bool(re.search(needle, stdout))
        except re.error as e:
            return {"score": 0.0, "passed": False, "justification": f"invalid regex pattern {needle!r}: {e}"}
        return {
            "score": 1.0 if found else 0.0,
            "passed": found,
            "justification": ("contains" if found else "does not contain") + f" {needle!r}",
        }

    async def _eval_shell(
        self, sandbox, container: Optional[str], bash_cmd: str,
    ) -> dict:
        from agent_env.providers.sandbox_provider import SANDBOX_MODE_VM

        if not bash_cmd:
            return {"score": 0.0, "passed": False, "justification": "no bash_cmd"}
        if sandbox.mode == SANDBOX_MODE_VM:
            args = (
                "sudo", "docker", "exec", "-w", self.base_dir, container,
                "bash", "-c", bash_cmd,
            )
        else:
            args = ("bash", "-c", f"cd {shlex.quote(self.base_dir)} && ({bash_cmd})")
        try:
            exit_code, _, stderr = await self._exec(sandbox, args)
        except asyncio.TimeoutError:
            return {
                "score": 0.0, "passed": False,
                "justification": f"timed out after {self.shell_timeout_seconds}s",
            }
        ok = exit_code == 0
        return {
            "score": 1.0 if ok else 0.0,
            "passed": ok,
            "justification": f"exit={exit_code}; stderr={stderr[:200]}",
        }

    async def _discover_container(self, sandbox) -> str:
        """Find the agent container on a VM sandbox. Mirrors collect_artifacts._discover_container."""
        exit_code, stdout, stderr = await sandbox.exec_with_output(
            "sudo", "docker", "ps", "--format", "{{.Names}}",
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to list containers: {stderr[:300]}")
        running = [n.strip() for n in stdout.splitlines() if n.strip()]
        if sandbox.container_name in running:
            return sandbox.container_name
        fallback = [n for n in running if n.startswith("a2a-agent-")]
        if fallback:
            if len(fallback) > 1:
                logger.warning(f"Multiple a2a-agent-* containers found; using first: {fallback}")
            return fallback[0]
        raise RuntimeError(
            f"No agent container found on the VM (looked for 'agent-api' or 'a2a-agent-*'). "
            f"Running containers: {running}."
        )
