"""Run a Python script in a deployed sandbox and store its JSON result.

The script defines ``run(input) -> JSON``, where ``input = {"args", "results"}``.
The result is stored at ``context.metadata["script_results"][result_id]``.

``script_artifact_id`` is either a single-file ``FileArtifact`` or a
``FileArtifactUniverse`` whose ``.py`` members are staged side by side so the entry
file can import its siblings; non-``.py`` members are skipped. ``script_file`` names
the entry, defaulting to the sole ``.py`` member or to ``run.py``.

Target: with ``env_id`` set, the script runs on that env's VM host (no agent, no
LLM key); otherwise it runs inside the agent container (``agent_name``), where it
can see the agent's workspace.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Optional, Protocol

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


_RUN_DIR = "/tmp/agent_env_run_code"

# The bootstrap we run in the sandbox: `python3 runner.py <run_dir> <entrypoint> <script_file>`.
_RUNNER_SOURCE = (Path(__file__).parent / "run_code_runner.py").read_text(encoding="utf-8")

# A universe member with one of these names would shadow run_code's own staging files.
_RESERVED_MEMBERS = frozenset({"runner.py", "input.json", "output.json"})

# Where a single-file FileArtifact is always staged, regardless of its own filename.
_STAGED_SCRIPT = "script.py"

class RunCodeValidationError(ValueError):
    """Bad step config or malformed/oversized script output."""


class RunCodeExecutionError(Exception):
    """Script crashed, timed out, or otherwise failed to run."""


class RunCodeTaskStep(TaskStep):
    type: ClassVar[str] = "run_code"
    entity_refs = (
        EntityRef.artifact("script_artifact_id", version_field="script_artifact_version"),
        EntityRef.env("env_id"),
    )
    DEFAULT_ENTRYPOINT: ClassVar[str] = "run"
    DEFAULT_TIMEOUT_SECONDS: ClassVar[int] = 600
    MAX_OUTPUT_BYTES: ClassVar[int] = 50 * 1024 * 1024
    MAX_SCRIPT_BYTES: ClassVar[int] = 5 * 1024 * 1024  # total .py bytes staged from a universe
    # Not an __init__ default (script_file defaults to None) — the universe entry
    # chosen when several .py members exist.
    ENTRY_WHEN_AMBIGUOUS: ClassVar[str] = "run.py"
    _TIMEOUT_EXIT_CODE: ClassVar[int] = 124  # `timeout` coreutil convention
    _STDERR_TRUNCATE: ClassVar[int] = 1000  # max stderr chars surfaced in errors
    # A collect_artifacts universe can hold hundreds of non-.py members; name a few.
    _SKIPPED_MEMBERS_LOGGED: ClassVar[int] = 10

    # from_dict forwards these as-is; __init__ owns per-field defaulting.
    _SERIALIZED_FIELDS: ClassVar[tuple[str, ...]] = (
        "script_artifact_id",
        "script_artifact_version",
        "script_file",
        "entrypoint",
        "args",
        "result_id",
        "timeout_seconds",
        "env_id",
        "agent_name",
    )

    def __init__(
        self,
        id: str,
        version: Optional[int],
        # --- step-specific fields ---
        script_artifact_id: str,
        entrypoint: str = DEFAULT_ENTRYPOINT,
        script_artifact_version: Optional[int] = None,
        script_file: Optional[str] = None,
        args: Optional[dict] = None,
        result_id: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        env_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        # --- base TaskStep params ---
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.script_artifact_id = script_artifact_id
        self.script_artifact_version = script_artifact_version
        # Universe-only; rejected at resolve time for a single-file FileArtifact.
        self.script_file = script_file
        self.entrypoint = entrypoint or self.DEFAULT_ENTRYPOINT
        self.args = args or {}
        self.result_id = result_id or id
        self.timeout_seconds = timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS
        # `env_id` set → host mode (no agent); otherwise agent mode via `agent_name`.
        self.env_id = env_id
        self.agent_name = agent_name or self.DEFAULT_AGENT_NAME

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({f: getattr(self, f) for f in self._SERIALIZED_FIELDS})
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "RunCodeTaskStep":
        return cls(
            **cls._base_from_dict(data),
            **{f: data[f] for f in cls._SERIALIZED_FIELDS if f in data},
        )

    def preflight(self) -> list[str]:
        """Config problems detectable without a sandbox; resolves documents, never content."""
        from agent_env.store.base import NotFoundError

        # Infra failures (expired creds, unreachable store) are not config problems: propagate.
        try:
            self._resolve_script_artifact()
        except RunCodeValidationError as exc:
            return [str(exc)]
        except NotFoundError:
            return [
                f"run_code '{self.id}': script_artifact '{self.script_artifact_id}' "
                f"(version {self.script_artifact_version or 'latest'}) does not exist"
            ]
        return []

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        script_file, files = self._load_script()
        target = await self._resolve_target(context)
        run_dir = f"{_RUN_DIR}/{self.id}"

        await self._stage(target, run_dir, files, self._build_input(context))
        await self._run(target, run_dir, script_file)
        result = await self._read_output(target, run_dir)

        context.metadata.setdefault("script_results", {})[self.result_id] = result
        logger.info(
            "run_code '%s' completed (result_id=%s, target=%s)",
            self.id, self.result_id, "host" if self.env_id else "agent",
        )
        return context

    # --- execution phases ---

    def _resolve_script_artifact(self) -> tuple[str, Any]:
        """Resolve to ``(entry_filename, artifact)``; reads documents only, never content."""
        from agent_env.artifact import Artifact, FileArtifact
        from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse

        artifact = Artifact.get(self.script_artifact_id, self.script_artifact_version)
        if isinstance(artifact, FileArtifact):
            if self.script_file:
                raise self._validation_error(
                    f"script_file '{self.script_file}' is set but "
                    f"'{self.script_artifact_id}' is a single-file FileArtifact; drop "
                    f"script_file or point at a FileArtifactUniverse"
                )
            # Never the artifact's filename: a non-.py suffix breaks importlib, and
            # runner.py would be overwritten by the bootstrap.
            return (_STAGED_SCRIPT, artifact)
        if isinstance(artifact, FileArtifactUniverse):
            return (self._pick_entry(_member_names(artifact)), artifact)
        raise self._validation_error(
            f"script_artifact '{self.script_artifact_id}' must be a FileArtifact or "
            f"FileArtifactUniverse, got {type(artifact).__name__}"
        )

    def _pick_entry(self, members: list[str]) -> str:
        py = self._validated_py_members(members)
        if self.script_file:
            if not self.script_file.endswith(".py"):
                raise self._validation_error(
                    f"script_file '{self.script_file}' must be a .py file"
                )
            if self.script_file not in py:
                raise self._validation_error(
                    f"script_file '{self.script_file}' is not a .py member of "
                    f"'{self.script_artifact_id}'; .py members: {py}"
                )
            return self.script_file
        if not py:
            raise self._validation_error(
                f"script_artifact '{self.script_artifact_id}' has no .py member; "
                f"members: {sorted(members)}"
            )
        if len(py) == 1:
            return py[0]
        # Universe members can be nested paths, so match the entry by basename.
        default = [n for n in py if os.path.basename(n) == self.ENTRY_WHEN_AMBIGUOUS]
        if len(default) == 1:
            return default[0]
        # Two different reasons, and they need different wording: "no single 'run.py'" reads as
        # "there is no run.py" when in fact there are several, sending the author to look for a file
        # that is not missing. The hub's selector already distinguishes these.
        if default:
            raise self._validation_error(
                f"script_artifact '{self.script_artifact_id}' has {len(default)} files named "
                f"'{self.ENTRY_WHEN_AMBIGUOUS}' ({default}), so the default entry is ambiguous; "
                f"set script_file to one of {py}"
            )
        raise self._validation_error(
            f"script_artifact '{self.script_artifact_id}' has {len(py)} .py members "
            f"and no '{self.ENTRY_WHEN_AMBIGUOUS}'; set script_file to one of {py}"
        )

    def _validated_py_members(self, members: list[str]) -> list[str]:
        """Reject an unstageable member set; return its sorted ``.py`` members."""
        if not members:
            raise self._validation_error(
                f"script_artifact '{self.script_artifact_id}' is an empty FileArtifactUniverse"
            )
        # Member names are unvalidated on write and can come from agent output; we stage as root.
        unsafe = sorted(n for n in members if not _is_safe_member(n))
        if unsafe:
            raise self._validation_error(
                f"script_artifact '{self.script_artifact_id}' has member name(s) {unsafe} "
                f"that are absolute or escape the staging directory"
            )
        py = sorted(n for n in members if n.endswith(".py"))
        clash = sorted(set(py) & _RESERVED_MEMBERS)
        if clash:
            raise self._validation_error(
                f"script_artifact '{self.script_artifact_id}' has member(s) {clash} "
                f"that would shadow run_code's staging files; rename them"
            )
        return py

    def _load_script(self) -> tuple[str, dict[str, bytes]]:
        """Resolve the artifact and load only its ``.py`` members: ``(entry, {name: bytes})``."""
        script_file, artifact = self._resolve_script_artifact()
        if not hasattr(artifact, "get_file_artifacts"):
            return (script_file, {script_file: artifact.load()})

        # .py only: these universes are often collect_artifacts output, uncapped and binary.
        members = artifact.get_file_artifacts()
        py = {name: fa for name, fa in members.items() if name.endswith(".py")}
        if script_file not in py:
            # Unreachable unless the universe doc's refs and ids disagree.
            raise self._validation_error(
                f"entry '{script_file}' is missing from the resolved members of "
                f"'{self.script_artifact_id}' ({sorted(members)})"
            )
        files = {name: fa.load() for name, fa in py.items()}
        total = sum(len(b) for b in files.values())
        if total > self.MAX_SCRIPT_BYTES:
            raise self._validation_error(
                f"script_artifact '{self.script_artifact_id}' .py members total "
                f"{total} bytes, over the {self.MAX_SCRIPT_BYTES}-byte staging cap"
            )
        skipped = sorted(set(members) - set(py))
        if skipped:
            logger.info(
                "run_code '%s': staging %d .py member(s); skipped %d non-.py member(s): %s",
                self.id, len(files), len(skipped), skipped[:self._SKIPPED_MEMBERS_LOGGED],
            )
        return (script_file, files)

    def _build_input(self, context: TaskStepContext) -> dict:
        """Build the {args, results} input passed to the script."""
        return {
            "args": self.args,
            "results": context.metadata.get("script_results", {}),
        }

    async def _resolve_target(self, context: TaskStepContext) -> _ExecTarget:
        """Pick where the script runs: the env's VM host or the agent container."""
        if self.env_id is not None:
            return await self._host_target(context)
        return await self._agent_target(context)

    async def _stage(
        self, target: _ExecTarget, run_dir: str, files: dict[str, bytes], script_input: dict
    ) -> None:
        """Clear the run dir, then write the script files, the input, and the runner.

        Sandboxes are reused across reruns and retries: a stale output.json would be
        read back as this run's result, and a stale sibling module would satisfy an import.
        """
        await target.reset_dir(run_dir)
        for name, content in files.items():
            await target.write_file(f"{run_dir}/{name}", content.decode("utf-8"))
        await target.write_file(f"{run_dir}/input.json", json.dumps(script_input))
        await target.write_file(f"{run_dir}/runner.py", _RUNNER_SOURCE)

    async def _run(self, target: _ExecTarget, run_dir: str, script_file: str) -> None:
        """Run the script under a timeout."""
        exit_code, _stdout, stderr = await target.run(
            "timeout", str(self.timeout_seconds),
            "python3", f"{run_dir}/runner.py", run_dir, self.entrypoint, script_file,
        )
        if exit_code == 0:
            return
        reason = (
            f"timed out after {self.timeout_seconds}s"
            if exit_code == self._TIMEOUT_EXIT_CODE
            else f"exited {exit_code}: {stderr[:self._STDERR_TRUNCATE]}"
        )
        raise self._exec_error(reason)

    async def _read_output(self, target: _ExecTarget, run_dir: str) -> Any:
        """Read and parse the script's JSON result, enforcing the size cap."""
        # Check the byte size before reading any content: a complete,
        # runner-written file is always valid UTF-8 JSON, so an over-cap result is
        # rejected here rather than slurped into the worker — and we never decode a
        # truncated file (which would crash strict UTF-8 decode mid-codepoint).
        path = f"{run_dir}/output.json"
        exit_code, size_out, stderr = await target.run("wc", "-c", path)
        if exit_code != 0:
            raise self._exec_error(f"wrote no result file: {stderr[:self._STDERR_TRUNCATE]}")
        if int(size_out.split()[0]) > self.MAX_OUTPUT_BYTES:
            raise self._validation_error(
                f"result exceeds {self.MAX_OUTPUT_BYTES}-byte cap"
            )
        exit_code, stdout, stderr = await target.run("cat", path)
        if exit_code != 0:
            raise self._exec_error(f"could not read result file: {stderr[:self._STDERR_TRUNCATE]}")
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise self._validation_error(f"result is not valid JSON: {exc}") from exc

    # --- run targets ---

    async def _host_target(self, context: TaskStepContext) -> _ExecTarget:
        from agent_env.providers.sandbox_provider import (
            build_sandbox_provider,
            get_env_sandbox_provider,
        )

        deployed = next(
            (d for d in context.deployed_envs if d.env_id == self.env_id), None
        )
        if deployed is None or not deployed.sandbox_id:
            raise RuntimeError(
                f"Env '{self.env_id}' not found in context.deployed_envs "
                "(or missing sandbox_id)"
            )
        # env sandbox provider, honoring a custom sandbox_type
        provider = (
            build_sandbox_provider(deployed.sandbox_type)
            if deployed.sandbox_type
            else get_env_sandbox_provider()
        )
        sandbox = await provider.get_sandbox(deployed.sandbox_id)
        return _HostTarget(sandbox)

    async def _agent_target(self, context: TaskStepContext) -> _ExecTarget:
        # Runs inside the agent container. VM mode needs `sudo docker exec -u 0`:
        # docker on the host needs sudo, and -u 0 reads the root-owned copied files.
        from agent_env.providers.sandbox_provider import (
            SANDBOX_MODE_VM,
            get_agent_sandbox_provider,
        )

        agent = next(
            (a for a in context.deployed_agents if a.agent_name == self.agent_name), None
        )
        if agent is None or not agent.sandbox_id:
            raise RuntimeError(
                f"Agent '{self.agent_name}' not found in context.deployed_agents "
                "(or missing sandbox_id)"
            )
        sandbox = await get_agent_sandbox_provider().get_sandbox(agent.sandbox_id)
        prefix = (
            ("sudo", "docker", "exec", "-u", "0", sandbox.container_name)
            if sandbox.mode == SANDBOX_MODE_VM
            else ()
        )
        return _AgentTarget(sandbox, prefix)

    # --- errors ---

    def _exec_error(self, msg: str) -> RunCodeExecutionError:
        return RunCodeExecutionError(f"run_code '{self.id}': {msg}")

    def _validation_error(self, msg: str) -> RunCodeValidationError:
        return RunCodeValidationError(f"run_code '{self.id}': {msg}")


# --- module helpers ---


def _member_names(universe: Any) -> list[str]:
    """Member names in get_file_artifacts' precedence, so preflight sees what execute stages."""
    if universe.file_artifact_refs:
        return list(universe.file_artifact_refs)
    return list(universe.file_artifact_ids or {})


def _is_safe_member(name: str) -> bool:
    """True when `name` stays inside the staging dir as a relative path."""
    if not name or name.startswith(("/", "~")):
        return False
    return ".." not in PurePosixPath(name).parts


class _ExecTarget(Protocol):
    async def reset_dir(self, path: str) -> None: ...
    async def write_file(self, path: str, content: str) -> None: ...
    async def run(self, *command: str) -> tuple[int, str, str]: ...


@dataclass
class _HostTarget:
    """Runs the script on the env's VM host (no agent)."""

    sandbox: Any

    async def reset_dir(self, path: str) -> None:
        await self.sandbox.exec_script(
            f"rm -rf {shlex.quote(path)} && mkdir -p {shlex.quote(path)}"
        )

    async def write_file(self, path: str, content: str) -> None:
        # base64 so any bytes survive the heredoc; exec_script writes as root.
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        await self.sandbox.exec_script(
            f"mkdir -p {shlex.quote(os.path.dirname(path))} && "
            f"base64 -d > {shlex.quote(path)} <<'AE_RUN_CODE_B64'\n{encoded}\nAE_RUN_CODE_B64"
        )

    async def run(self, *command: str) -> tuple[int, str, str]:
        # Files are written as root, so run as root too.
        return await self.sandbox.exec_with_output("sudo", *command)


@dataclass
class _AgentTarget:
    """Runs the script inside the agent container."""

    sandbox: Any
    prefix: tuple[str, ...]

    async def reset_dir(self, path: str) -> None:
        await self.run("rm", "-rf", path)
        await self.run("mkdir", "-p", path)

    async def write_file(self, path: str, content: str) -> None:
        # write_file_from_text takes (content, path); we expose (path, content).
        await self.sandbox.write_file_from_text(content, path)

    async def run(self, *command: str) -> tuple[int, str, str]:
        return await self.sandbox.exec_with_output(*self.prefix, *command)
