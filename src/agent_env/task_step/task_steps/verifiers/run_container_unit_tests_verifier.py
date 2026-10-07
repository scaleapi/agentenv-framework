"""Run a unit-tests-style verifier inside a deployed task container.

Optionally runs setup_commands first (to install verifier dependencies the
agent's container didn't have). The main command's exit code, stdout/stderr,
and any extracted result files land on context.metadata["verifications"].

stdout and stderr are each persisted as their own FileArtifact in S3 (so the
verification entry stays small and the full output is addressable via the
artifact API). Result files declared via `result_paths` are docker-cp'd out of
the container and stored inline in `extracted_files` (parsed JSON when valid).

The command and env are resolved per run from ``step_overrides`` and
``metadata["seed"]`` (see ``_resolve_command``), so one stored task can be pointed
at a different input without a new task version.

A non-zero exit (or timeout) raises after the entry has been recorded — so the
run instance has the verification context even when the verifier aborts the
task. Verifiers that always exit 0 and encode pass/fail inside a result JSON
(Harbor-style test.sh) will always look "passed" at the step level; downstream
consumers must read `extracted_files` to grade.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import tempfile
import time
import uuid
from typing import Any, ClassVar, Optional

from agent_env.providers.sandbox_providers.sandbox_provider import (
    all_sandbox_container_env,
    registered_sandbox_provider_classes,
)
from agent_env.store.ids import derive_id, is_local_id, key_segment, validate_local_id
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

# How many characters of stdout/stderr to inline on the verification entry as a
# preview. Full output lives in the FileArtifact uploaded to S3; this is just a
# convenience for `agent-env task run` / hub UI to show a head without an S3
# round-trip.
_OUTPUT_PREVIEW_CHARS = 500

# Every env var key — stored, override or seed — reaches the `docker exec` string
# unquoted (`-e 'K=v'` and `-e K=v` differ to docker), so it must be a POSIX name
# first. `\Z` not `$`: `$` also matches before a trailing newline, so `"URL\n"` would
# pass and split the command.
_SAFE_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\Z")

# Names execute() injects before the seed overlay lands. A CSV column that upper-cases
# onto one of these is almost certainly accidental, and would redirect the verifier's
# LLM traffic or swap its credentials. An explicit step_params override still wins —
# that one is deliberate. Keep in sync with the injection block in execute(); the
# sandbox platforms' own names come from their providers, so a new provider
# does not need this list edited.
_RESERVED_ENV_KEYS = frozenset({
    "LITELLM_API_KEY", "ANTHROPIC_API_KEY",
    "LITELLM_BASE_URL", "ANTHROPIC_BASE_URL",
    "PATH", "HOME",
})


class RunContainerUnitTestsVerifierTaskStep(TaskStep):
    type: ClassVar[str] = "run_container_unit_tests_verifier"
    entity_refs = ()

    def __init__(
        self,
        id: str,
        version: Optional[int],
        sandbox_name: str,
        container_name: str,
        command: str,
        verifier_id: Optional[str] = None,
        setup_commands: Optional[list[str]] = None,
        user: str = "root",
        timeout_sec: int = 300,
        env_vars: Optional[dict[str, str]] = None,
        result_paths: Optional[list[str]] = None,
        reward_path: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.sandbox_name = sandbox_name
        self.container_name = container_name
        self.command = command
        self.verifier_id = verifier_id or id
        self.setup_commands = list(setup_commands or [])
        self.user = user
        self.timeout_sec = timeout_sec
        self.env_vars = dict(env_vars or {})
        for key in self.env_vars:
            if not _SAFE_ENV_KEY.match(key):
                raise ValueError(
                    f"env_vars key {key!r} is not a POSIX name ([A-Za-z_][A-Za-z0-9_]*)"
                )
        self.result_paths = list(result_paths or [])
        # When set, the recorded score is the float read from this extracted
        # result file (a Harbor-style reward, 0.0–1.0) instead of the process
        # exit-code surrogate — a test.sh that exits 0 while writing reward 0
        # must NOT be recorded as a pass. Must also appear in `result_paths`.
        self.reward_path = reward_path
        if reward_path and reward_path not in self.result_paths:
            self.result_paths.append(reward_path)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["sandbox_name"] = self.sandbox_name
        base["container_name"] = self.container_name
        base["command"] = self.command
        base["verifier_id"] = self.verifier_id
        base["setup_commands"] = self.setup_commands
        base["user"] = self.user
        base["timeout_sec"] = self.timeout_sec
        base["env_vars"] = self.env_vars
        base["result_paths"] = self.result_paths
        base["reward_path"] = self.reward_path
        return base

    @classmethod
    def from_dict(cls, data: dict) -> RunContainerUnitTestsVerifierTaskStep:
        return cls(
            **cls._base_from_dict(data),
            sandbox_name=data["sandbox_name"],
            container_name=data["container_name"],
            command=data["command"],
            verifier_id=data.get("verifier_id"),
            setup_commands=data.get("setup_commands"),
            user=data.get("user", "root"),
            timeout_sec=data.get("timeout_sec", 300),
            env_vars=data.get("env_vars"),
            result_paths=data.get("result_paths"),
            reward_path=data.get("reward_path"),
        )

    def _resolve_command(self, context: TaskStepContext) -> tuple[str, dict[str, str]]:
        """Resolve ``(command, extra_env)`` for this run, so a value can vary without a
        new task version: ``step_params`` for a targeted or resumed run, ``seed`` for
        ``task run-batch --seeds`` fan-out. Both are additive to the stored ``env_vars``.

        Seed values reach the container as env vars rather than by string substitution,
        because ``-e K=<quoted>`` is a separate docker argument and bash expands ``"$K"``
        without re-parsing metacharacters. ``<key>`` placeholders are still rendered for
        rows that vary a flag rather than a value, in one left-to-right pass so
        substituted text is never re-scanned: replacing key-by-key, a value of ``<b>``
        quotes to ``'<b>'`` and a later key ``b`` rewrites the placeholder *inside* those
        quotes, yielding ``''$(id)''`` — quotes that close each other and leave the
        value bare for bash.
        """
        overrides = self.step_param_overrides(context)
        seed = context.metadata.get("seed") or {}

        command = overrides.get("command", self.command)
        if seed:
            placeholders = re.compile("|".join(re.escape(f"<{k}>") for k in seed))
            command = placeholders.sub(
                lambda m: shlex.quote(str(seed[m.group(0)[1:-1]])), command
            )

        extra_env = {}
        for key, value in (overrides.get("env_vars") or {}).items():
            if not _SAFE_ENV_KEY.match(key):
                # Raises where a seed key is skipped: an explicit override is a caller
                # bug, not one odd column of a batch.
                raise ValueError(
                    f"step_params env_vars key {key!r} is not a POSIX name "
                    f"([A-Za-z_][A-Za-z0-9_]*)"
                )
            extra_env[key] = str(value)
        reserved = _RESERVED_ENV_KEYS | {
            name for cls in registered_sandbox_provider_classes() for name in cls.CONTAINER_ENV
        }
        for key, value in seed.items():
            env_key = key.upper()
            if not _SAFE_ENV_KEY.match(env_key):
                # Still reachable via the <key> placeholder above; skipping keeps one
                # odd CSV column from failing a whole batch.
                logger.warning(
                    f"[{self.verifier_id}] seed key {key!r} is not a POSIX name; "
                    f"not exported as an env var"
                )
                continue
            if env_key in reserved:
                logger.warning(
                    f"[{self.verifier_id}] seed key {key!r} would overwrite {env_key}; "
                    f"not exported as an env var"
                )
                continue
            extra_env[env_key] = str(value)
        return command, extra_env

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.artifact.artifacts.file import FileArtifact
        from agent_env.providers.sandbox_providers.sandbox_provider import (
            build_sandbox_provider,
            get_sandbox_provider,
        )
        from agent_env.config import ConfigError, get_config

        # The outputs are named after the run: refuse an id they can't take before the tests run.
        if context.instance_id and is_local_id(context.instance_id):
            validate_local_id(derive_id(context.instance_id, f"verifier-stdout-{self.id}"))

        # 1. Lookup container + sandbox, rehydrate VmSandbox
        containers = context.metadata.get("deployed_docker_containers", [])
        container_entry = next(
            (
                c for c in containers
                if c.get("container_name") == self.container_name
                and c.get("sandbox_name") == self.sandbox_name
            ),
            None,
        )
        if container_entry is None:
            raise RuntimeError(
                f"Container '{self.container_name}' not found on sandbox '{self.sandbox_name}' in "
                f"context.metadata['deployed_docker_containers']"
            )
        ds = next((s for s in context.deployed_sandboxes if s.sandbox_name == self.sandbox_name), None)
        if ds is None:
            raise RuntimeError(f"Sandbox '{self.sandbox_name}' not found in context.deployed_sandboxes")

        provider = (
            build_sandbox_provider(ds.sandbox_type) if ds.sandbox_type else get_sandbox_provider()
        )
        sandbox = await provider.get_sandbox(ds.sandbox_id)

        # 2. Build the merged env (auto-injected from agent-env config + user overlay)
        config = get_config()
        merged_env: dict[str, str] = {}
        try:
            litellm_key = config.get_litellm_api_key()
            merged_env["LITELLM_API_KEY"] = litellm_key
            merged_env["ANTHROPIC_API_KEY"] = litellm_key
        except (KeyError, ConfigError):
            logger.warning("LITELLM_API_KEY unavailable; verifier won't have LLM creds")
        try:
            litellm_url = config.get_litellm_base_url()
            merged_env["LITELLM_BASE_URL"] = litellm_url
            merged_env["ANTHROPIC_BASE_URL"] = litellm_url
        except (KeyError, ConfigError):
            logger.warning("LITELLM_BASE_URL unavailable; verifier won't have LLM base URL")
        merged_env.update(all_sandbox_container_env())
        command, extra_env = self._resolve_command(context)
        merged_env.update(self.env_vars)
        merged_env.update(extra_env)
        env_flags = " ".join(f"-e {k}={shlex.quote(v)}" for k, v in merged_env.items())

        # 3. Run setup_commands (fail loud)
        for i, setup_cmd in enumerate(self.setup_commands, 1):
            logger.info(
                f"[{self.verifier_id}] setup [{i}/{len(self.setup_commands)}]: "
                f"{setup_cmd.splitlines()[0][:140]}"
            )
            await sandbox.exec_script(
                f"docker exec -u {shlex.quote(self.user)} {env_flags} "
                f"{shlex.quote(self.container_name)} bash -c {shlex.quote(setup_cmd)}"
            )

        # 4. Run main command, bounded by in-container timeout(1)
        logger.info(
            f"[{self.verifier_id}] running verifier (timeout={self.timeout_sec}s) "
            f"in container '{self.container_name}': {command[:160]}"
        )
        wrapped = (
            f"docker exec -u {shlex.quote(self.user)} {env_flags} "
            f"{shlex.quote(self.container_name)} "
            f"timeout --kill-after=10 {self.timeout_sec} bash -c {shlex.quote(command)}"
        )
        exit_code, stdout, stderr = await sandbox.exec_with_output("sudo", "bash", "-c", wrapped)
        timed_out = exit_code in (124, 137)
        logger.info(
            f"[{self.verifier_id}] exit_code={exit_code} timed_out={timed_out} "
            f"stdout_chars={len(stdout)} stderr_chars={len(stderr)}"
        )
        if stdout:
            logger.info(f"[{self.verifier_id}] stdout:\n{stdout}")
        if stderr:
            logger.info(f"[{self.verifier_id}] stderr:\n{stderr}")

        # 5. Extract result files (docker cp + cat + try JSON parse)
        extracted_files: dict[str, Any] = {}
        for result_path in self.result_paths:
            try:
                content = await self._extract_file(sandbox, result_path)
                try:
                    extracted_files[result_path] = json.loads(content)
                except json.JSONDecodeError:
                    extracted_files[result_path] = content
            except Exception as e:
                logger.warning(f"[{self.verifier_id}] failed to extract {result_path}: {e}")
                extracted_files[result_path] = None

        # 7. Persist stdout + stderr as separate FileArtifacts
        # Every execution gets its own key: a retried step keeps its run's instance id, and the
        # outputs are write-once. The instance id and the time only group and order them.
        stamp = f"{int(time.time())}-{uuid.uuid4().hex[:12]}"
        run = "-".join(filter(None, (context.instance_id and key_segment(context.instance_id), stamp)))
        base = context.instance_id or f"adhoc-{uuid.uuid4().hex[:12]}"
        store = config.get_object_store()
        outputs = f"{config.get_artifact_key_prefix()}verifier-outputs/{self.id}/{run}"
        stdout_artifact = await asyncio.to_thread(
            self._upload_text_artifact,
            text=stdout,
            artifact_id=derive_id(base, f"verifier-stdout-{self.id}-{stamp}"),
            description=f"stdout of {self.verifier_id} from step {self.id}",
            object_url=store.object_url(f"{outputs}/stdout.txt"),
        )
        stderr_artifact = await asyncio.to_thread(
            self._upload_text_artifact,
            text=stderr,
            artifact_id=derive_id(base, f"verifier-stderr-{self.id}-{stamp}"),
            description=f"stderr of {self.verifier_id} from step {self.id}",
            object_url=store.object_url(f"{outputs}/stderr.txt"),
        )

        # 8. Record on context (shape compatible with aggregate_verifiers).
        # Harbor-style verifiers (reward_path set) grade by the reward file, not
        # the exit code: test.sh exits 0 and encodes pass/fail in the reward.
        passed_surrogate = exit_code == 0 and not timed_out
        if self.reward_path is not None:
            reward = self._parse_reward(extracted_files.get(self.reward_path))
            if reward is None:
                score = 0.0
                result_bool = False
                message = (
                    f"reward file {self.reward_path!r} missing or non-numeric "
                    f"(exit_code={exit_code} timed_out={timed_out})"
                )
            else:
                score = reward
                result_bool = reward >= 1.0
                message = f"reward={reward} (exit_code={exit_code} timed_out={timed_out})"
            result_id = "reward"
        else:
            score = 1.0 if passed_surrogate else 0.0
            result_bool = passed_surrogate
            message = f"exit_code={exit_code} timed_out={timed_out}"
            result_id = "exit_code"
        context.metadata.setdefault("verifications", {})[self.verifier_id] = {
            "results": [
                {
                    "id": result_id,
                    "score": score,
                    "result": result_bool,
                    "message": message,
                },
            ],
            "score": score,
            "command": command,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "stdout_artifact": {
                "id": stdout_artifact.id,
                "version": stdout_artifact.version,
                "s3_url": stdout_artifact.object_url,
            },
            "stderr_artifact": {
                "id": stderr_artifact.id,
                "version": stderr_artifact.version,
                "s3_url": stderr_artifact.object_url,
            },
            "stdout_head": stdout[:_OUTPUT_PREVIEW_CHARS],
            "stderr_head": stderr[:_OUTPUT_PREVIEW_CHARS],
            "extracted_files": extracted_files,
        }

        # 9. Raise on timeout / non-zero exit AFTER recording. In reward mode a
        # non-zero exit is a graded failure (already recorded as score 0), not a
        # step error — only an unfinished (timed-out) run is unreliable.
        if timed_out:
            raise RuntimeError(
                f"Verifier '{self.verifier_id}' timed out after {self.timeout_sec}s"
            )
        if exit_code != 0 and self.reward_path is None:
            raise RuntimeError(
                f"Verifier '{self.verifier_id}' exited {exit_code}; "
                f"stderr (last 1500 chars):\n{stderr[-1500:]}"
            )

        return context

    @staticmethod
    def _parse_reward(raw: Any) -> Optional[float]:
        """Coerce an extracted reward file's content to a float in [0, 1], or None.

        ``raw`` is whatever ``extracted_files`` holds — a JSON-parsed number
        (``0``/``1``/``0.5``) or a raw string (``"0\\n"``). Returns None when the
        value is missing or non-numeric so the caller can record a hard 0.
        """
        if raw is None or isinstance(raw, bool):
            return None
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, value))

    @staticmethod
    def _upload_text_artifact(text: str, artifact_id: str, description: str, object_url: str):
        from agent_env.artifact.artifacts.file import FileArtifact

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write(text)
            tmp_path = tmp.name
        try:
            return FileArtifact.put_at(
                id=artifact_id,
                description=description,
                file_path=tmp_path,
                object_url=object_url,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def _extract_file(self, sandbox, path_in_container: str) -> str:
        """`docker cp` a file out of the container, read it from the VM, return text."""
        vm_temp = f"/tmp/_verifier_out_{uuid.uuid4().hex[:8]}"
        await sandbox.docker_cp(f"{self.container_name}:{path_in_container}", vm_temp)
        try:
            exit_code, stdout, stderr = await sandbox.exec_with_output("sudo", "cat", vm_temp)
            if exit_code != 0:
                raise RuntimeError(f"cat {vm_temp} failed (exit {exit_code}): {stderr}")
            return stdout
        finally:
            await sandbox.exec_script(f"rm -f {shlex.quote(vm_temp)}")
