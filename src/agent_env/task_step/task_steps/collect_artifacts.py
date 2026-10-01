"""Collect artifact files from an agent's sandbox VM and upload to S3.

Reads file paths from the VM, uploads each to S3 via the artifact store,
and stores the resulting S3 URIs in context.metadata["artifacts"].

File-access modes:
  - Default (no `env_id`): read files via `docker exec` into the deployed
    agent's container on a VM-mode sandbox.
  - VM host (`sandbox_name` alone): read straight off the VM, for a host-mode agent.
  - CUA (`env_id` set): read files via the CUA controller's `cua_get_file`
    over the gateway's step/v1, as the deployed env's stored card declares it.
    Required for CUA tasks, where deliverables live on the desktop VM (e.g.
    /home/docker/Desktop) which is NOT mounted into the a2a `agent-api`
    container — so the docker-exec path reports "File not found".

Artifact paths come from one of these sources (in priority order):
  1. Manifest (`manifest_step_id`): a prior prompt step's declared artifacts from
     that step's PromptResponse.structured_output["artifacts"] (logical name -> abs path);
     collected files are keyed by logical name.
  2. Seed data: context.metadata["seed"][artifacts_key] — a comma-separated
     list of filenames (e.g., "Dockerfile,run.sh,codebase.patch")
  3. Static fallback: artifact_paths list in the step definition

Each entry may be relative (joined to base_path) OR absolute (a leading '/',
read verbatim). Absolute entries let a task collect deliverables that aren't
under a single base_path — outputs aren't guaranteed to live on the desktop /
in /app — without assuming a common directory. base_path only applies to
relative entries and to the no-list enumeration fallback.

This allows the same step definition to collect different artifacts per seed.

Usage in task JSON:
    {
        "id": "collect",
        "type": "collect_artifacts",
        "agent_name": "solver",
        "base_path": "/app/artifact",
        "artifacts_key": "expected_artifacts"
    }

With seed CSV:
    name,platform,expected_artifacts
    VPC Endpoints,Terraform,"Dockerfile,run.sh,codebase.patch,test.patch,gold.patch,prompt.md,before.json,after.json,p2p_tests.json,f2p_tests.json,validation.json,progress.md"

Or with static fallback (no seed data needed):
    {
        "id": "collect",
        "type": "collect_artifacts",
        "agent_name": "solver",
        "base_path": "/app/artifact",
        "artifact_paths": ["Dockerfile", "run.sh", "codebase.patch"]
    }

CUA task (read from the desktop VM via the controller, absolute paths so no
base directory is assumed):
    {
        "id": "collect-artifacts",
        "type": "collect_artifacts",
        "env_id": "ubuntu-cua",
        "artifact_paths": [
            "/home/docker/Desktop/report.pdf",
            "/tmp/output/summary.docx"
        ]
    }
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import os
import json
import re
import shlex
import tempfile
import time
from typing import ClassVar, Optional

from agent_env.config import get_config
from agent_env.env.gateway.constants import EXT_STEP_URI
from agent_env.store.ids import derive_id, is_local_id, validate_local_id
from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.thread_work import finish_on_thread

logger = logging.getLogger(__name__)

# Stamped into DeployedEnv.metadata only by a desktop-VM env's deploy — used to confirm
# a step's `env_id` resolved to a CUA env before issuing cua_* controller calls.
_CUA_ENV_METADATA_MARKER = "cua_vm_sandbox_id"

# artifact_paths entries with a URL scheme are not VM files — they're gold/source
# URLs that downstream consumers (evaluator, artifact previewer) fetch directly.
# Collect skips them rather than joining them to base_path (which would mangle
# them to `/app/artifact/https://…` and fail the VM read).
_URL_ENTRY = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _is_url_entry(entry: str) -> bool:
    return isinstance(entry, str) and _URL_ENTRY.match(entry) is not None

_CONTENT_TYPES = {
    ".json": "application/json",
    ".md": "text/markdown",
    ".sh": "text/x-shellscript",
    ".patch": "text/x-diff",
    ".diff": "text/x-diff",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".ics": "text/calendar",
    ".eml": "message/rfc822",
}


def _sanitize_artifact_id(value: str) -> str:
    """Artifact ids become S3 key components, so spaces and separators can't survive."""
    return value.replace(" ", "-").replace("/", "_")


def _exec_args(sandbox, container: Optional[str], cmd: tuple) -> tuple:
    """Wrap a command for where it has to run: inside the container, on the VM host as root, or as-is."""
    from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM
    if sandbox.mode != SANDBOX_MODE_VM:
        return cmd  # container-mode sandboxes are the runtime and already root
    if container is not None:
        return ("sudo", "docker", "exec", container, *cmd)
    return ("sudo", *cmd)  # host-mode agents write as uid 0, so read them back as root


def _cua_bash_stdout(text: str) -> str:
    """Unwrap `cua_bash`'s JSON envelope down to the command's stdout.

    The CUA MCP tools do NOT share a return shape: `cua_get_file` answers with a
    bare base64 body, but `cua_bash` answers with
    `{"output": ..., "error": ..., "returncode": ..., "status": ...}`. Treating
    the latter like the former makes the whole JSON blob look like a single
    filename, and collection then tries to read
    `<base_path>/{"error": "", "output": "report.pdf\\n", ...}` and 404s — with
    the real filename visible inside the path it failed on.

    Tolerates a plain-stdout response so a controller that stops wrapping does
    not break this in the other direction.
    """
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return text
    if not isinstance(payload, dict) or "output" not in payload:
        return text
    return payload.get("output") or ""


def _remove(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _write_and_upload(store, content: bytes, artifact_id: str, version: int, object_name: str, content_type: str) -> str:
    """Stage ``content`` in a temp file, upload it as a collected artifact and remove the file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"-{object_name.replace('/', '_')}") as tmp:
        tmp.write(content)
    try:
        return store.put_object_file(
            artifact_type="collected_artifacts",
            id=artifact_id,
            version=version,
            object_name=object_name,
            file_path=tmp.name,
            content_type=content_type,
        )
    finally:
        _remove(tmp.name)


class CollectArtifactsTaskStep(TaskStep):
    type: ClassVar[str] = "collect_artifacts"
    entity_refs = (EntityRef.env("env_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        agent_name: Optional[str] = None,
        base_path: str = "/app/artifact",
        artifacts_key: str = "expected_artifacts",
        artifact_paths: Optional[list[str]] = None,
        env_id: Optional[str] = None,
        manifest_step_id: Optional[str] = None,
        exclude_basenames: Optional[list[str]] = None,
        sandbox_name: Optional[str] = None,
        container_name: Optional[str] = None,
        universe_id_suffix: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        # Three mutually exclusive collection modes: `agent_name` (agent container), `env_id` (CUA
        # controller) and `container_name` (a plain container started by run_docker_container on a
        # deploy_sandbox sandbox). A defaulted agent_name is tolerated: older serialized CUA steps
        # persist DEFAULT_AGENT_NAME alongside env_id.
        explicit_agent = agent_name not in (None, TaskStep.DEFAULT_AGENT_NAME)
        if env_id is not None and explicit_agent:
            raise ValueError(
                "collect_artifacts: set either `agent_name` (agent-container path) "
                "or `env_id` (CUA controller path), not both"
            )
        if sandbox_name is not None and container_name is None and (env_id is not None or explicit_agent):
            raise ValueError(
                "collect_artifacts: `sandbox_name` alone (VM-host path) is exclusive with "
                "`agent_name` and `env_id`"
            )
        if container_name is not None:
            if env_id is not None or explicit_agent:
                raise ValueError(
                    "collect_artifacts: `container_name` (sandbox-container path) is exclusive with "
                    "`agent_name` and `env_id`"
                )
            if not sandbox_name:
                # The lookup key in `deployed_docker_containers` is the pair, and a container name
                # alone is not unique across sandboxes.
                raise ValueError(
                    "collect_artifacts: `container_name` requires `sandbox_name`"
                )
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        # Appended to the run's artifact id so two collect steps in one run produce two
        # named artifacts instead of two versions of one. A suffix rather than an absolute
        # id because the latter would be identical across every seed in a batch.
        self.universe_id_suffix = universe_id_suffix
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME
        self.base_path = base_path.rstrip("/")
        self.artifacts_key = artifacts_key
        self.artifact_paths = artifact_paths or []
        self.env_id = env_id
        self.manifest_step_id = manifest_step_id
        self.exclude_basenames = set(exclude_basenames or [])
        self.sandbox_name = sandbox_name
        self.container_name = container_name

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["agent_name"] = self.agent_name
        base["base_path"] = self.base_path
        base["artifacts_key"] = self.artifacts_key
        base["artifact_paths"] = self.artifact_paths
        base["env_id"] = self.env_id
        base["manifest_step_id"] = self.manifest_step_id
        base["exclude_basenames"] = sorted(self.exclude_basenames)
        base["sandbox_name"] = self.sandbox_name
        base["container_name"] = self.container_name
        base["universe_id_suffix"] = self.universe_id_suffix
        return base

    @classmethod
    def from_dict(cls, data: dict) -> CollectArtifactsTaskStep:
        return cls(
            **cls._base_from_dict(data),
            agent_name=data.get("agent_name"),
            base_path=data.get("base_path", "/app/artifact"),
            artifacts_key=data.get("artifacts_key", "expected_artifacts"),
            artifact_paths=data.get("artifact_paths"),
            env_id=data.get("env_id"),
            manifest_step_id=data.get("manifest_step_id"),
            exclude_basenames=data.get("exclude_basenames"),
            sandbox_name=data.get("sandbox_name"),
            container_name=data.get("container_name"),
            universe_id_suffix=data.get("universe_id_suffix"),
        )

    def _resolve_filenames(self, context: TaskStepContext) -> list[str]:
        """Determine which filenames to collect, from seed data or static config."""
        seed = context.metadata.get("seed", {})
        seed_value = seed.get(self.artifacts_key, "")
        if seed_value:
            return [f.strip() for f in str(seed_value).split(",") if f.strip()]
        return self.artifact_paths

    def _configured_entries(self, context: TaskStepContext) -> list[str]:
        """The artifact_paths entries in effect: a re-run override wins over seed data and
        static config."""
        override = self.step_param_overrides(context)
        if "artifact_paths" in override:
            raw = override["artifact_paths"]
            return [raw] if isinstance(raw, str) else [str(p) for p in raw]
        return self._resolve_filenames(context)

    def _resolve_items(self, context: TaskStepContext) -> list[tuple[str, str, str]]:
        """``(key, source_path, object_name)`` triples to collect — keyed by logical name in
        manifest mode, by the artifact_paths entry otherwise."""
        # A re-run-from-step can override artifact_paths via step_overrides on
        # the run; it wins over manifest/seed/static so a corrected path actually takes effect.
        if "artifact_paths" in self.step_param_overrides(context):
            paths = self._configured_entries(context)
            logger.info(f"collect '{self.id}': using {len(paths)} overridden artifact_path(s)")
            return self._items_from_entries(paths)
        if self.manifest_step_id:
            # reversed: if the step re-ran (retry/resume), the latest response wins.
            response = next(
                (r for r in reversed(context.prompt_responses) if r.step_id == self.manifest_step_id), None
            )
            structured = response.structured_output if response else None
            if structured is None:
                # Transitional: pre-DataPart-worker tasks carry it in the old metadata channel.
                structured = (context.metadata.get("structured_outputs") or {}).get(self.manifest_step_id)
            manifest = (structured.get("artifacts") if isinstance(structured, dict) else None) or {}
            items: list[tuple[str, str, str]] = []
            skipped_nonfile: list[str] = []
            for name, path in manifest.items():
                if not (isinstance(path, str) and path.startswith("/")):
                    skipped_nonfile.append(name)  # literal value (e.g. coverage_state) or relative path
                    continue
                if "*" in path:
                    continue  # unresolved glob (e.g. per-iteration debug logs) — can't fetch a literal *
                if os.path.basename(path) in self.exclude_basenames:
                    continue
                rel = (
                    path[len(self.base_path) + 1:]
                    if path.startswith(self.base_path + "/")
                    else os.path.basename(path)
                )
                items.append((name, path, rel))
            if skipped_nonfile:
                logger.debug(
                    "collect(manifest=%s): skipped non-absolute manifest entries "
                    "(literal values or relative paths, not collected): %s",
                    self.manifest_step_id, skipped_nonfile,
                )
            return items
        return self._items_from_entries(self._configured_entries(context))

    def _items_from_entries(self, entries: list[str]) -> list[tuple[str, str, str]]:
        """Resolve raw artifact_paths entries into (key, source_path, object_name)
        triples, skipping URL-scheme entries (gold/source URLs fetched directly by
        consumers, not collected from the VM)."""
        items: list[tuple[str, str, str]] = []
        skipped_urls: list[str] = []
        for entry in entries:
            if _is_url_entry(entry):
                skipped_urls.append(entry)
                continue
            items.append((entry, *self._resolve_paths(entry)))
        if skipped_urls:
            logger.info(
                "collect '%s': skipping %d URL entr%s (not VM files; fetched from source by "
                "downstream consumers): %s",
                self.id, len(skipped_urls), "y" if len(skipped_urls) == 1 else "ies", skipped_urls,
            )
        return items

    def _drop_excluded(self, names: list[str]) -> list[str]:
        """Drop names whose basename is in exclude_basenames."""
        if not self.exclude_basenames:
            return names
        return [f for f in names if os.path.basename(f) not in self.exclude_basenames]

    async def _list_base_directory(self, sandbox, container: Optional[str]) -> list[str]:
        """Recursively enumerate regular files under base_path.
        Returns paths relative to base_path (e.g. ``"sub/file.txt"``).
        Used as the fallback when no explicit filename list was configured.

        Branches on `sandbox.mode`: VM-mode sandboxes wrap `find` in
        `docker exec <container>`; container-mode sandboxes (Modal, for one)
        ARE the agent runtime so we run `find` directly."""
        args = _exec_args(sandbox, container,
                          ("find", self.base_path, "-type", "f", "-printf", "%P\n"))
        exit_code, stdout, stderr = await sandbox.exec_with_output(*args)
        if exit_code != 0:
            logger.warning(f"Failed to enumerate {self.base_path}: {stderr[:300]}")
            return []
        return [f.strip() for f in stdout.splitlines() if f.strip()]

    async def _list_top_level(self, sandbox, container: Optional[str]) -> list[str]:
        """Bounded top-level listing of base_path for the 'what was produced' hint on a failed collect."""
        args = _exec_args(sandbox, container,
                          ("find", self.base_path, "-mindepth", "1", "-maxdepth", "1", "-printf", "%P\\n"))
        exit_code, stdout, _ = await sandbox.exec_with_output(*args)
        if exit_code != 0:
            return []
        return sorted(f.strip() for f in stdout.splitlines() if f.strip())[:100]

    async def _resolve_live_sandbox(self, provider, sandbox_id: str):
        """Resolve the sandbox and confirm it's reachable. A re-run-from-step can target one already
        reaped past its TTL, so fail with a clear "expired" message rather than a raw provider error."""
        err: Optional[Exception] = None
        try:
            sandbox = await provider.get_sandbox(sandbox_id)
            exit_code, _, _ = await sandbox.exec_with_output("echo", "ok")
            if exit_code == 0:
                return sandbox
        except Exception as e:
            err = e
        # Log the real error — TTL expiry is the common cause but not the only one (auth, quota, network).
        logger.warning(f"collect: sandbox {sandbox_id!r} unreachable{f' ({err})' if err else ''}")
        await provider.close()
        raise RuntimeError(
            f"collect: agent sandbox {sandbox_id!r} is unreachable (likely expired past its TTL) — "
            f"a re-run-from-step needs the original live sandbox; re-run the full task."
        ) from err

    async def _discover_container(self, sandbox) -> str:
        """Find the agent container running on the VM.

        The A2A agent deploy hardcodes the container name to 'agent-api'
        (see agent_env/a2a_agent/a2a_agent.py). We also accept any
        'a2a-agent-*' container as a fallback in case the naming scheme
        evolves.
        """
        exit_code, stdout, stderr = await sandbox.exec_with_output(
            "sudo", "docker", "ps",
            "--format", "{{.Names}}",
        )
        if exit_code != 0:
            raise RuntimeError(f"Failed to list containers: {stderr[:300]}")
        running = [n.strip() for n in stdout.splitlines() if n.strip()]
        # Prefer this sandbox's own container name, fall back to the a2a-agent-* prefix.
        if sandbox.container_name in running:
            return sandbox.container_name
        fallback = [n for n in running if n.startswith("a2a-agent-")]
        if fallback:
            if len(fallback) > 1:
                logger.warning(f"Multiple a2a-agent-* containers found; using first: {fallback}")
            return fallback[0]
        raise RuntimeError(
            f"No agent container found on the VM (looked for 'agent-api' or 'a2a-agent-*'). "
            f"Running containers: {running}. Has deploy_agent been run in this task?"
        )

    async def _get_file_size(self, sandbox, container: Optional[str], source_path: str) -> int:
        """Get file size on the agent's filesystem. Returns -1 if the file doesn't exist.
        VM-mode wraps in `docker exec <container>`; container-mode runs `stat` directly."""
        args = _exec_args(sandbox, container, ("stat", "-c", "%s", source_path))
        exit_code, stdout, _ = await sandbox.exec_with_output(*args)
        if exit_code != 0:
            return -1
        return int(stdout.strip())

    async def _collect_file(
        self, sandbox, container: Optional[str], source_path: str, object_name: str, store, artifact_id: str,
        version: int, content_type: str,
    ) -> str:
        """Read a file out of the sandbox and upload it to S3 — one path, any size.

        base64-encode in the sandbox (raw binary over a websocket exec transport is
        silently dropped; ASCII survives), stream stdout + decode incrementally to
        a temp file (bounds memory), then multipart-upload. stderr is drained in
        parallel so the shared output pump can't backpressure-stall."""
        import base64
        bash_cmd = f"base64 < {shlex.quote(source_path)}"
        args = _exec_args(sandbox, container, ("bash", "-c", bash_cmd))

        local_path = None
        handed_off = False
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"-{object_name.replace('/', '_')}") as tmp:
                local_path = tmp.name

            process = await sandbox.exec(*args)
            stderr_task = asyncio.create_task(process.stderr.read())
            stdout = process.stdout
            with open(local_path, "wb") as fh:
                if hasattr(stdout, "__aiter__"):
                    # Async-iterable stdout (websocket exec): stream, decoding 4-char-aligned base64.
                    pending = bytearray()
                    async for chunk in stdout:
                        pending += chunk.translate(None, b"\r\n")  # strip 76-col wrap
                        n = (len(pending) // 4) * 4
                        if n:
                            fh.write(base64.b64decode(pending[:n]))
                            del pending[:n]
                    if pending:
                        fh.write(base64.b64decode(pending))
                else:
                    # Modal stdout has read() but no __aiter__: read all, decode once.
                    data = (await stdout.read()).translate(None, b"\r\n")
                    fh.write(base64.b64decode(data))
            exit_code = await process.wait()
            try:
                await stderr_task
            except Exception:
                pass
            if exit_code != 0:
                raise RuntimeError(f"base64 read failed for {source_path} (exit {exit_code})")

            # Integrity check: base64-over-exec can silently truncate large streams on a
            # websocket exec transport under load. Fail loudly on a size mismatch rather than upload a
            # truncated artifact (retryable failure > undetectable short file).
            src_size = await self._get_file_size(sandbox, container, source_path)
            got_size = os.path.getsize(local_path)
            if src_size >= 0 and got_size != src_size:
                raise RuntimeError(
                    f"collect integrity check failed for {source_path}: decoded {got_size} bytes but "
                    f"source is {src_size} (exec-channel truncation) — refusing to upload a corrupt artifact"
                )

            path = local_path

            def upload() -> str:
                try:
                    return store.put_object_file(
                        artifact_type="collected_artifacts",
                        id=artifact_id,
                        version=version,
                        object_name=object_name,
                        file_path=path,
                        content_type=content_type,
                    )
                finally:
                    _remove(path)

            # The upload owns the file from here: a cancel must not remove it under the upload.
            handed_off = True
            return await finish_on_thread(
                upload, f"Uploading collected {source_path}", if_never_run=lambda: _remove(path)
            )
        finally:
            if local_path and not handed_off:
                _remove(local_path)

    async def _controller_call(self, deployed_env, tool_name: str, arguments: dict) -> str:
        """Call a CUA MCP tool through the gateway's step/v1, as the env card declares it. Returns the text content."""
        data = await deployed_env.invoke(
            EXT_STEP_URI, "step", {"action": "call_tool", "tool_name": tool_name, "arguments": arguments}, timeout=600,
        )
        if data.get("isError"):
            raise RuntimeError(f"{tool_name} failed: {data}")
        content = data.get("content")
        if not content:
            raise RuntimeError(f"{tool_name} returned no content: {data}")
        return content[0].get("text", "")

    async def _controller_get_file(self, deployed_env, path: str) -> bytes:
        """Fetch a file from the CUA VM as raw bytes (controller returns base64)."""
        import base64

        b64 = await self._controller_call(deployed_env, "cua_get_file", {"path": path})
        return base64.b64decode(b64)

    async def _controller_list_dir(self, deployed_env) -> list[str]:
        """Recursively enumerate regular files under base_path via cua_bash.
        Returns paths relative to base_path. Fallback when no explicit list
        was configured."""
        script = f"find {shlex.quote(self.base_path)} -type f -printf '%P\\n'"
        try:
            out = await self._controller_call(deployed_env, "cua_bash", {"script": script})
        except Exception as e:
            logger.warning(f"Failed to enumerate {self.base_path} via controller: {e}")
            return []
        return [f.strip() for f in _cua_bash_stdout(out).splitlines() if f.strip()]

    def _resolve_cua_env(self, context):
        """Resolve self.env_id to a deployed CUA env, refusing non-CUA envs.

        cua_get_file / cua_bash are CUA-controller tools; firing them at an
        arbitrary env's gateway would hang or error confusingly. A desktop-VM env's
        deploy is the only path that stamps `cua_vm_sandbox_id` into the
        DeployedEnv metadata, so its presence authoritatively marks a CUA env."""
        deployed_env = next((d for d in context.deployed_envs if d.env_id == self.env_id), None)
        if deployed_env is None:
            raise RuntimeError(
                f"Env '{self.env_id}' not found in context.deployed_envs — cannot collect artifacts via controller"
            )
        if _CUA_ENV_METADATA_MARKER not in (deployed_env.metadata or {}):
            raise RuntimeError(
                f"Env '{self.env_id}' is not a CUA environment "
                f"(no '{_CUA_ENV_METADATA_MARKER}' in its deployed metadata) — refusing to issue "
                f"cua_* controller calls against a non-CUA gateway. collect_artifacts with `env_id` "
                f"set is only valid for CUA envs; omit `env_id` to use the agent-container path."
            )
        # Up front: enumeration swallows errors into [], so a missing step/v1 would become a silent empty collect.
        deployed_env.require(EXT_STEP_URI, "step")
        return deployed_env

    def _resolve_paths(self, entry: str) -> tuple[str, str]:
        """Map an artifact_paths entry to (source_path_on_vm, s3_object_name).
        Absolute entries are read verbatim (base_path ignored) and stored under
        their full path; relative entries keep the legacy base_path join."""
        if entry.startswith("/"):
            return entry, entry.lstrip("/")
        return f"{self.base_path}/{entry}", entry

    async def _collect_via_controller(self, context, store, artifact_id, version):
        """Collect artifacts from a CUA env's desktop VM through the controller."""
        from agent_env.artifact import FileArtifact

        deployed_env = self._resolve_cua_env(context)
        # Read files via the CUA MCP server (cua_get_file / cua_bash) over the
        # gateway's step/v1. These are MCP tools registered on the gateway on both
        # Ubuntu and macOS — NOT endpoints on the macOS controller sidecar, whose
        # /step only accepts a ScaleCuaAction and 500s on a call_tool payload.
        logger.info(f"Collecting artifacts via CUA MCP server (env_id={self.env_id}, gateway={deployed_env.environment_url})")

        items = self._resolve_items(context)
        if not items:
            if self.manifest_step_id:
                raise RuntimeError(
                    f"collect(manifest): step '{self.manifest_step_id}' declared no collectable "
                    f"artifacts in its structured output — did it run and emit StructuredOutput.artifacts?"
                )
            if self._configured_entries(context):
                logger.info(f"collect '{self.id}': every configured entry is a URL; nothing to read from the VM")
                return {}, {}
            enumerated = self._drop_excluded(await self._controller_list_dir(deployed_env))
            if not enumerated:
                logger.warning(
                    f"No artifact filenames resolved and {self.base_path} is empty — nothing to collect"
                )
                return {}, {}
            items = [(f, f"{self.base_path}/{f}", f) for f in enumerated]
            logger.info(
                f"No manifest/artifacts_key/artifact_paths configured; enumerated "
                f"{len(items)} file(s) under {self.base_path}"
            )
        else:
            logger.info(f"Collecting {len(items)} artifacts from {self.base_path} via controller")

        collected: dict[str, str] = {}
        file_artifacts: dict[str, FileArtifact] = {}
        for key, source_path, object_name in items:
            try:
                content = await self._controller_get_file(deployed_env, source_path)
            except Exception as e:
                logger.warning(f"Failed to read {source_path} via controller: {e}")
                continue

            ext = os.path.splitext(object_name)[1]
            content_type = _CONTENT_TYPES.get(ext, "application/octet-stream")

            try:
                s3_url = await finish_on_thread(
                    functools.partial(
                        _write_and_upload, store, content, artifact_id, version, object_name, content_type,
                    ),
                    f"Uploading collected {source_path}",
                )

                collected[key] = s3_url
                self._register_file_artifact(
                    store, file_artifacts, artifact_id, key, object_name, content_type, s3_url, context,
                )
                logger.info(f"Collected {source_path} -> {s3_url} ({len(content)} bytes) via controller")
            except Exception as e:
                logger.warning(f"Failed to collect {source_path} via controller: {e}")

        logger.info(f"Collected {len(collected)}/{len(items)} artifacts via controller")
        return collected, file_artifacts

    async def _collect_via_agent_container(self, context, store, artifact_id, version):
        from agent_env.providers.sandbox_providers.sandbox_provider import (
            SANDBOX_MODE_VM,
            build_sandbox_provider,
            get_agent_sandbox_provider,
        )

        agent = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
        if agent is None:
            raise RuntimeError(f"Agent '{self.agent_name}' not found in context.deployed_agents")
        if not agent.sandbox_id:
            raise RuntimeError(f"Agent '{self.agent_name}' has no sandbox_id — cannot collect artifacts")

        items = self._resolve_items(context)
        if not items and self.manifest_step_id:
            raise RuntimeError(
                f"collect(manifest): step '{self.manifest_step_id}' declared no collectable "
                f"artifacts in its structured output — did it run and emit StructuredOutput.artifacts?"
            )

        provider = (
            build_sandbox_provider(agent.sandbox_type)
            if agent.sandbox_type
            else get_agent_sandbox_provider()
        )
        sandbox = await self._resolve_live_sandbox(provider, agent.sandbox_id)
        logger.info(f"Connected to sandbox {agent.sandbox_id} (mode={sandbox.mode})")

        container = await self._discover_container(sandbox) if sandbox.mode == SANDBOX_MODE_VM else None
        if container:
            logger.info(f"Using agent container: {container}")

        return await self._collect_items(
            provider, sandbox, container, items, context, store, artifact_id, version,
        )

    async def _collect_via_vm_host(self, context, store, artifact_id, version):
        """Collect off the VM's own filesystem — host-mode agents leave no container to exec into."""
        from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider, get_sandbox_provider

        ds = next(
            (sb for sb in context.deployed_sandboxes if sb.sandbox_name == self.sandbox_name), None
        )
        if ds is None:
            raise RuntimeError(
                f"Sandbox '{self.sandbox_name}' not found in context.deployed_sandboxes"
            )

        items = self._resolve_items(context)
        if not items and self.manifest_step_id:
            raise RuntimeError(
                f"collect(manifest): step '{self.manifest_step_id}' declared no collectable "
                f"artifacts in its structured output — did it run and emit StructuredOutput.artifacts?"
            )

        provider = build_sandbox_provider(ds.sandbox_type) if ds.sandbox_type else get_sandbox_provider()
        sandbox = await self._resolve_live_sandbox(provider, ds.sandbox_id)
        logger.info(f"Collecting from the VM host of sandbox '{self.sandbox_name}' ({ds.sandbox_id})")

        return await self._collect_items(
            provider, sandbox, None, items, context, store, artifact_id, version,
        )

    async def _collect_via_sandbox_container(self, context, store, artifact_id, version):
        """Collect from a plain container started by ``run_docker_container``.

        The agent path finds its container by discovery, and ``_discover_container`` only accepts
        ``agent-api`` / ``a2a-agent-*`` names — so a task that deploys a sandbox and runs an ordinary
        image had no way to get its files out. Here the (sandbox, container) pair is named
        explicitly and resolved exactly as ``load_artifact`` resolves it.
        """
        from agent_env.providers.sandbox_providers.sandbox_provider import (
            build_sandbox_provider,
            get_sandbox_provider,
        )

        containers = context.metadata.get("deployed_docker_containers", [])
        entry = next(
            (
                c for c in containers
                if c.get("container_name") == self.container_name
                and c.get("sandbox_name") == self.sandbox_name
            ),
            None,
        )
        if entry is None:
            raise RuntimeError(
                f"Container '{self.container_name}' not found on sandbox '{self.sandbox_name}' in "
                f"context.metadata['deployed_docker_containers']"
            )
        deployed = next(
            (s for s in context.deployed_sandboxes if s.sandbox_name == self.sandbox_name), None
        )
        if deployed is None:
            raise RuntimeError(
                f"Sandbox '{self.sandbox_name}' not found in context.deployed_sandboxes"
            )

        items = self._resolve_items(context)
        if not items and self.manifest_step_id:
            raise RuntimeError(
                f"collect(manifest): step '{self.manifest_step_id}' declared no collectable "
                f"artifacts in its structured output — did it run and emit StructuredOutput.artifacts?"
            )

        provider = (
            build_sandbox_provider(deployed.sandbox_type)
            if deployed.sandbox_type
            else get_sandbox_provider()
        )
        sandbox = await self._resolve_live_sandbox(provider, deployed.sandbox_id)
        logger.info(
            f"Connected to sandbox {deployed.sandbox_id} (mode={sandbox.mode}), "
            f"container={self.container_name}"
        )
        return await self._collect_items(
            provider, sandbox, self.container_name, items, context, store, artifact_id, version,
        )

    async def _collect_items(self, provider, sandbox, container, items, context, store,
                             artifact_id, version):
        """Pull ``items`` out of ``container`` (or the VM itself when None) and register them.

        Shared by the agent-container and sandbox-container paths — everything from here down
        needs only a reachable sandbox plus an optional container name.
        """
        from agent_env.artifact import FileArtifact

        if not items:
            if self._configured_entries(context):
                logger.info(f"collect '{self.id}': every configured entry is a URL; nothing to read from the VM")
                await provider.close()
                return {}, {}
            # Neither seed-driven nor static list was configured. Fall back
            # to enumerating base_path on the VM — useful when the agent's
            # output set is dynamic (e.g. "everything the solver wrote to
            # /app/artifact"). Recursive; returns paths relative to base_path.
            enumerated = self._drop_excluded(await self._list_base_directory(sandbox, container))
            if not enumerated:
                logger.warning(
                    f"No artifact filenames resolved and {self.base_path} is empty — nothing to collect"
                )
                await provider.close()
                return {}, {}
            items = [(f, f"{self.base_path}/{f}", f) for f in enumerated]
            logger.info(
                f"No manifest/artifacts_key/artifact_paths configured; enumerated "
                f"{len(items)} file(s) under {self.base_path}"
            )
        else:
            logger.info(f"Collecting {len(items)} artifacts from {self.base_path}")

        collected: dict[str, str] = {}
        file_artifacts: dict[str, FileArtifact] = {}
        for key, source_path, object_name in items:
            try:
                file_size = await self._get_file_size(sandbox, container, source_path)
                if file_size < 0:
                    logger.warning(f"File not found: {source_path}")
                    continue

                ext = os.path.splitext(object_name)[1]
                content_type = _CONTENT_TYPES.get(ext, "application/octet-stream")

                s3_url = await self._collect_file(
                    sandbox, container, source_path, object_name, store, artifact_id, version, content_type,
                )

                collected[key] = s3_url
                self._register_file_artifact(
                    store, file_artifacts, artifact_id, key, object_name, content_type, s3_url, context,
                )
                logger.info(f"Collected {source_path} -> {s3_url} ({file_size} bytes)")

            except Exception as e:
                logger.warning(f"Failed to collect {source_path}: {e}")

        if not collected and items:
            # Surface what the agent actually left in base_path so a re-collect is a pick, not a guess.
            context.metadata["collect_produced_files"] = {
                "step_id": self.id,
                "base_path": self.base_path,
                "files": await self._list_top_level(sandbox, container),
            }
        logger.info(f"Collected {len(collected)}/{len(items)} artifacts")
        await provider.close()
        return collected, file_artifacts

    def _register_file_artifact(self, store, file_artifacts, artifact_id, key, object_name, content_type, s3_url, context):
        """Register a FileArtifact document pointing at the already-uploaded S3
        object. We do NOT re-upload the bytes — FileArtifact.object_url is just a URL
        reference, and the object is already in S3 under the collected_artifacts
        path. `key` is the original artifact_paths entry (an absolute path when
        the file lives outside base_path); `object_name` is the S3 key. The
        FileArtifact filename is the basename for clean downloads."""
        from agent_env.artifact import FileArtifact

        if is_local_id(artifact_id):
            fa_id = derive_id(artifact_id, hashlib.sha256(object_name.encode("utf-8")).hexdigest()[:16])
        else:
            fa_id = f"{artifact_id}-{object_name}".replace("/", "_")[:200]
        fa_version = store.next_version(fa_id)
        fa = FileArtifact(
            id=fa_id,
            version=fa_version,
            description=f"Collected artifact '{object_name}' from task instance {context.instance_id or artifact_id}",
            filename=object_name.split("/")[-1],
            content_type=content_type,
            s3_url=s3_url,
        )
        try:
            store.put_document(fa)
            file_artifacts[key] = fa
        except Exception as doc_err:
            # Don't fail the whole collection if the document write fails.
            logger.warning(f"Failed to register FileArtifact for {key}: {doc_err}")

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.artifact import FileArtifactUniverse
        from agent_env.artifact.store import get_artifact_store

        store = get_artifact_store()
        artifact_id = (
            context.metadata.get("universe_id")
            or context.instance_id
            or "unknown"
        )
        # An @local id is kept whole, since key_segment bounds its object keys. Any other is capped at 100
        # chars, with room reserved for the suffix rather than appending past it, so a long run id can never
        # truncate two collects back onto the same name.
        suffix = _sanitize_artifact_id(self.universe_id_suffix or "")
        if is_local_id(artifact_id):
            artifact_id += suffix
            validate_local_id(artifact_id)
        else:
            artifact_id = _sanitize_artifact_id(artifact_id)[:100 - len(suffix)] + suffix
        get_config().check_local_run_write(artifact_id)
        version = int(time.time())

        if self.env_id:
            collected, file_artifacts = await self._collect_via_controller(context, store, artifact_id, version)
        elif self.container_name:
            collected, file_artifacts = await self._collect_via_sandbox_container(context, store, artifact_id, version)
        elif self.sandbox_name:
            collected, file_artifacts = await self._collect_via_vm_host(context, store, artifact_id, version)
        else:
            collected, file_artifacts = await self._collect_via_agent_container(context, store, artifact_id, version)

        # `artifacts` and `file_artifact_universe` (set below) are legacy flat keys
        # that mirror the latest collect step; they can be removed once all
        # consumers read `collected_artifacts[<step_id>]` instead.
        context.metadata["artifacts"] = collected
        collected_entry: dict = {"artifacts": collected, "file_artifact_universe": None}
        context.metadata.setdefault("collected_artifacts", {})[self.id] = collected_entry

        if file_artifacts:
            try:
                universe = FileArtifactUniverse.put(
                    id=artifact_id,
                    file_artifacts=file_artifacts,
                )
                universe_ref = {"id": universe.id, "version": universe.version}
                context.metadata["file_artifact_universe"] = universe_ref
                collected_entry["file_artifact_universe"] = universe_ref
                logger.info(
                    f"Created FileArtifactUniverse id={universe.id} v{universe.version} "
                    f"with {len(file_artifacts)} file(s)"
                )
            except Exception as u_err:
                logger.warning(f"Failed to create FileArtifactUniverse: {u_err}")

        # After the metadata write, so tolerant (fail_task_on_error=False) dependents still see
        # present-but-empty. Enumeration fallback resolves to empty, so it stays lenient.
        expected = self._resolve_items(context)
        if expected and collected_entry["file_artifact_universe"] is None:
            raise RuntimeError(
                f"collect: resolved {len(expected)} artifact path(s) but produced no file_artifact_universe "
                f"(see per-file warnings for the cause): {[src for _, src, _ in expected]}"
            )

        return context
