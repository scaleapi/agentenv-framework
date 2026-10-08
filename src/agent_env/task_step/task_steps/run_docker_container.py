"""Run a Docker container task step — builds a user-provided Dockerfile on a
pre-deployed VM sandbox and runs the resulting container in the background."""

from __future__ import annotations

import logging
import posixpath
import shlex
from typing import ClassVar, Optional
from urllib.parse import urlparse

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.providers.sandbox_providers.sandbox import SANDBOX_LABEL, port_bindings
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.utils.paths import validate_relative_filename

logger = logging.getLogger(__name__)


class RunDockerContainerTaskStep(TaskStep):
    type: ClassVar[str] = "run_docker_container"
    entity_refs = (
        EntityRef.artifact(
            "docker_context_artifact_id",
            version_field="docker_context_artifact_version",
            artifact_type="file_artifact_universe",
        ),
    )

    def __init__(
        self,
        id: str,
        version: Optional[int],
        sandbox_name: str,
        docker_context_artifact_id: Optional[str] = None,
        docker_context_artifact_version: Optional[int] = None,
        docker_context_url: Optional[str] = None,
        dockerfile_path: str = "Dockerfile",
        container_name: str = "task-container",
        image_tag: Optional[str] = None,
        ports: Optional[list[int]] = None,
        env_vars: Optional[dict[str, str]] = None,
        build_args: Optional[dict[str, str]] = None,
        command_override: Optional[str] = None,
        keep_alive_with_base_command: bool = False,
        network: Optional[str] = None,
        extra_networks: Optional[list[str]] = None,
        ready_command: Optional[str] = None,
        devices: Optional[list[str]] = None,
        cap_add: Optional[list[str]] = None,
        privileged: bool = False,
        volumes: Optional[list[str]] = None,
        shm_size: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        modes = sum(x is not None for x in (docker_context_artifact_id, docker_context_url))
        if modes != 1:
            raise ValueError(
                "Must pass exactly one of `docker_context_artifact_id` or `docker_context_url`"
            )
        if command_override and keep_alive_with_base_command:
            raise ValueError(
                "Pass either `command_override` or `keep_alive_with_base_command`, not both"
            )
        self.sandbox_name = sandbox_name
        self.docker_context_artifact_id = docker_context_artifact_id
        self.docker_context_artifact_version = docker_context_artifact_version
        self.docker_context_url = docker_context_url
        self.dockerfile_path = dockerfile_path
        self.container_name = container_name
        self.image_tag = image_tag
        self.ports = ports
        self.env_vars = env_vars
        self.build_args = build_args
        self.command_override = command_override
        self.keep_alive_with_base_command = keep_alive_with_base_command
        self.network = network
        self.extra_networks = list(extra_networks or [])
        # host / none / container:<name> replace the container's whole netns, so they can't be attached as extra networks.
        netns_extra = [n for n in self.extra_networks if n in ("host", "none") or n.startswith("container:")]
        if netns_extra:
            raise ValueError(f"extra_networks cannot contain netns-replacing modes {netns_extra}; use user-defined bridges")
        self.ready_command = ready_command
        self.devices = list(devices or [])
        self.cap_add = list(cap_add or [])
        self.privileged = privileged
        self.volumes = list(volumes or [])
        self.shm_size = shm_size

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["sandbox_name"] = self.sandbox_name
        base["docker_context_artifact_id"] = self.docker_context_artifact_id
        base["docker_context_artifact_version"] = self.docker_context_artifact_version
        base["docker_context_url"] = self.docker_context_url
        base["dockerfile_path"] = self.dockerfile_path
        base["container_name"] = self.container_name
        base["image_tag"] = self.image_tag
        base["ports"] = self.ports
        base["env_vars"] = self.env_vars
        base["build_args"] = self.build_args
        base["command_override"] = self.command_override
        base["keep_alive_with_base_command"] = self.keep_alive_with_base_command
        base["network"] = self.network
        base["extra_networks"] = self.extra_networks
        base["ready_command"] = self.ready_command
        base["devices"] = self.devices
        base["cap_add"] = self.cap_add
        base["privileged"] = self.privileged
        base["volumes"] = self.volumes
        base["shm_size"] = self.shm_size
        return base

    @classmethod
    def from_dict(cls, data: dict) -> RunDockerContainerTaskStep:
        return cls(
            **cls._base_from_dict(data),
            sandbox_name=data["sandbox_name"],
            docker_context_artifact_id=data.get("docker_context_artifact_id"),
            docker_context_artifact_version=data.get("docker_context_artifact_version"),
            docker_context_url=data.get("docker_context_url"),
            dockerfile_path=data.get("dockerfile_path", "Dockerfile"),
            container_name=data.get("container_name", "task-container"),
            image_tag=data.get("image_tag"),
            ports=data.get("ports"),
            env_vars=data.get("env_vars"),
            build_args=data.get("build_args"),
            command_override=data.get("command_override"),
            keep_alive_with_base_command=data.get("keep_alive_with_base_command", False),
            network=data.get("network"),
            extra_networks=data.get("extra_networks"),
            ready_command=data.get("ready_command"),
            devices=data.get("devices"),
            cap_add=data.get("cap_add"),
            privileged=data.get("privileged", False),
            volumes=data.get("volumes"),
            shm_size=data.get("shm_size"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.providers.sandbox_providers.sandbox_provider import (
            SANDBOX_MODE_VM,
            build_sandbox_provider,
            get_sandbox_provider,
        )

        ds = next(
            (s for s in context.deployed_sandboxes if s.sandbox_name == self.sandbox_name),
            None,
        )
        if ds is None:
            raise RuntimeError(
                f"Sandbox '{self.sandbox_name}' not found in context.deployed_sandboxes; "
                f"deploy it with DeploySandboxTaskStep before this step"
            )
        if ds.sandbox_mode != SANDBOX_MODE_VM:
            raise RuntimeError(
                f"Cannot run docker container on sandbox '{self.sandbox_name}' "
                f"(mode={ds.sandbox_mode!r}); only VM-mode sandboxes support docker build/run"
            )

        provider = (
            build_sandbox_provider(ds.sandbox_type) if ds.sandbox_type else get_sandbox_provider()
        )
        sandbox = await provider.get_sandbox(ds.sandbox_id)

        image_tag = self.image_tag or f"{self.container_name}:latest"

        work_dir = f"/tmp/docker-context-{self.container_name}"
        await sandbox.exec_script(
            f"rm -rf {shlex.quote(work_dir)} && mkdir -p {shlex.quote(work_dir)}"
        )

        if self.docker_context_artifact_id is not None:
            await self._stage_from_universe(sandbox, ds.sandbox_id, work_dir)
        else:
            await self._stage_zip_from_url(sandbox, self.docker_context_url, work_dir)

        dockerfile_abs = posixpath.join(work_dir, self.dockerfile_path)
        await sandbox.exec_script(
            f"test -f {shlex.quote(dockerfile_abs)} "
            f"|| (echo 'Dockerfile not found at {self.dockerfile_path} inside the build context' >&2; exit 1)"
        )

        build_arg_flags = ""
        if self.build_args:
            build_arg_flags = " ".join(
                f"--build-arg {shlex.quote(f'{k}={v}')}" for k, v in self.build_args.items()
            )
        label = shlex.quote(f"{SANDBOX_LABEL}={ds.sandbox_id}")
        build_cmd = (
            f"cd {shlex.quote(work_dir)} && "
            f"docker build --platform linux/amd64 --label {label} "
            f"-f {shlex.quote(self.dockerfile_path)} "
            f"-t {shlex.quote(image_tag)} "
            f"{build_arg_flags} ."
        )
        logger.info(
            f"Building docker image '{image_tag}' on sandbox {ds.sandbox_id} "
            f"(dockerfile={self.dockerfile_path})"
        )
        await sandbox.exec_script(build_cmd)

        network_flag = ""
        if self.network:
            network_flag = f"--network {await self._ensure_network(sandbox, self.network)} "

        port_flags = " ".join(f"-p {spec}" for p in (self.ports or []) for spec in port_bindings(sandbox.host_ips, p, p))
        env_flag_parts = []
        if self.env_vars:
            for k, v in self.env_vars.items():
                escaped = v.replace("'", "'\\''")
                env_flag_parts.append(f"-e {k}='{escaped}'")
        env_flags = " ".join(env_flag_parts)
        device_flags = " ".join(f"--device {shlex.quote(d)}" for d in self.devices)
        cap_flags = " ".join(f"--cap-add {shlex.quote(c)}" for c in self.cap_add)
        volume_flags = " ".join(f"-v {shlex.quote(v)}" for v in self.volumes)
        host_flags = " ".join(f"--add-host {shlex.quote(entry)}" for entry in sandbox.extra_hosts)
        limits = sandbox.container_limits
        limit_flags = " ".join(shlex.quote(arg) for arg in limits.docker_args) if limits else ""
        extra_flags = " ".join(f for f in (
            "--privileged" if self.privileged else "",
            f"--shm-size {shlex.quote(self.shm_size)}" if self.shm_size else "",
            device_flags, cap_flags, volume_flags, host_flags, limit_flags,
        ) if f)
        entrypoint_flag = ""
        if self.command_override:
            command_tail = f" {self.command_override}"
        elif self.keep_alive_with_base_command:
            # Keep the container alive for install/exec WITHOUT discarding the
            # image's own startup (a `sleep infinity` override would stop a base
            # CMD that launches a DB/server the task needs). Replay the recorded
            # ENTRYPOINT+CMD in the background, then hold PID 1 open.
            base_cmd = await self._inspect_base_command(sandbox, image_tag)
            if base_cmd:
                base_str = " ".join(shlex.quote(part) for part in base_cmd)
                script = f"{base_str} & exec sleep infinity"
                entrypoint_flag = "--entrypoint /bin/sh "
                command_tail = f" -c {shlex.quote(script)}"
            else:
                command_tail = " sleep infinity"
        else:
            command_tail = ""
        run_cmd = (
            f"docker run -d --name {shlex.quote(self.container_name)} --label {label} "
            f"{network_flag}{entrypoint_flag}{extra_flags} {port_flags} {env_flags} "
            f"{shlex.quote(image_tag)}{command_tail}"
        )
        logger.info(
            f"Starting container '{self.container_name}' from image '{image_tag}' "
            f"on sandbox {ds.sandbox_id}"
        )
        await sandbox.exec_script(run_cmd)

        # Dual-home the container onto additional networks (e.g. a jump host straddling segments).
        # Dedup and skip the primary: `docker network connect` errors on an already-attached endpoint.
        for extra in dict.fromkeys(self.extra_networks):
            if extra == self.network:
                continue
            enet = await self._ensure_network(sandbox, extra)
            await sandbox.exec_script(f"docker network connect {enet} {shlex.quote(self.container_name)}")
            logger.info(f"Connected container '{self.container_name}' to extra network '{extra}'")

        if self.ready_command:
            ready = shlex.quote(self.ready_command)
            logger.info(f"Waiting for container '{self.container_name}' readiness: {self.ready_command}")
            await sandbox.exec_script(
                f"for i in $(seq 1 60); do "
                f"if docker exec {shlex.quote(self.container_name)} sh -c {ready} >/dev/null 2>&1; then exit 0; fi; "
                f"sleep 2; done; "
                f"echo 'container {self.container_name} failed readiness check within 120s' >&2; exit 1"
            )

        context.metadata.setdefault("deployed_docker_containers", []).append({
            "container_name": self.container_name,
            "sandbox_name": self.sandbox_name,
            "sandbox_id": ds.sandbox_id,
            "image_tag": image_tag,
            "ports": self.ports,
            "network": self.network,
        })
        return context

    async def _ensure_network(self, sandbox, name: str) -> str:
        net = shlex.quote(name)
        # host / none / bridge / container:<name> are Docker built-in modes, not user bridges -- nothing to create.
        if name in ("host", "none", "bridge") or name.startswith("container:"):
            return net
        await sandbox.exec_script(
            f"docker network inspect {net} >/dev/null 2>&1 "    # already exists -> reuse it
            f"|| docker network create --label {shlex.quote(f'{SANDBOX_LABEL}={sandbox.sandbox_id}')} {net} >/dev/null 2>&1 "  # else create the user bridge
            f"|| docker network inspect {net} >/dev/null"       # lost a create race -> confirm it exists
        )
        return net

    async def _inspect_base_command(self, sandbox, image_tag: str) -> list[str]:
        """Return the built image's ENTRYPOINT+CMD as a single argv list (or [])."""
        import json

        out = await sandbox.exec_script(
            "docker inspect --format "
            "'{{json .Config.Entrypoint}}|{{json .Config.Cmd}}' "
            f"{shlex.quote(image_tag)}"
        )
        ep_raw, _, cmd_raw = out.strip().partition("|")
        entrypoint = json.loads(ep_raw) if ep_raw and ep_raw != "null" else []
        cmd = json.loads(cmd_raw) if cmd_raw and cmd_raw != "null" else []
        return list(entrypoint or []) + list(cmd or [])

    async def _stage_from_universe(self, sandbox, sandbox_id: str, work_dir: str) -> None:
        from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse

        universe = FileArtifactUniverse.get(
            self.docker_context_artifact_id, self.docker_context_artifact_version,
        )
        file_artifacts = universe.get_file_artifacts()
        if not file_artifacts:
            raise RuntimeError(
                f"FileArtifactUniverse '{universe.id}' v{universe.version} has no files; "
                f"nothing to use as docker build context"
            )

        dirs_to_make: set[str] = set()
        loaded: dict[str, str] = {}
        for filename in file_artifacts:
            validate_relative_filename(filename)
            dest_path = posixpath.join(work_dir, filename)
            parent = posixpath.dirname(dest_path)
            if parent and parent != work_dir:
                dirs_to_make.add(parent)
            loaded[filename] = dest_path
        if dirs_to_make:
            mkdir_cmd = " && ".join(f"mkdir -p {shlex.quote(d)}" for d in sorted(dirs_to_make))
            await sandbox.exec_script(mkdir_cmd)

        total = len(file_artifacts)
        logger.info(
            f"Loading docker context universe '{universe.id}' v{universe.version} "
            f"({total} file(s)) onto sandbox {sandbox_id} at {work_dir}"
        )
        for idx, (filename, fa) in enumerate(file_artifacts.items(), 1):
            dest_path = loaded[filename]
            logger.info(f"  [{idx}/{total}] {fa.object_url} -> {dest_path}")
            await sandbox.load_object_file(fa.object_url, dest_path)

    @staticmethod
    async def _stage_zip_from_url(sandbox, url: str, work_dir: str) -> None:
        """http(s) urls are fetched with curl; any other url is read through the configured
        object store, which reaches whatever its backend can."""
        parsed = urlparse(url)
        if not parsed.scheme:
            raise ValueError(
                f"docker_context_url={url!r} must be an object store url or an http(s) url"
            )
        from_store = parsed.scheme not in ("http", "https")
        path = url.rsplit("/", 1)[-1] if from_store else parsed.path
        if path.endswith(".zip"):
            pass
        elif path.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")):
            raise NotImplementedError(
                f"tar archives are not yet supported (URL: {url}); only .zip is supported today"
            )
        else:
            raise ValueError(
                f"Unsupported archive format for docker_context_url={url!r}; "
                f"only .zip is supported"
            )

        archive_path = posixpath.join(work_dir, "_context.zip")
        logger.info(f"Downloading docker context from {url} into {work_dir}")
        if from_store:
            await sandbox.load_object_file(url, archive_path)
        else:
            await sandbox.exec_script(
                f"curl -fsSL {shlex.quote(url)} -o {shlex.quote(archive_path)}"
            )

        await sandbox.exec_script(
            f"unzip -q -o {shlex.quote(archive_path)} -d {shlex.quote(work_dir)} && "
            f"rm -f {shlex.quote(archive_path)}"
        )
        logger.info(f"Extracted {url} into {work_dir}")
