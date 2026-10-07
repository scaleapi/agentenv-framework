"""Docker image artifact for storing Docker images in the object store."""

from __future__ import annotations

import asyncio
import gzip
import logging
import os
import posixpath
import re
import shlex
import subprocess
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal
from urllib.parse import urlparse

from pydantic import ConfigDict, Field, model_serializer

from agent_env.artifact.artifact import Artifact, _write_twin
from agent_env.store.ids import fs_safe, image_repository, is_local_id
from agent_env.store.image_store.oci_registry_credentials import is_loopback_host, registry_host_from_ref
from agent_env.utils.deprecation import OMITTED, renamed_keyword

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str, int], None]

_SAFE_GIT_NAME = re.compile(r"^[a-zA-Z0-9._-]+$")


def _validate_git_name(value: str, label: str) -> str:
    if not _SAFE_GIT_NAME.match(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value



def _git_clone_commands(owner: str, repo: str, ref: str | None, token: str | None) -> list[str]:
    """Shell commands that clone ``owner/repo`` at ``ref`` into /tmp/repo. A token rides a
    throwaway GIT_ASKPASS helper so it never appears on a command line, and the helper is
    removed whether or not the clone succeeds; without a token the clone is anonymous and
    fails fast on a private repository instead of prompting."""
    ref_flag = f"--branch {ref}" if ref else ""
    clone_url = f"https://github.com/{owner}/{repo}.git"
    if not token:
        return [f"GIT_TERMINAL_PROMPT=0 git clone --depth 1 {ref_flag} {clone_url} /tmp/repo"]
    askpass_script = (
        '#!/bin/sh\n'
        'case "$1" in\n'
        '*sername*) echo x-access-token;;\n'
        f'*) echo "{token}";;\n'
        'esac\n'
    )
    return [
        f"cat > /tmp/git-askpass.sh << 'ASKEOF'\n{askpass_script}ASKEOF",
        "chmod +x /tmp/git-askpass.sh",
        f"trap 'rm -f /tmp/git-askpass.sh' EXIT; GIT_ASKPASS=/tmp/git-askpass.sh git clone --depth 1 {ref_flag} {clone_url} /tmp/repo",
    ]

class DockerImageArtifact(Artifact):
    """A Docker image artifact stored as tar.gz in the object store."""

    model_config = ConfigDict(populate_by_name=True)

    DOCKER_SAVE_TIMEOUT_SECONDS: ClassVar[int] = 900

    type: Literal["docker_image"] = "docker_image"
    description: str = Field(description="Description of the Docker image")
    image_name: str = Field(description="Docker image name/tag")
    tar_gz_object_url: str = Field(alias="tar_gz_s3_url", description="Object-store locator of the tar.gz file")
    build_context_object_url: str | None = Field(default=None, alias="build_context_s3_url", description="Object-store locator of the build context tar.gz")

    # No return annotation: pydantic builds the serialization schema from one, and a dict drops the fields.
    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any):
        # Dual-write, from the attributes: `alias=` emits one spelling, which one depending on the caller's `by_alias`.
        data = handler(self)
        _write_twin(data, "tar_gz_s3_url", "tar_gz_object_url", self.tar_gz_object_url)
        _write_twin(data, "build_context_s3_url", "build_context_object_url", self.build_context_object_url)
        return data

    @classmethod
    def put(
        cls,
        id: str,
        *,
        description: str,
        image_name: str,
        build_context_path: str | None = None,
        dockerfile_path: str | None = None,
    ) -> "DockerImageArtifact":
        from agent_env.artifact.store import get_artifact_store
        from agent_env.config import get_config

        store = get_artifact_store()
        version = store.next_version(id)
        # Objects are written once, so each attempt writes under a prefix of its own: one that stopped
        # before its document was written doesn't block the next.
        prefix = store.attempt_prefix("docker_image", id)

        config = get_config()
        objects = config.get_object_store_to_write(prefix, id)
        image_store = config.get_image_store_for(id)
        repository = image_repository(id)
        image_ref = image_store.image_ref(repository, f"v{version}")
        image_store.ensure_repository(repository)
        _push_local_image(image_name, image_ref, image_store)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_proc = subprocess.Popen(
                ["docker", "save", image_ref],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with gzip.open(tmp_path, "wb") as gz_file:
                while chunk := save_proc.stdout.read(8192):
                    gz_file.write(chunk)
            try:
                save_proc.wait(timeout=cls.DOCKER_SAVE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                save_proc.kill()
                raise RuntimeError(f"docker save {image_ref} timed out after {cls.DOCKER_SAVE_TIMEOUT_SECONDS} seconds")
            if save_proc.returncode != 0:
                stderr = save_proc.stderr.read().decode() if save_proc.stderr else ""
                raise RuntimeError(f"docker save {image_ref} failed: {stderr}")

            tar_gz_object_url = objects.put_file_at(
                f"{prefix}{fs_safe(id)}-v{version}.tar.gz", str(tmp_path), "application/gzip"
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        build_context_object_url = None
        if build_context_path:
            context_dir = Path(build_context_path)
            paths_to_include = _get_dockerfile_copy_sources(context_dir, dockerfile_path)
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as ctx_tmp:
                ctx_tmp_path = Path(ctx_tmp.name)
            try:
                with tarfile.open(ctx_tmp_path, "w:gz") as tar:
                    for rel_path in paths_to_include:
                        full_path = context_dir / rel_path
                        if full_path.exists():
                            tar.add(str(full_path), arcname=rel_path)
                build_context_object_url = objects.put_file_at(
                    f"{prefix}build-context.tar.gz", str(ctx_tmp_path), "application/gzip"
                )
            finally:
                if ctx_tmp_path.exists():
                    ctx_tmp_path.unlink()

        return cls.put_tar(
            id,
            description=description,
            image_name=image_ref,
            tar_gz_object_url=tar_gz_object_url,
            build_context_object_url=build_context_object_url,
        )

    @classmethod
    def put_tar(
        cls,
        id: str,
        *,
        description: str,
        image_name: str,
        tar_gz_object_url: str | None = None,
        build_context_object_url: str | None = None,
        tar_gz_s3_url: str | None = OMITTED,
        build_context_s3_url: str | None = OMITTED,
    ) -> "DockerImageArtifact":
        from agent_env.artifact.store import get_artifact_store
        from agent_env.config import get_config

        owner = "DockerImageArtifact.put_tar"
        tar_gz_object_url = renamed_keyword(owner, "tar_gz_object_url", tar_gz_object_url, "tar_gz_s3_url", tar_gz_s3_url)
        build_context_object_url = renamed_keyword(
            owner, "build_context_object_url", build_context_object_url, "build_context_s3_url", build_context_s3_url
        )
        if tar_gz_object_url is None:
            raise TypeError(f"{owner}() missing required keyword argument: 'tar_gz_object_url'")
        for url in (tar_gz_object_url, build_context_object_url):
            if url:
                get_config().check_object_url(id, url)
        store = get_artifact_store()
        version = store.next_version(id)
        instance = cls(
            id=id,
            version=version,
            description=description,
            image_name=image_name,
            tar_gz_object_url=tar_gz_object_url,
            build_context_object_url=build_context_object_url,
        )
        return store.put_document(instance)

    def load(self) -> bytes:
        from agent_env.artifact.store import get_artifact_store
        return get_artifact_store().get_object(self.tar_gz_object_url)

    @classmethod
    async def put_from_github(
        cls,
        id: str,
        dockerfile_github_url: str,
        docker_context_github_url: str | None = None,
        on_progress: ProgressCallback | None = None,
        github_token: str | None = None,
    ) -> GitHubBuildResult:
        """Build a Docker image from a GitHub repo on a temporary VM.

        Clones the repo (with ``github_token`` when the repository is private), runs docker
        build, saves the image, uploads it to the object store, and creates a DockerImageArtifact. Returns a
        GitHubBuildResult with the artifact and git metadata.

        Note: Branch names containing slashes (e.g. feature/fix-bug) are not
        supported in GitHub URLs due to path ambiguity. Use branches/tags
        without slashes, or the default branch.
        """
        from agent_env.providers import get_sandbox_provider
        from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
        from agent_env.providers.sandbox_providers.sandbox import upload_vm_file
        from agent_env.config import get_config

        refuse_local_github_build(id)
        log = on_progress or (lambda step, msg, pct: None)

        log("validate", "Parsing GitHub URLs...", 5)
        df_parts = cls._parse_github_url(dockerfile_github_url)
        if df_parts.path is None:
            raise ValueError(f"Dockerfile URL must include a path to the Dockerfile: {dockerfile_github_url}")

        if docker_context_github_url:
            ctx_parts = cls._parse_github_url(docker_context_github_url)
            if (ctx_parts.owner, ctx_parts.repo) != (df_parts.owner, df_parts.repo):
                raise ValueError("Dockerfile and context URLs must reference the same repository")
            if ctx_parts.ref and df_parts.ref and ctx_parts.ref != df_parts.ref:
                raise ValueError("Dockerfile and context URLs must reference the same ref")
            context_repo_path = ctx_parts.path or ""
        else:
            context_repo_path = posixpath.dirname(df_parts.path)

        owner, repo, ref = df_parts.owner, df_parts.repo, df_parts.ref
        dockerfile_repo_path = df_parts.path
        config = get_config()
        suffix = uuid.uuid4().hex[:16]
        image_tag = f"{id}-{suffix}"

        from agent_env.artifact.store import get_artifact_store
        store = get_artifact_store()
        version = store.next_version(id)
        image_store = config.get_image_store()
        repository = image_repository(id)
        image_ref = image_store.image_ref(repository, f"v{version}")
        provider = get_sandbox_provider()
        if is_loopback_host(registry_host_from_ref(image_ref)) and not isinstance(provider, LocalSandboxProvider):
            raise ValueError(
                f"{image_ref} is in a registry on this machine, which a {type(provider).__name__} build VM can't push to; "
                "build on the local sandbox provider, or configure an image store a remote VM can reach"
            )
        await asyncio.to_thread(image_store.ensure_repository, repository)
        auth = await asyncio.to_thread(image_store.auth, image_ref)

        log("create_vm", "Creating VM...", 10)
        logger.info(f"put_from_github: cloning {owner}/{repo} ref={ref} dockerfile={dockerfile_repo_path} context={context_repo_path}")
        sandbox = await provider.create_vm(disk_size_gb=20, timeout=1800, exposed_ports=[])
        try:
            log("install_git", "Installing git...", 20)
            await sandbox.exec_script("apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1")

            log("clone", f"Cloning {owner}/{repo}...", 30)
            _validate_git_name(owner, "owner")
            _validate_git_name(repo, "repo")
            if ref:
                _validate_git_name(ref, "ref")
            for command in _git_clone_commands(owner, repo, ref, github_token):
                await sandbox.exec_script(command)

            commit_hash = (await sandbox.exec_script("git -C /tmp/repo rev-parse HEAD")).strip()

            log("build", "Building Docker image...", 50)
            dockerfile_abs = shlex.quote(f"/tmp/repo/{dockerfile_repo_path}")
            context_abs = shlex.quote(f"/tmp/repo/{context_repo_path}" if context_repo_path else "/tmp/repo")
            await sandbox.exec_script(f"test -f {dockerfile_abs}")
            await sandbox.exec_script(f"docker build --platform linux/amd64 -f {dockerfile_abs} -t {shlex.quote(image_tag)} {context_abs}")

            log("push", "Pushing image...", 60)
            login_cmd = ""
            if auth is not None:
                login_cmd = (
                    f"echo {shlex.quote(auth.password)} | docker login "
                    f"--username {shlex.quote(auth.username)} --password-stdin {shlex.quote(auth.registry)} && "
                )
            await sandbox.exec_script(
                f"docker tag {shlex.quote(image_tag)} {shlex.quote(image_ref)} && "
                f"{login_cmd}"
                f"docker push {shlex.quote(image_ref)}"
            )

            log("save", "Saving Docker image...", 65)
            await sandbox.exec_script(f"docker save {shlex.quote(image_ref)} | gzip > /tmp/image.tar.gz")

            log("upload", "Uploading to object store...", 75)
            object_store = config.get_object_store()
            builds = f"{config.get_artifact_key_prefix()}github-builds/{id}"
            tar_gz_object_url = object_store.object_url(f"{builds}/{suffix}.tar.gz")
            await upload_vm_file(sandbox, "/tmp/image.tar.gz", object_store, tar_gz_object_url)

            log("upload_context", "Uploading build context...", 80)
            dockerfile_content = await sandbox.exec_script(f"cat {dockerfile_abs}")
            df_rel = posixpath.relpath(dockerfile_repo_path, context_repo_path) if context_repo_path else dockerfile_repo_path
            copy_sources = _parse_copy_sources(dockerfile_content, df_rel)
            tar_paths = " ".join(shlex.quote(p) for p in copy_sources)
            await sandbox.exec_script(f"tar czf /tmp/build-context.tar.gz -C {context_abs} {tar_paths}")
            build_context_object_url = object_store.object_url(f"{builds}/{suffix}-context.tar.gz")
            await upload_vm_file(sandbox, "/tmp/build-context.tar.gz", object_store, build_context_object_url)
        finally:
            log("cleanup", "Terminating build VM...", 85)
            await sandbox.terminate()

        log("store", "Storing artifact...", 90)
        artifact = cls.put_tar(
            id=id,
            description=f"Built from GitHub: {dockerfile_github_url}",
            image_name=image_ref,
            tar_gz_object_url=tar_gz_object_url,
            build_context_object_url=build_context_object_url,
        )
        logger.info(f"put_from_github: created artifact id={artifact.id} version={artifact.version}")

        result = GitHubBuildResult(
            artifact=artifact,
            github_owner=owner,
            github_repo=repo,
            github_ref=ref,
            github_commit=commit_hash,
            dockerfile_github_url=dockerfile_github_url,
            docker_context_github_url=docker_context_github_url,
        )
        log("done", f"Created artifact {id}", 100)
        return result

    @dataclass
    class _GitHubURLParts:
        owner: str
        repo: str
        ref: str | None
        path: str | None

    @staticmethod
    def _parse_github_url(url: str) -> DockerImageArtifact._GitHubURLParts:
        """Parse a GitHub browser URL into components.

        Accepts:
            https://github.com/{owner}/{repo}
            https://github.com/{owner}/{repo}/tree/{ref}
            https://github.com/{owner}/{repo}/tree/{ref}/{path}
            https://github.com/{owner}/{repo}/blob/{ref}/{path}
        """
        parsed = urlparse(url)
        if parsed.hostname != "github.com":
            raise ValueError(f"Expected github.com URL, got: {parsed.hostname}")
        segments = [s for s in parsed.path.strip("/").split("/") if s]
        if len(segments) < 2:
            raise ValueError(f"GitHub URL must include owner and repo: {url}")
        owner, repo = segments[0], segments[1]
        if len(segments) == 2:
            return DockerImageArtifact._GitHubURLParts(owner=owner, repo=repo, ref=None, path=None)
        if segments[2] not in ("tree", "blob"):
            raise ValueError(f"Expected /tree/ or /blob/ in GitHub URL, got /{segments[2]}/: {url}")
        if len(segments) < 4:
            raise ValueError(f"GitHub URL must include a ref after /tree/: {url}")
        ref = segments[3]
        path = "/".join(segments[4:]) if len(segments) > 4 else None
        return DockerImageArtifact._GitHubURLParts(owner=owner, repo=repo, ref=ref, path=path)


def refuse_local_github_build(entity_id: str) -> None:
    """A GitHub build publishes to the configured registry, so it takes a registry id; an ``@local``
    image is built from its own Dockerfile."""
    if is_local_id(entity_id):
        raise ValueError(
            f"{entity_id!r} is an @local id, but a GitHub build publishes to the configured registry; give it a registry id"
        )


@dataclass
class GitHubBuildResult:
    artifact: DockerImageArtifact
    github_owner: str
    github_repo: str
    github_ref: str | None
    github_commit: str
    dockerfile_github_url: str
    docker_context_github_url: str | None


def _push_local_image(local_tag: str, image_ref: str, image_store) -> None:
    auth = image_store.auth(image_ref)
    if auth is not None:
        login = subprocess.run(
            ["docker", "login", "--username", auth.username, "--password-stdin", auth.registry],
            input=auth.password, text=True, capture_output=True,
        )
        if login.returncode != 0:
            raise RuntimeError(f"docker login failed: {login.stderr}")

    for cmd, label in [
        (["docker", "tag", local_tag, image_ref], "tag"),
        (["docker", "push", image_ref], "push"),
    ]:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"docker {label} failed: {result.stderr}")
    logger.info(f"Pushed {local_tag} -> {image_ref}")


def _parse_copy_sources(dockerfile_text: str, dockerfile_rel_path: str | None = None) -> list[str]:
    """Parse COPY/ADD source paths from Dockerfile text. Returns top-level directories to include in context tar."""
    paths: list[str] = []
    if dockerfile_rel_path:
        paths.append(dockerfile_rel_path)

    for line in dockerfile_text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("COPY ") and not stripped.upper().startswith("ADD "):
            continue
        parts = stripped.split()
        args = [p for p in parts[1:] if not p.startswith("--")]
        if len(args) < 2:
            continue
        for src in args[:-1]:
            if src.startswith("/") or "://" in src:
                continue
            src_clean = src.rstrip("/")
            top_dir = src_clean.split("/")[0]
            if top_dir and top_dir != "." and top_dir not in paths:
                paths.append(top_dir)

    return paths if paths else ["."]


def _get_dockerfile_copy_sources(context_dir: Path, dockerfile_path: str | None) -> list[str]:
    """Parse COPY source paths from a local Dockerfile and return relative paths to include in the build context tar."""
    if not dockerfile_path:
        return ["."]

    df = Path(dockerfile_path)
    if not df.is_absolute() and not df.exists():
        df = context_dir / df

    try:
        df_rel = str(df.relative_to(context_dir))
    except ValueError:
        df_rel = None

    paths = _parse_copy_sources(df.read_text(), df_rel)
    return [p for p in paths if (context_dir / p).exists()] or ["."]
