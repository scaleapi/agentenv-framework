"""The computer artifact: a base a computer env composes a driver and apps onto. It declares what the computer is, its
architectures, OS, session user and settings, and how each form starts it. A form's image is a docker_image artifact,
pinned when the computer is written and read back from its registry to check it holds what the computer says
(``metadata.validated_images``)."""

from __future__ import annotations

import json
import re
import tarfile
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.ref import ArtifactRef
from agent_env.entity.refs import EntityRef
from agent_env.store.ids import derive_id
from agent_env.store.image_store.oci_registry_credentials import names_registry
from agent_env.store.image_store.registry_api import read_blob, read_manifest

if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext

OS_TYPES = ("linux",)
FORMS = ("container",)
_INDEX_TYPES = ("application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json")
# os-release(5): /etc/os-release first, else /usr/lib/os-release; on Ubuntu the first links to the second.
_OS_RELEASE = ("etc/os-release", "usr/lib/os-release")


class LinuxOS(BaseModel):
    """A Linux base: its distribution and version (``rolling`` for one with no versions, such as Arch), and the display
    server its desktop runs."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["linux"] = "linux"
    distribution: str = Field(pattern=r"^[a-z][a-z0-9-]*$", description="os-release's ID: ubuntu, debian, fedora, arch")
    version: str = Field(min_length=1, description="os-release's VERSION_ID, or rolling")
    display: Literal["x11", "wayland"]
    build: str | None = None


class Session(BaseModel):
    """The user a desktop session runs as, and so every driver and app added to it."""

    model_config = ConfigDict(extra="forbid")

    user: str = Field(pattern=r"^[a-z_][a-z0-9_-]{0,31}$")


class Setting(BaseModel):
    """A value the computer reads, which an env or a deploy can override: ``build`` when it changes the image, ``deploy``
    when the computer reads it as it starts; ``user`` when it names the user's own setup, such as a secret's reference."""

    model_config = ConfigDict(extra="forbid")

    default: str | int | float | bool
    phase: Literal["build", "deploy"]
    scope: Literal["task", "user"] = "task"
    secret: bool = False
    generate: bool = Field(default=False, description="minted afresh for every deploy")

    @model_validator(mode="after")
    def _phase_fits(self) -> Self:
        if self.phase == "build" and self.scope != "task":
            raise ValueError("a build-phase setting is task-scoped, so one env version always composes one image")
        if self.generate and self.phase != "deploy":
            raise ValueError("a generated setting is minted for every deploy, so its phase is deploy")
        return self


class ContainerForm(BaseModel):
    """The computer as an OCI image a container runtime runs: ``init`` starts its programs, and deploy settings reach
    them through ``input_channel``."""

    model_config = ConfigDict(extra="forbid")

    image: ArtifactRef
    init: Literal["supervisord"]
    input_channel: Literal["env"]


class Forms(BaseModel):
    model_config = ConfigDict(extra="forbid")

    container: ContainerForm


class ComputerArtifact(Artifact):
    """A base computer, which composing adds drivers and apps to."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["computer"] = "computer"
    description: str = Field(default="", description="What the computer is")
    arch: list[Literal["amd64", "arm64"]] = Field(min_length=1, description="The architectures every form runs on")
    os: LinuxOS
    session: Session
    forms: Forms
    config: dict[str, Setting] = Field(default_factory=dict, description="Its settings, by name")

    toml_refs: ClassVar[tuple[EntityRef, ...]] = (EntityRef.artifact("forms.container.image",
                                                                     artifact_type="docker_image"),)
    toml_keys: ClassVar[dict[str, type]] = {"description": str, "arch": list, "os": dict, "session": dict,
                                            "forms": dict, "config": dict}

    @field_validator("os", mode="before")
    @classmethod
    def _os_type_supported(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("type", "linux") not in OS_TYPES:
            raise ValueError(f"os.type {value['type']!r} isn't supported yet; a computer is {', '.join(OS_TYPES)}")
        return value

    @field_validator("forms", mode="before")
    @classmethod
    def _forms_supported(cls, value: Any) -> Any:
        if isinstance(value, dict) and (others := sorted(set(value) - set(FORMS))):
            raise ValueError(f"the {', '.join(others)} form{'s' * (len(others) > 1)} isn't supported yet; a computer has "
                             f"{', '.join(FORMS)}")
        return value

    @field_validator("arch")
    @classmethod
    def _arch_once(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError(f"lists an architecture twice: {value}")
        return value

    @field_validator("config")
    @classmethod
    def _setting_names(cls, value: dict[str, Setting]) -> dict[str, Setting]:
        if bad := sorted(name for name in value if not re.fullmatch(r"[a-z][a-z0-9_]*", name)):
            raise ValueError(f"setting names are lowercase letters, digits and _, since each reaches the computer as "
                             f"AGENTENV_SETTING_<NAME>: {', '.join(map(repr, bad))}")
        return value

    @classmethod
    def put(cls, *, inspect: bool = True, **fields: Any) -> Self:
        """Write a computer, reading each form's image first unless ``inspect`` is False. Refused, naming every
        problem by form and architecture, when an image isn't what the computer declares."""
        from agent_env.artifact.store import get_artifact_store

        computer = cls(**fields)
        found = computer.inspect() if inspect else {"container": _unread(computer.arch, "inspection was skipped")}
        problems = [f"forms.{form} ({arch}): {result['problem']}"
                    for form, by_arch in found.items() for arch, result in by_arch.items() if result["status"] == "failed"]
        if problems:
            raise ValueError(f"computer {computer.id!r}: " + "; ".join(problems))
        computer.metadata = {**computer.metadata, "validated_images": found}
        return get_artifact_store().put_document(computer)

    @classmethod
    def put_folder(cls, id: str, folder: Path, *, inspect: bool = True) -> Self:
        """Write the computer ``folder``'s artifact.toml declares, outside a bundle, its form's image key written as a
        bundle writes one: a Dockerfile in the folder as its build context alone, a registry ref pinned now, or the image
        artifact it names."""
        fields = {key: value for key, value in tomllib.loads((folder / "artifact.toml").read_text()).items()
                  if key not in ("id", "type", "version")}
        container = (fields.get("forms") or {}).get("container")
        if isinstance(container, dict) and "image" in container:
            container["image"] = _folder_image(derive_id(id, "forms_container_image"), folder, container["image"])
        return cls.put(id=id, inspect=inspect, **fields)

    @classmethod
    def from_toml(cls, data: dict[str, Any], ctx: AuthoringContext) -> Self:
        """Write the computer an artifact.toml declares, its form's image as the bundle wrote it or the artifact the
        key names."""
        fields = {key: value for key, value in data.items() if key not in ("id", "type", "version")}
        container = (fields.get("forms") or {}).get("container")
        if isinstance(container, dict) and "image" in container:
            image = ctx.artifact(container["image"])
            container["image"] = {"id": image.id, "version": image.version}
        try:
            return cls.put(id=ctx.id, **fields)
        except ValidationError as e:
            ctx.refuse([ctx.config_problem(f"{'.'.join(map(str, error['loc']))}: {error['msg']}") for error in e.errors()])
        except ValueError as e:
            ctx.refuse([str(e)])

    def inspect(self) -> dict[str, dict[str, dict[str, Any]]]:
        """What each form's image holds for each architecture, read from its registry: ``passed``, ``failed`` with the
        problem, or ``uninspected`` with why, for an image no registry serves yet."""
        image = DockerImageArtifact.get(self.forms.container.image.id, self.forms.container.image.version)
        return {"container": _inspect_container(image, self.arch, self.os)}


def _folder_image(image_id: str, folder: Path, key: Any) -> dict[str, Any]:
    if isinstance(key, dict) and key.keys() == {"dockerfile"}:
        image = DockerImageArtifact.put_context(image_id, description=f"built from {folder.name}/{key['dockerfile']}",
                                                context_path=str(folder), dockerfile_path=key["dockerfile"], platform=None)
    elif isinstance(key, dict) and key.keys() == {"ref"}:
        image = DockerImageArtifact.put_ref(image_id, description=f"referenced by {folder.name}", image_name=key["ref"])
    elif isinstance(key, dict) and key.keys() <= {"artifact", "version"} and "artifact" in key:
        image = DockerImageArtifact.get(key["artifact"], key.get("version"))
    elif isinstance(key, str):
        image = DockerImageArtifact.get(key)
    else:
        raise ValueError(f"forms.container.image is {{ dockerfile = \"<path>\" }}, {{ ref = \"<image>\" }}, an image "
                         f"artifact's id or {{ artifact = \"<id>\", version = <n> }}, not {key!r}")
    return {"id": image.id, "version": image.version}


def _inspect_container(image: DockerImageArtifact, arches: list[str], os: LinuxOS) -> dict[str, dict[str, Any]]:
    from agent_env.config import get_config

    if not names_registry(image.image_name):
        how = "only its build context is stored" if image.context_only else "it's stored under a tag no registry serves"
        return _unread(arches, f"{how}, so its first boot checks it")
    ref = image.pinned_name
    auth = get_config().get_image_store().auth(ref)
    try:
        manifests = _platform_manifests(ref, auth)
    except ValueError as e:
        return {arch: _result("failed", ref, problem=str(e)) for arch in arches}
    results = {}
    for arch in arches:
        if (digest := manifests.get(arch)) is None:
            results[arch] = _result("failed", ref, problem=f"it has no linux/{arch} image, only "
                                                           f"{', '.join(f'linux/{a}' for a in sorted(manifests)) or 'none'}")
            continue
        try:
            release = _os_release(ref, digest, auth)
        except (ValueError, OSError, tarfile.TarError) as e:
            results[arch] = _result("failed", ref, manifest=digest, problem=f"its os-release couldn't be read: {e}")
            continue
        expected = {"ID": os.distribution, "VERSION_ID": None if os.version == "rolling" else os.version}
        problems = [f"os-release has {key}={release.get(key)!r}, not {want!r}" for key, want in expected.items()
                    if release.get(key) != want]
        results[arch] = _result("failed" if problems else "passed", ref, manifest=digest,
                                os_release={key: release[key] for key in expected if key in release},
                                problem="; ".join(problems) or None)
    return results


def _platform_manifests(ref: str, auth) -> dict[str, str]:
    """Each Linux architecture's image manifest digest: an index's, or a single image's own by its config."""
    top = read_manifest(ref, auth)
    if top.get("mediaType") in _INDEX_TYPES or "manifests" in top:
        return {entry["platform"]["architecture"]: entry["digest"] for entry in top.get("manifests") or ()
                if (entry.get("platform") or {}).get("os") == "linux"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config"
        read_blob(ref, top["config"]["digest"], auth, path)
        config = json.loads(path.read_text())
    return {config["architecture"]: ref.rpartition("@")[2]} if config.get("os") == "linux" else {}


def _os_release(ref: str, digest: str, auth) -> dict[str, str]:
    """The os-release of the image manifest ``digest``, from the lowest layer that holds it, normally the OS's own, so
    only that layer is fetched."""
    name = ref.partition("@")[0]
    manifest = read_manifest(f"{name}@{digest}", auth)
    found: dict[str, bytes | str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for layer in manifest.get("layers") or ():
            path = Path(tmp) / layer["digest"].replace(":", "-")
            read_blob(ref, layer["digest"], auth, path)
            with tarfile.open(path, "r:*") as tar:
                for member in tar:
                    member_name = member.name.removeprefix("./")
                    if member_name in _OS_RELEASE:
                        if member.issym():
                            found[member_name] = member.linkname
                        elif (extracted := tar.extractfile(member)) is not None:
                            found[member_name] = extracted.read()
            path.unlink()
            if (text := _resolved(found)) is not None:
                return _parse_os_release(text)
    raise ValueError("no layer holds /etc/os-release or /usr/lib/os-release")


def _resolved(found: dict[str, bytes | str]) -> str | None:
    for name in _OS_RELEASE:
        value = found.get(name)
        if isinstance(value, str):  # a link, to a path in the image
            target = PurePosixPath(value) if value.startswith("/") else PurePosixPath("/", name).parent / value
            value = found.get(_normalized(target))
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
    return None


def _normalized(path: PurePosixPath) -> str:
    parts: list[str] = []
    for part in path.parts[1:]:
        if part == "..":
            parts = parts[:-1]
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts)


def _parse_os_release(text: str) -> dict[str, str]:
    release = {}
    for line in text.splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            release[key] = value.strip().strip("\"'")
    return release


def _result(status: str, image: str | None, **found: Any) -> dict[str, Any]:
    return {"status": status, **({"image": image} if image else {}),
            **{key: value for key, value in found.items() if value is not None},
            "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _unread(arches: list[str], why: str) -> dict[str, dict[str, Any]]:
    return {arch: _result("uninspected", None, reason=why) for arch in arches}
