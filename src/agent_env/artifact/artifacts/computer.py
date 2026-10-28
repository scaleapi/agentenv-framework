"""The computer artifact: a base a computer env composes a driver and apps onto. It declares what the computer is, its
architectures, OS, session user and settings, and how each form starts it. A form's image is a docker_image artifact,
pinned when the computer is written and read back from its registry to check it holds what the computer says
(``metadata.validated_images``)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.ref import ArtifactRef
from agent_env.entity.refs import EntityRef
from agent_env.store.base import NotFoundError
from agent_env.store.image_store.oci_registry_credentials import names_registry
from agent_env.store.image_store.image_contents import UnreadableLayerError, os_release, platform_manifests

if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext

OS_TYPES = ("linux",)
FORMS = ("container",)


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


class ImageCheck(BaseModel):
    """What reading a form's image for one architecture found, as ``metadata.validated_images`` records it: ``passed``,
    ``failed`` with the problem, or ``uninspected`` with the reason it wasn't read."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["passed", "failed", "uninspected"]
    image: str | None = None  # the image read, pinned to its digest
    manifest: str | None = None  # the digest of the architecture's image manifest
    os_release: dict[str, str] | None = None
    problem: str | None = None
    reason: str | None = None
    checked_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


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
        problem by form and architecture, when an image isn't what the computer declares, and inspected or not, when
        it isn't a stored docker_image artifact."""
        from agent_env.artifact.store import get_artifact_store

        computer = cls(**fields)
        image = computer.image()
        checks = {"container": _inspect_container(image, computer.arch, computer.os) if inspect
                  else _unread(computer.arch, "inspection was skipped")}
        problems = [f"forms.{form} ({arch}): {check.problem}"
                    for form, by_arch in checks.items() for arch, check in by_arch.items() if check.status == "failed"]
        if problems:
            raise ValueError(f"computer {computer.id!r}: " + "; ".join(problems))
        computer.metadata = {**computer.metadata, "validated_images": {
            form: {arch: check.model_dump(exclude_none=True) for arch, check in by_arch.items()}
            for form, by_arch in checks.items()}}
        return get_artifact_store().put_document(computer)

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

    def image(self) -> DockerImageArtifact:
        """The container form's image artifact, refused when the store has no docker_image artifact of that id and
        version."""
        ref = self.forms.container.image
        try:
            return DockerImageArtifact.get(ref.id, ref.version)
        except (NotFoundError, TypeError) as e:
            raise ValueError(f"computer {self.id!r}: forms.container.image: {e}") from e


def _inspect_container(image: DockerImageArtifact, arches: list[str], os: LinuxOS) -> dict[str, ImageCheck]:
    from agent_env.config import get_config

    if not names_registry(image.image_name):
        how = "only its build context is stored" if image.context_only else "it's stored under a tag no registry serves"
        return _unread(arches, f"{how}, so its first boot checks it")
    ref = image.pinned_name
    auth = get_config().get_image_store().auth(ref)
    try:
        ref, manifests = platform_manifests(ref, auth)
    except ValueError as e:
        return {arch: ImageCheck(status="failed", image=ref, problem=str(e)) for arch in arches}
    checks = {}
    for arch in arches:
        if (digest := manifests.get(arch)) is None:
            served = ", ".join(f"linux/{a}" for a in sorted(manifests)) or "none"
            checks[arch] = ImageCheck(status="failed", image=ref, problem=f"it has no linux/{arch} image, only {served}")
            continue
        try:
            release = os_release(ref, digest, auth)
        except UnreadableLayerError as e:
            checks[arch] = ImageCheck(status="uninspected", image=ref, manifest=digest,
                                      reason=f"{e}, so its first boot checks it")
            continue
        except ValueError as e:
            checks[arch] = ImageCheck(status="failed", image=ref, manifest=digest,
                                      problem=f"its os-release couldn't be read: {e}")
            continue
        expected = {"ID": os.distribution, "VERSION_ID": None if os.version == "rolling" else os.version}
        problems = [f"os-release has {key}={release.get(key)!r}, not {want!r}" for key, want in expected.items()
                    if release.get(key) != want]
        checks[arch] = ImageCheck(status="failed" if problems else "passed", image=ref, manifest=digest,
                                  os_release={key: release[key] for key in expected if key in release},
                                  problem="; ".join(problems) or None)
    return checks


def _unread(arches: list[str], why: str) -> dict[str, ImageCheck]:
    return {arch: ImageCheck(status="uninspected", reason=why) for arch in arches}
