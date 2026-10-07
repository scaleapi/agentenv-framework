"""Artifact models for AgentEnv."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Optional, Self

from pydantic import BaseModel, Field, field_validator

from agent_env.entity_refs import EntityRef

if TYPE_CHECKING:
    from agent_env.bundle.authoring import AuthoringContext
    from agent_env.artifact.ref import ArtifactRef
    from agent_env.artifact.store import ArtifactQuery


def _write_twin(data: dict[str, Any], legacy: str, neutral: str, value: Any) -> None:
    """Write a renamed field under both its legacy and its neutral key, when the dump has it under either: readers
    of either spelling keep working, and a field the dump excluded stays excluded."""
    if legacy in data or neutral in data:
        data[legacy] = data[neutral] = value


class Artifact(BaseModel):
    """Base class for all environment artifacts. Immutable - each modification creates a new version."""

    id: str = Field(description="Unique artifact identifier")
    version: int = Field(default=0, description="Artifact version (auto-calculated on put)")
    type: str = Field(description="The type of Artifact")

    toml_refs: ClassVar[tuple[EntityRef, ...]] = ()

    @field_validator("type", mode="before")
    @classmethod
    def _canonical_type(cls, value):
        """Accept a spelling `[artifacts] type_aliases` maps to this type's own."""
        from agent_env.artifact.registry import canonical_type

        return canonical_type(value) if isinstance(value, str) else value

    def as_ref(self) -> "ArtifactRef":
        from agent_env.artifact.ref import ArtifactRef

        if self.version < 1:
            raise ValueError(
                f"Cannot create a ref for unpersisted artifact {self.id!r} "
                "(version=0). Call put() first."
            )
        return ArtifactRef(id=self.id, version=self.version)

    @classmethod
    def get(cls, id: str, version: Optional[int] = None) -> Self:
        """The artifact stored under ``id``, which must be a ``cls``: the store builds the class
        its stored type names, so ``FileArtifact.get`` refuses an id that holds another type."""
        from agent_env.artifact.store import get_artifact_store

        artifact = get_artifact_store().get(id, version)
        if not isinstance(artifact, cls):
            raise TypeError(f"artifact {id!r} is a {artifact.type!r} artifact, not a {cls.__name__}")
        return artifact

    @classmethod
    def put(cls, **kwargs: Any) -> Self:
        from agent_env.artifact.store import get_artifact_store

        instance = cls(**kwargs)
        return get_artifact_store().put_document(instance)

    @classmethod
    def from_toml(cls, data: dict[str, Any], ctx: AuthoringContext) -> Self:
        """Write the artifact authored as ``data`` (its toml, with the keys ``toml_refs`` declares
        resolved to ids) under ``ctx.id`` and return it. The default suits a type that is only a
        document; one whose ``put`` uploads files or images overrides this."""
        fields = {key: value for key, value in data.items() if key not in ("id", "type", "version")}
        return cls.put(id=ctx.id, **fields)

    @classmethod
    def query(cls) -> "ArtifactQuery":
        from agent_env.artifact.store import ArtifactQuery, get_artifact_store

        return ArtifactQuery(get_artifact_store())
