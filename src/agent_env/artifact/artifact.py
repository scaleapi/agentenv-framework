"""Artifact models for AgentEnv."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Self

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from agent_env.artifact.ref import ArtifactRef
    from agent_env.artifact.store import ArtifactQuery


class Artifact(BaseModel):
    """Base class for all environment artifacts. Immutable - each modification creates a new version."""

    id: str = Field(description="Unique artifact identifier")
    version: int = Field(default=0, description="Artifact version (auto-calculated on put)")
    type: str = Field(description="The type of Artifact")

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
        from agent_env.artifact.store import get_artifact_store

        return get_artifact_store().get(id, version)

    @classmethod
    def put(cls, **kwargs) -> Self:
        from agent_env.artifact.store import get_artifact_store

        instance = cls(**kwargs)
        return get_artifact_store().put_document(instance)

    @classmethod
    def query(cls) -> "ArtifactQuery":
        from agent_env.artifact.store import ArtifactQuery, get_artifact_store

        return ArtifactQuery(get_artifact_store())
