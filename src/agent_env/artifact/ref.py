"""ArtifactRef — a pinned reference to a specific (id, version) of an artifact."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, description="Artifact id")
    version: int = Field(ge=1, description="Pinned artifact version")

    def __str__(self) -> str:
        return f"{self.id}:{self.version}"
