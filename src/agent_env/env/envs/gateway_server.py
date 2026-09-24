"""Gateway server environment."""

from __future__ import annotations

from typing import ClassVar, Optional

from agent_env.artifact import Artifact, DockerImageArtifact
from agent_env.env.env import Env


class GatewayEnv(Env):
    type: ClassVar[str] = "gateway_server"
    description = "Gateway server that orchestrates multiple MCP servers"

    def __init__(self, id: str, version: Optional[int], docker_image_artifact: DockerImageArtifact, metadata: Optional[dict[str, str]] = None):
        super().__init__(id, version, metadata=metadata)
        self.docker_image_artifact = docker_image_artifact

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["docker_image_artifact"] = {
            "id": self.docker_image_artifact.id,
            "version": self.docker_image_artifact.version,
            "type": self.docker_image_artifact.type,
        }
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "GatewayEnv":
        artifact_ref = data["docker_image_artifact"]
        docker_image_artifact = Artifact.get(artifact_ref["id"], version=artifact_ref["version"])
        return cls(id=data["id"], version=data.get("version"), docker_image_artifact=docker_image_artifact, metadata=data.get("metadata", {}))
