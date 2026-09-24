"""ServiceDB environment for shared PostgreSQL database."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from agent_env.artifact import Artifact, DockerImageArtifact
from agent_env.env.env import Env

SERVICE_DB_PORT = 5432
DB_USER = "agentenv"
DB_PASSWORD = "agentenv"
DB_NAME = "agentenv"


@dataclass
class ServiceDBConfig:
    """Image configuration for the shared PostgreSQL database services."""
    db_image: str = "public.ecr.aws/docker/library/postgres:16-alpine"
    db_web_image: str | None = None
    db_mcp_image: str | None = None


class ServiceDBEnv(Env):
    """PostgreSQL database env for holding env state"""
    type: ClassVar[str] = "service_db"

    def __init__(
        self,
        id: str,
        db_docker_image_artifact: DockerImageArtifact,
        db_web_docker_image_artifact: DockerImageArtifact,
        db_mcp_docker_image_artifact: DockerImageArtifact,
        version: Optional[int] = None,
        metadata: Optional[dict[str, str]] = None,
    ):
        super().__init__(id, version, metadata=metadata)
        self.db_docker_image_artifact = db_docker_image_artifact
        self.db_web_docker_image_artifact = db_web_docker_image_artifact
        self.db_mcp_docker_image_artifact = db_mcp_docker_image_artifact

    def to_config(self) -> ServiceDBConfig:
        return ServiceDBConfig(
            db_image=self.db_docker_image_artifact.image_name,
            db_web_image=self.db_web_docker_image_artifact.image_name,
            db_mcp_image=self.db_mcp_docker_image_artifact.image_name,
        )

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["db_docker_image_artifact"] = {
            "id": self.db_docker_image_artifact.id,
            "version": self.db_docker_image_artifact.version,
            "type": self.db_docker_image_artifact.type,
        }
        base["db_web_docker_image_artifact"] = {
            "id": self.db_web_docker_image_artifact.id,
            "version": self.db_web_docker_image_artifact.version,
            "type": self.db_web_docker_image_artifact.type,
        }
        base["db_mcp_docker_image_artifact"] = {
            "id": self.db_mcp_docker_image_artifact.id,
            "version": self.db_mcp_docker_image_artifact.version,
            "type": self.db_mcp_docker_image_artifact.type,
        }
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "ServiceDBEnv":
        db_ref = data["db_docker_image_artifact"]
        db_docker_image_artifact = Artifact.get(db_ref["id"], version=db_ref.get("version"))
        db_web_ref = data["db_web_docker_image_artifact"]
        db_web_docker_image_artifact = Artifact.get(db_web_ref["id"], version=db_web_ref.get("version"))
        db_mcp_ref = data["db_mcp_docker_image_artifact"]
        db_mcp_docker_image_artifact = Artifact.get(db_mcp_ref["id"], version=db_mcp_ref.get("version"))
        return cls(
            id=data["id"],
            db_docker_image_artifact=db_docker_image_artifact,
            db_web_docker_image_artifact=db_web_docker_image_artifact,
            db_mcp_docker_image_artifact=db_mcp_docker_image_artifact,
            version=data.get("version"),
            metadata=data.get("metadata", {}),
        )
