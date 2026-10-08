"""Artifact module for AgentEnv."""

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.ref import ArtifactRef
from agent_env.artifact.artifacts.cli import CliArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.artifacts.environment_universe import EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.skill import AGENT_SKILLS_SPEC_VERSION, SkillArtifact
from agent_env.artifact.registry import (
    ARTIFACT_REGISTRY,
    canonical_type,
    equivalent_types,
    get_artifact_registry,
    get_type_aliases,
)
from agent_env.artifact.store import ArtifactStore, get_artifact_store, reset_artifact_store, set_artifact_store
from agent_env.artifact.universe import Universe

__all__ = [
    "Artifact",
    "ArtifactRef",
    "CliArtifact",
    "DockerImageArtifact",
    "FileArtifact",
    "FileArtifactUniverse",
    "EnvironmentArtifact",
    "EnvironmentUniverseArtifact",
    # Deprecated aliases. Kept in __all__ so `from agent_env.artifact import *`
    # still binds them; served by __getattr__ below rather than imported eagerly, so
    # importing this package does not itself trip the deprecation counter.
    "SkillArtifact",
    "AGENT_SKILLS_SPEC_VERSION",
    "Universe",
    "ARTIFACT_REGISTRY",
    "canonical_type",
    "equivalent_types",
    "get_artifact_registry",
    "get_type_aliases",
    "ArtifactStore",
    "get_artifact_store",
    "reset_artifact_store",
    "set_artifact_store",
]
