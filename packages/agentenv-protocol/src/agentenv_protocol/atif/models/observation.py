# From Harbor 0.26.0, harbor/models/trajectories/observation.py (Apache-2.0), with its import paths changed: see VENDORED.md.
"""Observation model for ATIF trajectories."""

from pydantic import BaseModel, Field

from agentenv_protocol.atif.models.observation_result import ObservationResult


class Observation(BaseModel):
    """Environment feedback/result after actions or system events."""

    results: list[ObservationResult] = Field(
        default=...,
        description="Array of result objects from tool calls or actions",
    )

    model_config = {"extra": "forbid"}
