"""agentenv's ATIF profile: what agentenv adds under ``extra.agentenv`` on the objects ATIF gives an ``extra``. Every
field is optional, for readers and producers, and a document with the profile removed is still correct ATIF that means
the same; PROFILE.md beside this module is the spec. The models are tolerant, so a reader keeps keys it doesn't know and
writes them back; strictness for producers belongs to the validator."""
from __future__ import annotations

from typing import Any, Mapping, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

#: The format id an agent serves and the worker stores.
FORMAT = "atif/1"
#: The key under each ``extra`` that holds the profile.
PROFILE_KEY = "agentenv"
PROFILE_VERSION = "1.0"
#: The ATIF version producers write: the oldest that holds the fields they use.
SCHEMA_VERSION = "ATIF-v1.8"
#: The only key outside ``extra.agentenv`` a producer writes: Harbor's, on a ToolCall whose input wasn't an object.
RAW_ARGUMENTS = "raw_arguments"

#: Vocabularies. A reader keeps a value it doesn't know.
TASK_STATES = ("completed", "failed", "canceled", "rejected")
RESULT_STATUSES = ("ok", "error", "canceled", "timeout")
ERROR_KINDS = ("agent_error", "infra_error")
VISIBILITIES = ("raw", "summary", "omitted", "redacted")
COST_SOURCES = ("harness", "provider")
VOCABULARIES = {
    "StepProfile.state": TASK_STATES,
    "ObservationResultProfile.status": RESULT_STATUSES,
    "StepError.kind": ERROR_KINDS,
    "ReasoningBlock.visibility": VISIBILITIES,
    "MetricsProfile.cost_source": COST_SOURCES,
    "FinalMetricsProfile.cost_source": COST_SOURCES,
}


class AtifError(ValueError):
    """A document, profile or adapter that breaks ATIF or the profile."""


class _Profile(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())


class AdapterRef(_Profile):
    name: str | None = None
    version: str | None = None


class MediaInfo(_Profile):
    """A media file the parts beside it name by ``media/<sha256>.<ext>``. ``media_type`` is written only for media ATIF
    has no part type for, since an image or audio part carries its own."""

    media_type: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    pages: int | None = None
    name: str | None = None


class ReasoningBlock(_Profile):
    """One reasoning block of a step, in native order. ``chars`` is its share of ``reasoning_content`` (0 when the
    provider withheld the text); ``data`` holds a redacted block's ciphertext byte for byte."""

    chars: int | None = None
    visibility: str | None = None
    signature: str | None = None
    data: str | None = None


class StepError(_Profile):
    kind: str | None = None
    message: str | None = None
    code: str | None = None


class Arrival(_Profile):
    """When a result arrived after its call's step had closed: its time, and the step open when it came."""

    at: str | None = None
    step_id: int | None = None


class TrajectoryProfile(_Profile):
    version: str | None = None
    native_format: str | None = None
    adapter: AdapterRef | None = None
    unmatched: dict[str, int] | None = None


class AgentProfile(_Profile):
    pass


class StepData(_Profile):
    """A DataPart of a task's final response that isn't its answer, such as a harness's diagnostics, kept whole."""

    data: Any = None
    metadata: dict[str, Any] | None = None


class StepProfile(_Profile):
    context_id: str | None = None
    task_id: str | None = None
    message_id: str | None = None
    turn: int | None = None
    state: str | None = None
    attempt: int | None = None
    error: StepError | None = None
    reasoning: list[ReasoningBlock] | None = None
    media: dict[str, MediaInfo] | None = None
    data: list[StepData] | None = None


class ToolCallProfile(_Profile):
    pass


class ObservationResultProfile(_Profile):
    status: str | None = None
    exit_code: int | None = None
    arrived: Arrival | None = None
    media: dict[str, MediaInfo] | None = None


class MetricsProfile(_Profile):
    cost_source: str | None = None
    cache_write_tokens: int | None = None


class FinalMetricsProfile(_Profile):
    cost_source: str | None = None
    total_cache_write_tokens: int | None = None


class SubagentTrajectoryRefProfile(_Profile):
    pass


#: The profile model for each ATIF object that has an ``extra``.
PROFILES: dict[str, type[_Profile]] = {
    "Trajectory": TrajectoryProfile,
    "Agent": AgentProfile,
    "Step": StepProfile,
    "ToolCall": ToolCallProfile,
    "ObservationResult": ObservationResultProfile,
    "Metrics": MetricsProfile,
    "FinalMetrics": FinalMetricsProfile,
    "SubagentTrajectoryRef": SubagentTrajectoryRefProfile,
}

P = TypeVar("P", bound=_Profile)


def profile_of(obj: Mapping[str, Any] | None) -> dict[str, Any]:
    """The raw ``extra.agentenv`` dict of an ATIF object as a dict, or an empty dict when it has none."""
    extra = (obj or {}).get("extra")
    value = extra.get(PROFILE_KEY) if isinstance(extra, Mapping) else None
    return dict(value) if isinstance(value, Mapping) else {}


def read_profile(obj: Mapping[str, Any] | None, model: type[P]) -> P:
    """An ATIF object's profile as ``model``, unknown keys kept."""
    try:
        return model.model_validate(profile_of(obj))
    except ValidationError as exc:
        raise AtifError(f"extra.{PROFILE_KEY}: {exc}") from exc


def dump_profile(profile: _Profile) -> dict[str, Any]:
    """A profile as JSON-ready data: what was set, unknown keys included, as it was read."""
    return profile.model_dump(mode="json", exclude_unset=True)
