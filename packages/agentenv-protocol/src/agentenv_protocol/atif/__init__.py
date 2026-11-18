"""ATIF in agentenv: Harbor's Agent Trajectory Interchange Format (v1.8), the record every agent serves beside its
native log as ``atif/1``, plus agentenv's optional ``extra.agentenv`` profile. ``models`` holds Harbor's own models,
vendored; ``PROFILE.md`` beside this module is the profile's spec. The package needs only the protocol's base
dependencies, so anything can read a document or write an adapter."""
from .adapter import Adapter, AdapterSpec, Native, Selector, atif_adapter, aux, finish, native
from .builder import AtifTurn, Document, Media, TurnContext, timestamp
from .models import (
    Agent,
    AudioSource,
    ContentPart,
    FinalMetrics,
    ImageSource,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)
from .profile import (
    FORMAT,
    PROFILE_KEY,
    PROFILE_VERSION,
    PROFILES,
    RAW_ARGUMENTS,
    SCHEMA_VERSION,
    VOCABULARIES,
    AdapterRef,
    AgentProfile,
    Arrival,
    AtifError,
    FinalMetricsProfile,
    MediaInfo,
    MetricsProfile,
    ObservationResultProfile,
    ReasoningBlock,
    StepError,
    StepProfile,
    SubagentTrajectoryRefProfile,
    ToolCallProfile,
    TrajectoryProfile,
    dump_profile,
    profile_of,
    read_profile,
)
from .reader import Task, check, is_prefix, load, media_paths, strip_profile, tasks
from .runner import AdapterRun, Fed, RecordedTask, replay

__all__ = [
    "Adapter", "AdapterRef", "AdapterRun", "AdapterSpec", "Agent", "AgentProfile", "Arrival", "AtifError", "AtifTurn",
    "AudioSource", "ContentPart", "Document", "FORMAT", "Fed", "FinalMetrics", "FinalMetricsProfile", "ImageSource",
    "Media", "MediaInfo", "Metrics", "MetricsProfile", "Native", "Observation", "ObservationResult",
    "ObservationResultProfile", "PROFILES", "PROFILE_KEY", "PROFILE_VERSION", "RAW_ARGUMENTS", "ReasoningBlock",
    "RecordedTask", "SCHEMA_VERSION", "Selector", "Step", "StepError", "StepProfile", "SubagentTrajectoryRef",
    "SubagentTrajectoryRefProfile", "Task", "ToolCall", "ToolCallProfile", "Trajectory", "TrajectoryProfile",
    "TurnContext", "VOCABULARIES", "atif_adapter", "aux", "check", "dump_profile", "finish", "is_prefix", "load",
    "media_paths", "native", "profile_of", "read_profile", "replay", "strip_profile", "tasks", "timestamp",
]
