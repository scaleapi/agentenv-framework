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
    StepData,
    StepError,
    StepProfile,
    SubagentTrajectoryRefProfile,
    ToolCallProfile,
    TrajectoryProfile,
    dump_profile,
    profile_of,
    read_profile,
)
from .kit import KIT_RULES, KitResult, check_adapter
from .reader import Task, check, is_prefix, load, media_paths, strip_profile, tasks
from .runner import AdapterRun, Fed, RecordedTask, replay
from .served import AuxEntry, ContentItem, MediaItem, ServedData, ServedError, ServedMedia, ServedTurn
from .validator import RULES, Finding, Rule, errors_in, validate, validate_live

__all__ = [
    "Adapter", "AdapterRef", "AdapterRun", "AdapterSpec", "Agent", "AgentProfile", "Arrival", "atif_adapter",
    "AtifError", "AtifTurn", "AudioSource", "aux", "AuxEntry", "check", "check_adapter", "ContentItem", "ContentPart",
    "Document", "dump_profile", "errors_in", "Fed", "FinalMetrics", "FinalMetricsProfile", "Finding", "finish",
    "FORMAT", "ImageSource", "is_prefix", "KIT_RULES", "KitResult", "load", "Media", "media_paths", "MediaInfo",
    "MediaItem", "Metrics", "MetricsProfile", "Native", "native", "Observation", "ObservationResult",
    "ObservationResultProfile", "PROFILE_KEY", "profile_of", "PROFILE_VERSION", "PROFILES", "RAW_ARGUMENTS",
    "read_profile", "ReasoningBlock", "RecordedTask", "replay", "Rule", "RULES", "SCHEMA_VERSION", "Selector",
    "ServedData", "ServedError", "ServedMedia", "ServedTurn", "Step", "StepData", "StepError", "StepProfile",
    "strip_profile", "SubagentTrajectoryRef", "SubagentTrajectoryRefProfile", "Task", "tasks", "timestamp", "ToolCall",
    "ToolCallProfile", "Trajectory", "TrajectoryProfile", "TurnContext", "validate", "validate_live", "VOCABULARIES",
]
