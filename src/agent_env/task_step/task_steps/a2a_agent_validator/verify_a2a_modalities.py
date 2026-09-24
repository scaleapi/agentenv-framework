"""Verify which input modalities an A2A agent functionally supports.

Covers `text`, `image/{png,jpeg,gif}`, `audio/{wav,mpeg,mp4,ogg}`,
`application/pdf`, and `video/mp4`.
The companion `PromptAgentTaskStep` probes (configured in
`A2AAgent.validate()`) actually send the messages via A2A; this step purely
grades the resulting `PromptResponse`s.

Grading is a deterministic case-insensitive substring match against a known
ground-truth string per probe. The outcome for each modality is one of:
- `passed`: response contains the expected string.
- `no_ingestion_evidence`: response succeeded but didn't reference the asset.
- `protocol_reject`: response was appended with `error_type` (wrapper accepted
  the message but the agent task ended in `failed`).
- `probe_step_did_not_run`: no matching response — the upstream
  `PromptAgentTaskStep` crashed before `context.prompt_responses.append(...)`.

The gemini wrapper currently passes all 12 modality probes (text, image
{png,jpeg,gif}, audio {wav,mpeg,mp4,ogg}, PDF, video/mp4, plus image/png
delivered via `FileWithUri` over `s3://` and presigned `https://`). New
A2A wrappers (claude-code, codex, grok) inherit this same probe set and
will report which subset they actually ingest.
"""

from __future__ import annotations

import base64
import functools
import logging
from pathlib import Path
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


_FIXTURES_DIR = Path(__file__).parent / "fixtures"


@functools.lru_cache(maxsize=None)
def _b64(name: str) -> str:
    """Load a binary fixture and return its base64-encoded form.

    Fixtures live as native binary files in `fixtures/` so they can be
    inspected (e.g. `afplay clip.wav`, `qlmanage -p document.pdf`) and
    replaced in-place without re-chunking base64 into Python source.
    """
    return base64.b64encode((_FIXTURES_DIR / name).read_bytes()).decode("ascii")


# 8×8 solid red squares in three formats — unambiguously red so the probe's
# deterministic "red" substring match holds. See `fixtures/README.md`.
IMAGE_PROBE_PNG_B64 = _b64("red.png")
IMAGE_PROBE_JPEG_B64 = _b64("red.jpg")
IMAGE_PROBE_GIF_B64 = _b64("red.gif")

# Per-format spoken "Say the word: <X>" clips with distinct target words —
# WAV/banana, MP3/telephone, M4A/bicycle, OGG/rainbow. Distinct targets force
# the model to actually decode each clip. See `fixtures/README.md` for the
# encoding pipeline, voice/volume/padding rationale, and regen commands.
AUDIO_PROBE_WAV_B64 = _b64("clip.wav")
AUDIO_PROBE_MP3_B64 = _b64("clip.mp3")
AUDIO_PROBE_M4A_B64 = _b64("clip.m4a")
AUDIO_PROBE_OGG_B64 = _b64("clip.ogg")

TEXT_PROBE_PROMPT = "Reply with exactly the word PINEAPPLE and nothing else."
TEXT_PROBE_EXPECTED = "PINEAPPLE"

IMAGE_PROBE_PROMPT = (
    "What is the dominant color of the image attached? Answer in one word."
)
IMAGE_PROBE_EXPECTED = "red"

AUDIO_PROBE_PROMPT = (
    "An audio clip is attached. Follow the spoken instruction in it and "
    "respond with only the requested word."
)
AUDIO_WAV_EXPECTED = "banana"
AUDIO_MP3_EXPECTED = "telephone"
AUDIO_M4A_EXPECTED = "bicycle"
AUDIO_OGG_EXPECTED = "rainbow"

# 1.1KB single-page PDF with "ELEPHANT" inside an instructional paragraph
# (bare word alone trips letter-truncation edge cases). See `fixtures/README.md`.
PDF_PROBE_PDF_B64 = _b64("document.pdf")
PDF_PROBE_PROMPT = (
    "A PDF document is attached. Read it and reply with only the target "
    "word it specifies."
)
PDF_PROBE_EXPECTED = "elephant"

# 8.7KB h.264 mp4 (3s @ 480×360, 15fps, no audio) showing "GIRAFFE" on a
# static frame with on-screen instructional context. Delivered via
# FileWithUri (s3://) at validator runtime. See `fixtures/README.md`.
VIDEO_PROBE_MP4_B64 = _b64("clip.mp4")
VIDEO_PROBE_PROMPT = (
    "A short video clip is attached. Read the target word displayed in "
    "the video frames and reply with only that word."
)
VIDEO_PROBE_EXPECTED = "giraffe"

TEXT_PROBE_PARTS: list[dict] = [{"kind": "text", "text": TEXT_PROBE_PROMPT}]
IMAGE_PROBE_PARTS: list[dict] = [
    {"kind": "text", "text": IMAGE_PROBE_PROMPT},
    {"kind": "file", "file": {
        "bytes": IMAGE_PROBE_PNG_B64,
        "mimeType": "image/png",
        "name": "red.png",
    }},
]
IMAGE_JPEG_PROBE_PARTS: list[dict] = [
    {"kind": "text", "text": IMAGE_PROBE_PROMPT},
    {"kind": "file", "file": {
        "bytes": IMAGE_PROBE_JPEG_B64,
        "mimeType": "image/jpeg",
        "name": "red.jpg",
    }},
]
IMAGE_GIF_PROBE_PARTS: list[dict] = [
    {"kind": "text", "text": IMAGE_PROBE_PROMPT},
    {"kind": "file", "file": {
        "bytes": IMAGE_PROBE_GIF_B64,
        "mimeType": "image/gif",
        "name": "red.gif",
    }},
]
AUDIO_WAV_PROBE_PARTS: list[dict] = [
    {"kind": "text", "text": AUDIO_PROBE_PROMPT},
    {"kind": "file", "file": {
        "bytes": AUDIO_PROBE_WAV_B64,
        "mimeType": "audio/wav",
        "name": "clip.wav",
    }},
]
AUDIO_MP3_PROBE_PARTS: list[dict] = [
    {"kind": "text", "text": AUDIO_PROBE_PROMPT},
    {"kind": "file", "file": {
        "bytes": AUDIO_PROBE_MP3_B64,
        "mimeType": "audio/mpeg",
        "name": "clip.mp3",
    }},
]
AUDIO_M4A_PROBE_PARTS: list[dict] = [
    {"kind": "text", "text": AUDIO_PROBE_PROMPT},
    {"kind": "file", "file": {
        "bytes": AUDIO_PROBE_M4A_B64,
        "mimeType": "audio/mp4",
        "name": "clip.m4a",
    }},
]
AUDIO_OGG_PROBE_PARTS: list[dict] = [
    {"kind": "text", "text": AUDIO_PROBE_PROMPT},
    {"kind": "file", "file": {
        "bytes": AUDIO_PROBE_OGG_B64,
        "mimeType": "audio/ogg",
        "name": "clip.ogg",
    }},
]
PDF_PROBE_PARTS: list[dict] = [
    {"kind": "text", "text": PDF_PROBE_PROMPT},
    {"kind": "file", "file": {
        "bytes": PDF_PROBE_PDF_B64,
        "mimeType": "application/pdf",
        "name": "document.pdf",
    }},
]


class VerifyA2AModalitiesStep(TaskStep):
    """Grade modality probes by reading `context.prompt_responses`.

    `probes` is a list of dicts, each `{"modality": str, "prompt_id": str,
    "expected": str}`. Each `prompt_id` should match an upstream
    `PromptAgentTaskStep`'s `prompt_id`.
    """

    type: ClassVar[str] = "verify_a2a_modalities"
    entity_refs = (EntityRef.agent("a2a_agent_id"),)

    def __init__(
        self,
        id: str,
        version: Optional[int],
        a2a_agent_id: str,
        probes: list[dict],
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.a2a_agent_id = a2a_agent_id
        self.probes = probes

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["a2a_agent_id"] = self.a2a_agent_id
        base["probes"] = self.probes
        return base

    @classmethod
    def from_dict(cls, data: dict) -> VerifyA2AModalitiesStep:
        return cls(
            **cls._base_from_dict(data),
            a2a_agent_id=data["a2a_agent_id"],
            probes=data["probes"],
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent

        results: dict[str, dict] = {}
        for probe in self.probes:
            modality = probe["modality"]
            prompt_id = probe["prompt_id"]
            expected = probe["expected"]
            response = next(
                (r for r in context.prompt_responses if r.prompt_id == prompt_id),
                None,
            )
            if response is None:
                # Upstream PromptAgentTaskStep crashed before appending its response
                # (e.g., wrapper returned HTTP 4xx on message/send).
                results[modality] = {
                    "supported": False,
                    "reason": "probe_step_did_not_run",
                }
            elif response.error_type:
                results[modality] = {
                    "supported": False,
                    "reason": "protocol_reject",
                    "error_type": response.error_type,
                    "probe_response": (response.response or "")[:200],
                }
            elif expected.lower() in (response.response or "").lower():
                results[modality] = {
                    "supported": True,
                    "reason": "passed",
                    "probe_response": response.response[:200],
                }
            else:
                results[modality] = {
                    "supported": False,
                    "reason": "no_ingestion_evidence",
                    "probe_response": (response.response or "")[:200],
                }
            logger.info(
                f"Modality '{modality}' graded: supported={results[modality]['supported']} "
                f"reason={results[modality]['reason']}"
            )

        # Echo what the AgentCard declared, for cross-reference with the
        # actual probe results.
        agent = A2AAgent.get(self.a2a_agent_id)
        card = agent.metadata.get("agent_card") or {}
        declared = {
            "default_input_modes": card.get("defaultInputModes"),
            "default_output_modes": card.get("defaultOutputModes"),
        }

        agent.update_metadata({
            **agent.metadata,
            "validated_modalities": {"input": results, "declared": declared},
        })
        context.metadata.setdefault("verifications", {})["a2a_modalities"] = {
            "input": results,
            "declared": declared,
        }
        return context
