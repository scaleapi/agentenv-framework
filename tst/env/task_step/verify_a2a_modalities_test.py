"""Unit tests for VerifyA2AModalitiesStep grading classification."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_modalities import (
    VerifyA2AModalitiesStep,
    TEXT_PROBE_EXPECTED,
    TEXT_PROBE_PARTS,
    IMAGE_PROBE_EXPECTED,
    IMAGE_PROBE_PARTS,
)


PROBES = [
    {"modality": "text",       "prompt_id": "text-probe",  "expected": "PINEAPPLE"},
    {"modality": "image/png",  "prompt_id": "image-probe", "expected": "red"},
]


def _make_step():
    return VerifyA2AModalitiesStep(
        id="t", version=None,
        a2a_agent_id="agent-x",
        probes=PROBES,
    )


def _make_context(responses):
    ctx = TaskStepContext()
    ctx.prompt_responses = list(responses)
    return ctx


def _run(step, ctx):
    """Execute the async step with a patched A2AAgent so we don't hit Mongo."""
    fake_agent = MagicMock()
    fake_agent.metadata = {"agent_card": {"defaultInputModes": ["text"], "defaultOutputModes": ["text"]}}
    fake_agent.update_metadata = MagicMock()
    with patch("agent_env.a2a_agent.A2AAgent.get", return_value=fake_agent):
        result_ctx = asyncio.run(step.execute(ctx))
    return result_ctx, fake_agent


class TestModalityGrading:
    def test_passed(self):
        ctx = _make_context([
            PromptResponse(prompt_id="text-probe",  response="PINEAPPLE"),
            PromptResponse(prompt_id="image-probe", response="It's red."),
        ])
        _, agent = _run(_make_step(), ctx)
        update = agent.update_metadata.call_args[0][0]
        results = update["validated_modalities"]["input"]
        assert results["text"]["supported"] is True
        assert results["text"]["reason"] == "passed"
        assert results["image/png"]["supported"] is True
        assert results["image/png"]["reason"] == "passed"

    def test_no_ingestion_evidence(self):
        ctx = _make_context([
            PromptResponse(prompt_id="text-probe",  response="PINEAPPLE"),
            PromptResponse(prompt_id="image-probe", response="I don't see any image attached."),
        ])
        _, agent = _run(_make_step(), ctx)
        results = agent.update_metadata.call_args[0][0]["validated_modalities"]["input"]
        assert results["text"]["supported"] is True
        assert results["image/png"]["supported"] is False
        assert results["image/png"]["reason"] == "no_ingestion_evidence"

    def test_protocol_reject(self):
        ctx = _make_context([
            PromptResponse(prompt_id="text-probe",  response="PINEAPPLE"),
            PromptResponse(prompt_id="image-probe", response="", error_type="cli_error",
                           error_message="wrapper rejected file part"),
        ])
        _, agent = _run(_make_step(), ctx)
        results = agent.update_metadata.call_args[0][0]["validated_modalities"]["input"]
        assert results["image/png"]["supported"] is False
        assert results["image/png"]["reason"] == "protocol_reject"
        assert results["image/png"]["error_type"] == "cli_error"

    def test_probe_step_did_not_run(self):
        # Only the text response exists; image probe step never appended a response
        ctx = _make_context([
            PromptResponse(prompt_id="text-probe", response="PINEAPPLE"),
        ])
        _, agent = _run(_make_step(), ctx)
        results = agent.update_metadata.call_args[0][0]["validated_modalities"]["input"]
        assert results["text"]["supported"] is True
        assert results["image/png"]["supported"] is False
        assert results["image/png"]["reason"] == "probe_step_did_not_run"

    def test_case_insensitive_match(self):
        ctx = _make_context([
            PromptResponse(prompt_id="text-probe",  response="pineapple"),       # lowercase
            PromptResponse(prompt_id="image-probe", response="THE COLOR IS RED"),# uppercase
        ])
        _, agent = _run(_make_step(), ctx)
        results = agent.update_metadata.call_args[0][0]["validated_modalities"]["input"]
        assert results["text"]["supported"] is True
        assert results["image/png"]["supported"] is True

    def test_declared_modes_echoed(self):
        ctx = _make_context([
            PromptResponse(prompt_id="text-probe",  response="PINEAPPLE"),
            PromptResponse(prompt_id="image-probe", response="red"),
        ])
        _, agent = _run(_make_step(), ctx)
        update = agent.update_metadata.call_args[0][0]
        assert update["validated_modalities"]["declared"] == {
            "default_input_modes": ["text"],
            "default_output_modes": ["text"],
        }

    def test_writes_to_context_metadata(self):
        ctx = _make_context([
            PromptResponse(prompt_id="text-probe",  response="PINEAPPLE"),
            PromptResponse(prompt_id="image-probe", response="red"),
        ])
        result_ctx, _ = _run(_make_step(), ctx)
        assert "a2a_modalities" in result_ctx.metadata["verifications"]


class TestSerialization:
    def test_roundtrip(self):
        step = _make_step()
        d = step.to_dict()
        assert d["a2a_agent_id"] == "agent-x"
        assert d["probes"] == PROBES
        assert d["type"] == "verify_a2a_modalities"

        restored = VerifyA2AModalitiesStep.from_dict(d)
        assert restored.a2a_agent_id == "agent-x"
        assert restored.probes == PROBES
        assert restored.id == "t"

    def test_fixture_constants_well_formed(self):
        # Sanity: the probe parts are valid A2A part dicts (text + file shapes)
        assert TEXT_PROBE_PARTS[0]["kind"] == "text"
        assert TEXT_PROBE_EXPECTED == "PINEAPPLE"
        assert IMAGE_PROBE_PARTS[0]["kind"] == "text"
        assert IMAGE_PROBE_PARTS[1]["kind"] == "file"
        assert IMAGE_PROBE_PARTS[1]["file"]["mimeType"] == "image/png"
        assert IMAGE_PROBE_PARTS[1]["file"]["bytes"]  # non-empty base64
        assert IMAGE_PROBE_EXPECTED == "red"
