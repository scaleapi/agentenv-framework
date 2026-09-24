"""Unit tests for PromptAgentTaskStep multimodal `parts` field."""

import pytest

from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep


def _text_part(text):
    return {"kind": "text", "text": text}


def _file_part_uri(uri, mime="image/png", name="x.png"):
    return {"kind": "file", "file": {"uri": uri, "mimeType": mime, "name": name}}


def _file_part_bytes(b64, mime="image/png", name="x.png"):
    return {"kind": "file", "file": {"bytes": b64, "mimeType": mime, "name": name}}


class TestPromptAgentPartsConstruction:
    def test_legacy_prompt_only(self):
        step = PromptAgentTaskStep(id="t", version=None, prompt="Hello")
        assert step.prompt == "Hello"
        assert step.parts is None

    def test_parts_only(self):
        parts = [_text_part("Describe this"), _file_part_uri("s3://b/k.png")]
        step = PromptAgentTaskStep(id="t", version=None, parts=parts)
        assert step.prompt is None
        assert step.parts == parts

    def test_both_set_raises(self):
        with pytest.raises(ValueError, match="`prompt` OR `parts`, not both"):
            PromptAgentTaskStep(
                id="t", version=None, prompt="Hi", parts=[_text_part("Hi")],
            )

    def test_neither_set_raises(self):
        with pytest.raises(ValueError, match="requires either `prompt` or `parts`"):
            PromptAgentTaskStep(id="t", version=None)

    def test_invalid_part_dict_raises(self):
        # Missing required `text` field on a text part
        with pytest.raises(ValueError, match="parts\\[0\\] is not a valid A2A Part"):
            PromptAgentTaskStep(
                id="t", version=None,
                parts=[{"kind": "text"}],
            )

    def test_unknown_part_kind_raises(self):
        with pytest.raises(ValueError, match="parts\\[0\\] is not a valid A2A Part"):
            PromptAgentTaskStep(
                id="t", version=None,
                parts=[{"kind": "videogram", "data": "x"}],
            )

    def test_parts_not_a_list_raises(self):
        with pytest.raises(ValueError, match="`parts` must be a list"):
            PromptAgentTaskStep(id="t", version=None, parts={"kind": "text", "text": "Hi"})

    def test_parts_entry_not_a_dict_raises(self):
        with pytest.raises(ValueError, match="parts\\[0\\] must be a dict"):
            PromptAgentTaskStep(id="t", version=None, parts=["not-a-dict"])


class TestPromptAgentPartsSerialization:
    def test_parts_roundtrip(self):
        parts = [
            _text_part("What is in this image?"),
            _file_part_uri("s3://bucket/img.png"),
        ]
        step = PromptAgentTaskStep(id="t", version=None, parts=parts, prompt_id="p")
        d = step.to_dict()
        assert d["parts"] == parts
        assert d["prompt"] is None

        restored = PromptAgentTaskStep.from_dict(d)
        assert restored.parts == parts
        assert restored.prompt is None

    def test_legacy_prompt_roundtrip(self):
        step = PromptAgentTaskStep(id="t", version=None, prompt="Hello", prompt_id="p")
        d = step.to_dict()
        assert d["prompt"] == "Hello"
        assert d["parts"] is None

        restored = PromptAgentTaskStep.from_dict(d)
        assert restored.prompt == "Hello"
        assert restored.parts is None

    def test_from_dict_legacy_dict_without_parts_key(self):
        """Mongo docs predating this change have no `parts` key — must still deserialize."""
        data = {
            "id": "t", "type": "prompt_agent", "version": 1,
            "prompt": "Hello",
            # no "parts" key
            "prompt_id": "p",
            "agent_name": "default-agent",
        }
        step = PromptAgentTaskStep.from_dict(data)
        assert step.prompt == "Hello"
        assert step.parts is None

    def test_to_dict_always_includes_parts_key(self):
        step = PromptAgentTaskStep(id="t", version=None, prompt="Hi")
        d = step.to_dict()
        assert "parts" in d
        assert d["parts"] is None


class TestEffectiveParts:
    def test_legacy_materializes_singleton_text_part(self):
        step = PromptAgentTaskStep(id="t", version=None, prompt="Hello world")
        assert step._effective_parts() == [{"kind": "text", "text": "Hello world"}]

    def test_parts_returned_verbatim(self):
        parts = [_text_part("A"), _file_part_uri("s3://b/x.png"), _text_part("B")]
        step = PromptAgentTaskStep(id="t", version=None, parts=parts)
        assert step._effective_parts() == parts
        # Must be the same object reference (caller is expected to deep-copy via _apply_seed)
        assert step._effective_parts() is step.parts


class TestApplySeed:
    def test_no_seed_returns_input_unchanged(self):
        step = PromptAgentTaskStep(id="t", version=None, prompt="Hi <name>")
        parts = step._effective_parts()
        assert step._apply_seed(parts, {}) is parts  # no copy when no seed

    def test_substitutes_text_part(self):
        step = PromptAgentTaskStep(
            id="t", version=None,
            parts=[_text_part("Hello <name>, your code is <code>")],
        )
        out = step._apply_seed(step._effective_parts(), {"name": "Alice", "code": "42"})
        assert out[0]["text"] == "Hello Alice, your code is 42"
        # Original parts not mutated (deep-copied)
        assert step.parts[0]["text"] == "Hello <name>, your code is <code>"

    def test_substitutes_file_uri_and_name(self):
        step = PromptAgentTaskStep(
            id="t", version=None,
            parts=[_file_part_uri(
                "s3://bucket/seeds/<seed_id>/input.png",
                name="<seed_id>_input.png",
            )],
        )
        out = step._apply_seed(step._effective_parts(), {"seed_id": "row42"})
        assert out[0]["file"]["uri"] == "s3://bucket/seeds/row42/input.png"
        assert out[0]["file"]["name"] == "row42_input.png"

    def test_does_not_substitute_bytes_field(self):
        """Seed substitution must NOT touch base64 bytes — they're binary content."""
        step = PromptAgentTaskStep(
            id="t", version=None,
            parts=[_file_part_bytes("<seed_id>BASE64DATA<seed_id>")],
        )
        out = step._apply_seed(step._effective_parts(), {"seed_id": "X"})
        # bytes field is unchanged — only uri and name get substituted
        assert out[0]["file"]["bytes"] == "<seed_id>BASE64DATA<seed_id>"

    def test_mixed_parts_substitution(self):
        step = PromptAgentTaskStep(
            id="t", version=None,
            parts=[
                _text_part("Look at <asset>"),
                _file_part_uri("s3://b/<asset>", name="<asset>"),
                _text_part("and report"),
            ],
        )
        out = step._apply_seed(step._effective_parts(), {"asset": "red.png"})
        assert out[0]["text"] == "Look at red.png"
        assert out[1]["file"]["uri"] == "s3://b/red.png"
        assert out[1]["file"]["name"] == "red.png"
        assert out[2]["text"] == "and report"

    def test_legacy_prompt_seed_substitution_via_effective_parts(self):
        """The materialized singleton text part should also get substituted —
        this preserves the existing legacy seed behavior."""
        step = PromptAgentTaskStep(id="t", version=None, prompt="Email <recipient>")
        out = step._apply_seed(step._effective_parts(), {"recipient": "alice@example.com"})
        assert out[0]["text"] == "Email alice@example.com"
