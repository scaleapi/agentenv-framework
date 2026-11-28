"""browser-use's ATIF adapter on Pydantic AI 2.66 messages as browser-use 1.0.3 dumps them, and on the aux records the
server feeds before them (the turn, the model requests agent.py saw, compactions' summaries and pruned screenshots).
Each case runs through the protocol's adapter kit, so the live rule holds after every record and the document
validates as a producer's."""
from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

import pytest
from agentenv_protocol.atif import Fed, RecordedTask, TurnContext, check_adapter, errors_in, profile_of, validate

from tst.unit.agents.browser_use.agent import AGENT_DIR, a2a_server, browser_use_atif

BrowserUseAtif = browser_use_atif.BrowserUseAtif
MODEL = "claude-haiku-4-5-20251001"
CONVERSATION = "01a220cf-c1cc-71e5-a71e-61ce74a0f827"
PROMPT = "Open https://example.com and reply with its heading."
JPEG = b"\xff\xd8\xff\xe0" + bytes(range(64))
PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x04\x00\x00\x00\x03\x08\x02\x00\x00\x00"


def binary(data: bytes, media_type: str = "image/jpeg") -> dict:
    """An image as Pydantic AI dumps it: base64 in the URL-safe alphabet, its identifier its sha1's first digits."""
    return {"data": base64.urlsafe_b64encode(data).decode(), "media_type": media_type, "vendor_metadata": None,
            "kind": "binary", "identifier": hashlib.sha1(data).hexdigest()[:6]}


def media_path(data: bytes, extension: str = ".jpg") -> str:
    return f"media/{hashlib.sha256(data).hexdigest()}{extension}"


def usage(input_tokens=1200, output=40, cache_read=1000, cache_write=100, cost: str | None = "0.0012",
          unmeasured=0) -> dict:
    counts = {"input_tokens": input_tokens, "cache_write_tokens": cache_write, "cache_write_1h_tokens": 0,
              "cache_read_tokens": cache_read, "output_tokens": output, "output_reasoning_tokens": 0,
              "input_audio_tokens": 0, "cache_audio_read_tokens": 0, "output_audio_tokens": 0, "audio_seconds": 0.0,
              "details": {}, "cost": cost}
    if unmeasured:
        counts["unmeasured_requests"] = unmeasured
    return counts


def thinking(text="", signature=None, part_id=None) -> dict:
    return {"content": text, "id": part_id, "signature": signature, "provider_name": "anthropic",
            "provider_details": None, "part_kind": "thinking"}


def text(content: str) -> dict:
    return {"content": content, "id": None, "provider_name": None, "provider_details": None, "citations": None,
            "part_kind": "text"}


def effort(level: str | None, top_level=True) -> dict:
    """The settings change Pydantic AI records in the request whose thinking effort differs from the one in effect."""
    return {"settings": {"thinking": level}, "top_level": top_level, "part_kind": "model-settings-delta"}


def call(call_id: str, name="python", args=None) -> dict:
    return {"tool_name": name, "args": {"code": "page_info()"} if args is None else args, "tool_call_id": call_id,
            "tool_kind": None, "id": None, "provider_name": None, "provider_details": None, "part_kind": "tool-call"}


class Feed:
    """One turn as the server feeds it: its aux records (the turn, then agent.py's attempts, summaries and pruned
    screenshots), then the run's whole conversation as natives, earlier turns' messages first. Times advance a
    millisecond a record, as the run made them."""

    clock = datetime(2026, 11, 27, 10, 0, tzinfo=timezone.utc)

    def __init__(self, run="run-1", earlier: list[dict] | None = None, prompt=PROMPT):
        self.run = run
        self.messages: list[dict] = list(earlier or [])
        self.attempts: list[dict] = []
        self.summaries: list[dict] = []
        self.pruned: dict[str, list[dict]] = {}
        self.ending = {"status": "completed", "text": "Example Domain", "cost_usd": None}
        self.request({"content": prompt, "timestamp": self.tick(), "part_kind": "user-prompt"})

    @classmethod
    def tick(cls) -> str:
        cls.clock += timedelta(milliseconds=1)
        return cls.clock.isoformat().replace("+00:00", "Z")

    def request(self, *parts: dict, run: str | None = None) -> Feed:
        self.messages.append({"parts": list(parts), "timestamp": self.tick(), "instructions": "You are browser-use.",
                              "kind": "request", "run_id": run or self.run, "conversation_id": CONVERSATION,
                              "metadata": None, "state": "complete"})
        return self

    def attempt(self, status: int | None = 200) -> Feed:
        record = {"started": self.tick()}
        if status is not None:
            record.update(status=status, timestamp=self.tick())
        self.attempts.append(record)
        return self

    def dumped_response(self, *parts: dict, counts: dict | None = None, response_id: str | None = None,
                        model=MODEL) -> dict:
        return {"parts": list(parts), "usage": counts or usage(), "model_name": model, "model_id": "claude-haiku-4-5",
                "timestamp": self.tick(), "kind": "response", "provider_name": "anthropic",
                "provider_url": "http://litellm.test", "canonical_system": None, "provider_details": None,
                "provider_response_id": response_id, "finish_reason": "tool_call", "run_id": self.run,
                "conversation_id": CONVERSATION, "metadata": None, "workspace_ref": None, "history_digest": None,
                "failed_attempts": None, "state": "complete"}

    def response(self, *parts: dict, counts: dict | None = None, response_id: str | None = None,
                 attempt=True) -> Feed:
        """A model call: its HTTP request answered, then its response."""
        if attempt:
            self.attempt(200)
        self.messages.append(self.dumped_response(*parts, counts=counts, response_id=response_id
                                                  or f"msg_{len(self.messages):02d}"))
        return self

    def returned(self, call_id: str, content, name="python", outcome="success") -> Feed:
        part = {"tool_name": name, "content": content, "tool_call_id": call_id, "tool_kind": None, "metadata": None,
                "timestamp": self.tick(), "outcome": outcome, "retry": None, "part_kind": "tool-return"}
        if isinstance(content, list):
            part["content_file_digests"] = []
        return self.request(part)

    def summary(self, summary_text: str, counts: dict, compacted=True) -> Feed:
        """A compaction's summarizer call, which the conversation never holds."""
        self.attempt(200)
        self.summaries.append({"compacted": compacted, "response": self.dumped_response(
            text(summary_text), counts=counts, response_id=f"msg_summary_{len(self.summaries)}")})
        return self

    def prune(self, call_id: str, index: int, data: bytes, media_type="image/jpeg") -> Feed:
        """agent.py's copy of a screenshot browser-use pruned from the conversation."""
        self.pruned.setdefault(call_id, []).append(
            {"index": index, "media_type": media_type, "data": base64.b64encode(data).decode()})
        return self

    def ends(self, **ending) -> Feed:
        self.ending.update(ending)
        return self

    def records(self) -> list[Fed]:
        ours = [m for m in self.messages if m["run_id"] == self.run]
        turn = {"run_id": ours[-1]["run_id"] if ours else None, "model": "claude-haiku-4-5", "effort": "low",
                **self.ending}
        return [Fed(turn, aux="turn"), *(Fed(a, aux="attempt") for a in self.attempts),
                *(Fed(s, aux="summary") for s in self.summaries),
                *(Fed({"tool_call_id": c, "images": i}, aux="images") for c, i in self.pruned.items()),
                *(Fed(message) for message in self.messages)]

    def task(self, index=0) -> RecordedTask:
        failed = self.ending["status"] != "completed"
        answer = self.ending["text"] or self.ending.get("error") or self.ending["status"]
        return RecordedTask(context=context(index), records=self.records(), state="failed" if failed else "completed",
                            final=[answer])


def context(index=0):
    return TurnContext(index=index, context_id="ctx", task_id=f"t{index}", message_id=f"m{index}", inbound=[PROMPT],
                       agent_name="browser-use", agent_version="2.0.0", agent_model="claude-haiku-4-5",
                       started_at=f"2026-11-27T09:00:0{index}Z")


def run(*feeds: Feed):
    kit = check_adapter(BrowserUseAtif, [feed.task(i) for i, feed in enumerate(feeds)])
    kit.raise_for_errors()
    assert not errors_in(validate(kit.document))
    return kit


def agent_steps(document):
    return [s for s in document["steps"] if s["source"] == "agent"]


def finished(feed: Feed, call_id="toolu_09", answer="Example Domain", counts: dict | None = None) -> Feed:
    return feed.response(call(call_id, "finish", {"result": answer, "success": True}), counts=counts).returned(
        call_id, "Final result processed.", name="finish")


def test_each_response_is_a_step_with_its_model_time_counts_and_cost_summing_to_the_turns():
    feed = Feed().response(thinking("Open the page.", "EqQBsig"), text("Opening it."),
                           call("toolu_01", args={"code": "goto('https://example.com')"}),
                           counts=usage(1200, 40, 1000, 100, "0.0012"))
    feed.returned("toolu_01", ["Example Domain", binary(JPEG)])
    finished(feed.response(call("toolu_02"), counts=usage(1500, 30, 1200, 0, "0.0010")).returned("toolu_02", "ok"),
             counts=usage(1600, 20, 1500, 50, "0.0007"))
    feed.ends(cost_usd=0.0029)
    document = run(feed).document
    steps = agent_steps(document)

    assert len(steps) == 3
    assert [s["model_name"] for s in steps] == [MODEL] * 3 and all(s["timestamp"] for s in steps)
    assert all(s["reasoning_effort"] == "low" for s in steps)
    assert steps[0]["metrics"] == {"prompt_tokens": 1200, "cached_tokens": 1000, "completion_tokens": 40,
                                   "cost_usd": 0.0012,
                                   "extra": {"agentenv": {"cache_write_tokens": 100, "cost_source": "harness"}}}
    totals = document["final_metrics"]
    assert (totals["total_prompt_tokens"], totals["total_completion_tokens"], totals["total_cached_tokens"]) == (
        1200 + 1500 + 1600, 40 + 30 + 20, 1000 + 1200 + 1500)
    assert profile_of(totals) == {"cost_source": "harness", "total_cache_write_tokens": 150}
    assert totals["total_cost_usd"] == pytest.approx(0.0029)
    assert sum(s["metrics"]["cost_usd"] for s in steps) == pytest.approx(0.0029)


def test_a_call_gets_its_result_in_its_step_and_a_screenshot_is_a_media_file():
    feed = Feed().response(call("toolu_01", args={"code": "screenshot()"})).returned(
        "toolu_01", ["Example Domain", binary(JPEG), binary(PNG, "image/png")])
    kit = run(finished(feed))
    (result,) = agent_steps(kit.document)[0]["observation"]["results"]

    assert result["source_call_id"] == "toolu_01"
    assert result["content"] == [{"type": "text", "text": "Example Domain"},
                                 {"type": "image", "source": {"media_type": "image/jpeg", "path": media_path(JPEG)}},
                                 {"type": "image", "source": {"media_type": "image/png",
                                                              "path": media_path(PNG, ".png")}}]
    assert kit.media[media_path(JPEG)] == JPEG and kit.media[media_path(PNG, ".png")] == PNG
    assert profile_of(result)["media"][media_path(PNG, ".png")]["width"] == 4


def test_a_screenshot_browser_use_pruned_comes_back_from_what_agent_py_kept():
    """One agent.py didn't keep, as one an earlier run pruned, stays the note browser-use left in its place."""
    older = JPEG + b"older"
    feed = Feed().response(call("toolu_01")).returned("toolu_01", ["page one", "[older screenshot removed]"])
    feed.response(call("toolu_02")).returned("toolu_02", ["page two", "[older screenshot removed]"])
    kit = run(finished(feed.prune("toolu_01", 1, older)))
    results = [s["observation"]["results"][0]["content"] for s in agent_steps(kit.document)[:2]]

    assert results[0][1] == {"type": "image", "source": {"media_type": "image/jpeg", "path": media_path(older)}}
    assert kit.media[media_path(older)] == older
    assert results[1] == "page two\n\n[older screenshot removed]"


class AuxLog:
    def __init__(self) -> None:
        self.fed: list[tuple[str, dict]] = []

    def aux(self, name: str, record: dict) -> None:
        self.fed.append((name, record))


def test_the_server_feeds_the_screenshots_only_of_the_runs_own_calls():
    """A run's first requests prune the earlier turn's last screenshots too, which that turn's copy holds already: in
    the recorded browse scenario, two of the second turn's four entries and two thirds of its aux log."""
    earlier = Feed(run="run-1").response(call("toolu_01")).returned("toolu_01", ["page one", binary(JPEG)])
    feed = Feed(run="run-2", earlier=earlier.messages).response(call("toolu_02")).returned(
        "toolu_02", ["page two", "[older screenshot removed]"])
    feed.prune("toolu_01", 1, JPEG).prune("toolu_02", 1, PNG, "image/png")
    log = AuxLog()
    a2a_server._feed_atif(log, {"messages": feed.messages, "status": "completed", "text": "Example Domain",
                                "model": "claude-haiku-4-5", "cost_usd": None,
                                "atif": {"attempts": feed.attempts, "summaries": [], "images": feed.pruned}}, None)

    assert [record["tool_call_id"] for name, record in log.fed if name == "images"] == ["toolu_02"]


def test_tool_outcomes_map_to_result_statuses_and_mcp_tools_keep_their_names():
    feed = Feed().response(call("toolu_01", "python", {"code": "boom()"}), call("toolu_02", "browser__navigate",
                                                                               {"url": "https://example.com"}))
    feed.request({"tool_name": "python", "content": "NameError: boom", "tool_call_id": "toolu_01",
                  "tool_kind": None, "metadata": None, "timestamp": Feed.tick(), "outcome": "retried",
                  "retry": {"cause": "model_retry", "attempt": 1, "max_retries": 2}, "part_kind": "tool-return"},
                 {"tool_name": "browser__navigate", "content": "Navigated.", "tool_call_id": "toolu_02",
                  "tool_kind": None, "metadata": None, "timestamp": Feed.tick(), "outcome": "interrupted",
                  "retry": None, "part_kind": "tool-return"})
    step = agent_steps(run(finished(feed)).document)[0]

    assert [c["function_name"] for c in step["tool_calls"]] == ["python", "browser__navigate"]
    assert [profile_of(r).get("status") for r in step["observation"]["results"]] == ["error", "canceled"]


def test_an_openai_routes_string_arguments_are_read_as_json():
    feed = Feed().response(call("call_1", args=json.dumps({"code": "1 + 1"})), call("call_2", args="not json"),
                           call("call_3", args=""))
    feed.request(*({"tool_name": "python", "content": "2", "tool_call_id": f"call_{n}", "tool_kind": None,
                    "metadata": None, "timestamp": Feed.tick(), "outcome": "success", "retry": None,
                    "part_kind": "tool-return"} for n in (1, 2, 3)))
    calls = agent_steps(run(finished(feed)).document)[0]["tool_calls"]

    assert calls[0]["arguments"] == {"code": "1 + 1"} and "extra" not in calls[0]
    assert calls[1]["arguments"] == {"input": "not json"} and calls[1]["extra"] == {"raw_arguments": "not json"}
    assert calls[2]["arguments"] == {}


@pytest.mark.parametrize(("part", "expected", "reasoning_content"), [
    (thinking("Plan it.", "EqQBsig"), {"chars": 8, "visibility": "summary", "signature": "EqQBsig"}, "Plan it."),
    (thinking("", "c2lnbmVk", "redacted_thinking"), {"chars": 0, "visibility": "redacted", "data": "c2lnbmVk"}, None),
    (thinking("", "EqQBsig"), {"chars": 0, "visibility": "omitted", "signature": "EqQBsig"}, None),
])
def test_thinking_is_summarized_redacted_or_withheld_reasoning(part, expected, reasoning_content):
    step = agent_steps(run(finished(Feed().response(part, text("Done."), call("toolu_01"))
                                    .returned("toolu_01", "ok"))).document)[0]

    assert profile_of(step)["reasoning"] == [expected]
    assert step.get("reasoning_content") == reasoning_content


def test_each_step_is_at_the_effort_the_conversations_settings_name_though_the_turn_configured_none():
    """browser-use asks Claude for medium unless told otherwise and records it once, in the conversation's first
    request, which a later turn's run returns again unstamped; a change from there on holds for what follows."""
    first = Feed(run="run-1").ends(effort=None)
    first.messages[0]["parts"].append(effort("medium"))
    finished(first.response(call("toolu_01")).returned("toolu_01", "ok"), call_id="toolu_02")
    second = Feed(run="run-2", earlier=first.messages, prompt="And its first link?").ends(effort=None)
    finished(second.response(call("toolu_03")).request(effort("high", top_level=False)).returned("toolu_03", "ok"),
             call_id="toolu_04")
    configured = Feed().response(text("Example Domain"))

    assert [s.get("reasoning_effort") for s in agent_steps(run(first, second).document)] == [
        "medium", "medium", "medium", "high"]
    assert agent_steps(run(configured).document)[0]["reasoning_effort"] == "low"


def test_unsigned_thinking_from_an_openai_compatible_route_is_raw():
    step = agent_steps(run(Feed().response(thinking("Hmm."), text("Example Domain"))).document)[0]

    assert step["reasoning_content"] == "Hmm." and "reasoning" not in profile_of(step)


def test_a_finish_calls_answer_is_its_steps_message_so_the_final_response_is_written_once():
    kit = run(finished(Feed()))
    (step,) = agent_steps(kit.document)

    assert step["message"] == "Example Domain" and step["tool_calls"][0]["function_name"] == "finish"
    assert profile_of(step)["state"] == "completed"


def test_an_answer_from_the_repl_is_the_finish_from_repl_steps_message():
    feed = Feed().response(call("toolu_01", "finish_from_repl", {"expression": "heading", "success": True}))
    feed.returned("toolu_01", "Final result processed.", name="finish_from_repl").ends(text="Example Domain")
    (step,) = agent_steps(run(feed).document)

    assert step["message"] == "Example Domain"


def test_a_text_reply_is_the_answer_and_a_step_with_its_own_text_keeps_it():
    reply = agent_steps(run(Feed().response(text("Example Domain"))).document)
    spoke = agent_steps(run(finished(Feed().response(text("Found it."), call("toolu_01")).returned(
        "toolu_01", "ok"), call_id="toolu_02")).document)

    assert [s["message"] for s in reply] == ["Example Domain"]
    assert [s["message"] for s in spoke] == ["Found it.", "Example Domain"]


def test_a_refused_call_the_sdk_retried_is_a_step_with_its_error_before_the_response():
    feed = Feed().attempt(503).attempt(529).response(text("Example Domain"))
    steps = agent_steps(run(feed).document)

    assert [s["llm_call_count"] for s in steps] == [0, 0, 1]
    assert [profile_of(s).get("error", {}).get("message") for s in steps[:2]] == [
        "the endpoint refused the call (HTTP 503)", "the endpoint refused the call (HTTP 529)"]
    assert "metrics" not in steps[0] and steps[2]["message"] == "Example Domain"


def test_a_run_the_endpoint_ended_is_its_refused_call_then_the_turns_error():
    error = "ModelHTTPError: status_code: 400, model_name: claude-haiku-4-5, body: {'error': 'bad request'}"
    feed = Feed().attempt(400).ends(status="error", text="", error=error)
    kit = run(feed)
    steps = agent_steps(kit.document)

    assert profile_of(steps[0])["error"] == {"kind": "infra_error", "message": "the endpoint refused the call (HTTP 400)"}
    assert steps[-1]["message"] == error and profile_of(steps[-1])["state"] == "failed"


def test_a_call_cut_off_by_the_runs_budget_has_no_counts_so_the_totals_are_unknown():
    feed = Feed().response(call("toolu_01", args={"code": "wait_long()"})).attempt(200).attempt(None)
    feed.ends(status="timeout", text="")
    document = run(feed).document
    steps = agent_steps(document)

    assert [s["llm_call_count"] for s in steps[:3]] == [1, 1, 1]
    assert [profile_of(s)["error"]["message"] for s in steps[1:3]] == [
        "the call's response (HTTP 200) never reached the agent", "the call got no answer"]
    assert "metrics" not in steps[1] and "total_prompt_tokens" not in document["final_metrics"]
    assert profile_of(steps[0]["observation"]["results"][0])["status"] == "timeout"


def test_a_call_its_deadline_cut_off_is_a_step_before_the_retry_feedback_it_led_to():
    """browser-use asks again when a model request gets no answer in time, as Pydantic AI's retry: a new request holding
    the feedback, stamped after the call that got none."""
    feed = Feed().attempt(None)
    feed.request({"content": "The model request got no answer within 300.0 s.", "retry": {
        "cause": "model_retry", "attempt": 1, "max_retries": 5}, "timestamp": Feed.tick(), "part_kind": "retry-feedback"})
    steps = run(feed.response(text("Example Domain"))).document["steps"]

    assert [s["source"] for s in steps] == ["user", "agent", "system", "agent"]
    assert profile_of(steps[1])["error"]["message"] == "the call got no answer"
    assert steps[1]["timestamp"] < steps[2]["timestamp"] < steps[3]["timestamp"]


def test_a_compactions_summary_is_a_step_where_it_was_written_and_its_usage_counts():
    feed = Feed().response(call("toolu_01"), counts=usage(1000, 10, 0, 0, None)).returned("toolu_01", "ok")
    feed.summary("The user wants example.com's heading.", usage(800, 60, 0, 0, None))
    feed.response(text("Example Domain"), counts=usage(300, 5, 0, 0, None))
    document = run(feed).document
    steps = agent_steps(document)

    assert [s["message"] for s in steps] == ["", "The user wants example.com's heading.", "Example Domain"]
    assert profile_of(steps[1])["compaction"] == {"written_by": "assistant", "trigger": "auto"}
    assert steps[1]["metrics"]["prompt_tokens"] == 800 and "reasoning_effort" not in steps[1]
    assert document["final_metrics"]["total_prompt_tokens"] == 2100


def test_a_compaction_browser_use_gave_up_on_is_its_calls_usage_with_why():
    feed = Feed().summary("too long", usage(800, 60, 0, 0, None), compacted=False).response(text("Example Domain"))
    step = agent_steps(run(feed).document)[0]

    assert step["message"] == "" and step["metrics"]["completion_tokens"] == 60
    assert "compaction" not in profile_of(step) and profile_of(step)["error"]["kind"] == "agent_error"


def test_the_harnesss_own_messages_to_the_model_are_system_steps():
    deliver = ("Budget nearly exhausted. Deliver the verified result now with finish/finish_from_repl. Reference "
               "saved files; explicitly list missing evidence. Do not perform more browser actions.")
    feed = Feed().response(text("<tool_call>finish</tool_call>"))
    feed.request({"content": "That reply wrote a tool call as text; it was not run.", "retry": {
        "cause": "model_retry", "attempt": 1, "max_retries": 1}, "timestamp": Feed.tick(), "part_kind": "retry-feedback"},
        {"content": deliver, "timestamp": Feed.tick(), "part_kind": "user-prompt"})
    document = run(finished(feed)).document

    assert [s["source"] for s in document["steps"]] == ["user", "agent", "system", "system", "agent"]
    assert document["steps"][2]["message"] == "That reply wrote a tool call as text; it was not run."
    assert document["steps"][3]["message"] == deliver


def test_a_later_turn_holds_only_its_own_runs_messages_though_the_native_repeats_the_conversation():
    first = finished(Feed(run="run-1").response(call("toolu_01")).returned("toolu_01", "ok"), call_id="toolu_02")
    second = finished(Feed(run="run-2", earlier=first.messages, prompt="And its first link?").response(
        call("toolu_03")).returned("toolu_03", "ok"), call_id="toolu_04", answer="More information...")
    second.ends(text="More information...")
    document = run(first, second).document

    assert [s["source"] for s in document["steps"]] == ["user", "agent", "agent", "user", "agent", "agent"]
    assert [c["tool_call_id"] for s in agent_steps(document) for c in s["tool_calls"]] == [
        "toolu_01", "toolu_02", "toolu_03", "toolu_04"]
    assert profile_of(document["steps"][3])["turn"] == 1
    assert document["steps"][-1]["message"] == "More information..."


def test_a_response_that_gave_no_usage_has_no_counts():
    step = agent_steps(run(Feed().response(text("Example Domain"), counts=usage(0, 0, 0, 0, None, unmeasured=1)))
                       .document)[0]

    assert "metrics" not in step and step["llm_call_count"] == 1


def test_responses_whose_provider_ids_repeat_are_steps_of_their_own():
    """An OpenAI-compatible server may give two responses one id: Ollama's draws them from a thousand."""
    feed = Feed().response(text("Opening it."), call("call_1"), response_id="chatcmpl-7").returned("call_1", "ok")
    steps = agent_steps(run(feed.response(text("Example Domain"), response_id="chatcmpl-7")).document)

    assert [s["message"] for s in steps] == ["Opening it.", "Example Domain"]
    assert [s["metrics"]["prompt_tokens"] for s in steps] == [1200, 1200]


def test_from_atif_gives_pydantic_ai_messages_back():
    document = run(finished(Feed().response(thinking("Plan.", "sig"), text("Looking."), call("toolu_01"))
                            .returned("toolu_01", ["ok", binary(JPEG)]))).document
    log = BrowserUseAtif.from_atif(document).log

    assert [m["kind"] for m in log] == ["request", "response", "request", "response", "request"]
    assert log[1]["parts"][0] == {"part_kind": "thinking", "content": "Plan."}
    assert log[1]["usage"] == {"input_tokens": 1200, "cache_read_tokens": 1000, "output_tokens": 40}
    assert log[2]["parts"] == [{"part_kind": "tool-return", "tool_name": "python", "tool_call_id": "toolu_01",
                                "content": "ok"}]
    assert BrowserUseAtif.from_atif(document).session is None


def test_the_tested_browser_use_is_the_one_the_image_pins():
    pinned = re.search(r"^browser-use==(\S+)$", (AGENT_DIR / "browser-use-requirements.txt").read_text(), re.M)[1]
    assert browser_use_atif.TESTED_WITH == (pinned,)
