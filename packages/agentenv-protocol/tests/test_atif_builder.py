"""The adapter framework and the step builder: declarations checked at import, records routed to the most specific
handler, each task's user step and final response written from its TurnContext, results in their call's step, usage
in ATIF's terms, and a live view that only ever grows."""
import math

import pytest

from agentenv_protocol.atif import (
    Adapter,
    AdapterRun,
    AtifError,
    Fed,
    Native,
    RecordedTask,
    Step,
    TurnContext,
    atif_adapter,
    check,
    finish,
    is_prefix,
    native,
    profile_of,
    replay,
    served_turn,
    timestamp,
)


def adapter(includes_cache=False):
    @atif_adapter(native_format="toy-events/1", version="0.1.0", input_includes_cache=includes_cache)
    class Toy(Adapter):
        @native(type="say")
        def say(self, r, turn):
            turn.message(step=r["step"], text=r["text"])

        @native(type="call")
        def call(self, r, turn):
            turn.tool_call(step=r["step"], tool_call_id=r["id"], function_name="Bash", arguments=r.get("args", {}))

        @native(type="result")
        def result(self, r, turn):
            turn.tool_result(source_call_id=r["id"], text=r.get("out", "done"), time=r.get("at"))

        @native(type="usage")
        def usage(self, r, turn):
            turn.step(key=r["step"], input_tokens=r.get("in"), cache_read_tokens=r.get("read"),
                      cache_write_tokens=r.get("write"), output_tokens=r.get("out"))
    return Toy


CTX = TurnContext(index=0, context_id="ctx", task_id="t1", message_id="m1", inbound=["hello"], agent_name="toy",
                  agent_version="1.0", started_at="2026-11-18T10:00:00Z")


def run_task(records, *, final=None, state="completed", cls=None):
    run = AdapterRun(cls or adapter())
    run.start_task(CTX)
    pages = [run.page()]
    for record in records:
        run.native(record)
        pages.append(run.page())
    run.end_task(final=final, state=state, ended_at="2026-11-18T10:05:00Z")
    pages.append(run.page())
    return run, pages


def test_the_user_step_comes_from_the_turn_context():
    run, _ = run_task([{"type": "say", "step": "s1", "text": "hi"}])
    user = run.document()["steps"][0]
    assert user["source"] == "user" and user["message"] == "hello" and user["timestamp"] == "2026-11-18T10:00:00Z"
    assert profile_of(user) == {"context_id": "ctx", "task_id": "t1", "message_id": "m1", "turn": 0}
    assert run.document()["agent"] == {"name": "toy", "version": "1.0"}


def test_a_final_response_the_last_step_holds_is_written_once():
    run, _ = run_task([{"type": "say", "step": "s1", "text": "all done"}], final=["all done"])
    steps = run.document()["steps"]
    assert len(steps) == 2 and profile_of(steps[-1])["state"] == "completed"


def test_a_final_response_the_last_step_lacks_gets_its_own_step_with_no_model_call():
    run, _ = run_task([{"type": "say", "step": "s1", "text": "working"}], final=["{\"ok\": true}"], state="failed")
    last = run.document()["steps"][-1]
    assert last["message"] == "{\"ok\": true}" and last["llm_call_count"] == 0 and "metrics" not in last
    assert profile_of(last)["state"] == "failed" and last["timestamp"] == "2026-11-18T10:05:00Z"
    check(run.document())


def test_the_terminal_state_is_never_served_before_the_task_ends():
    run, pages = run_task([{"type": "say", "step": "s1", "text": "a"}, {"type": "say", "step": "s2", "text": "b"}])
    for page in pages[:-1]:
        assert all("state" not in profile_of(s) for s in (page or {}).get("steps", []))
    assert profile_of(pages[-1]["steps"][-1])["state"] == "completed"
    with pytest.raises(AtifError):
        AdapterRun(adapter()).end_task()


def test_a_late_result_lands_in_its_calls_step_and_holds_back_the_steps_after_it():
    records = [{"type": "call", "step": "A", "id": "c1"}, {"type": "call", "step": "B", "id": "c2"},
               {"type": "result", "id": "c2"}, {"type": "say", "step": "C", "text": "waiting"},
               {"type": "result", "id": "c1", "out": "312 passed", "at": "2026-11-18T10:04:00.5Z"},
               {"type": "say", "step": "D", "text": "done"}]
    run, pages = run_task(records)
    served = [len(p["steps"]) if p else 0 for p in pages]
    assert served == [0, 1, 1, 1, 1, 3, 4, 5]
    step_a = run.document()["steps"][1]
    assert [r["source_call_id"] for r in step_a["observation"]["results"]] == ["c1"]
    assert profile_of(step_a["observation"]["results"][0]) == {"arrived": {"at": "2026-11-18T10:04:00.500Z",
                                                                           "step_id": 4}}
    assert all(is_prefix(p, run.document()) for p in pages)


def test_a_step_still_waiting_on_a_result_is_served_when_its_task_ends():
    run, pages = run_task([{"type": "call", "step": "A", "id": "c1"}, {"type": "say", "step": "B", "text": "x"}])
    assert [len(p["steps"]) if p else 0 for p in pages] == [0, 1, 1, 3]
    assert "observation" not in pages[-1]["steps"][1]
    check(pages[-1])


@pytest.mark.parametrize("includes,given,expected", [(False, 10, 95), (True, 100, 100)])
def test_usage_is_written_in_atifs_terms(includes, given, expected):
    run, _ = run_task([{"type": "usage", "step": "s1", "in": given, "read": 80, "write": 5, "out": 7}],
                      cls=adapter(includes))
    metrics = run.document()["steps"][1]["metrics"]
    assert metrics["prompt_tokens"] == expected and metrics["cached_tokens"] == 80
    assert metrics["completion_tokens"] == 7 and metrics["extra"] == {"agentenv": {"cache_write_tokens": 5}}
    with pytest.raises(AtifError):
        run_task([{"type": "usage", "step": "s1", "in": 10}], cls=adapter(None))


@pytest.mark.parametrize("value,expected", [
    ("2026-11-18T01:02:03.12345Z", "2026-11-18T01:02:03.123450Z"),
    ("2026-11-18T01:02:03.123456789Z", "2026-11-18T01:02:03.123456Z"),
    ("2026-11-18T01:02:03.1+05:30", "2026-11-17T19:32:03.100Z"),
    ("2026-11-18T01:02:03.123Z", "2026-11-18T01:02:03.123Z"),
    ("2026-11-18 01:02:03-0700", "2026-11-18T08:02:03Z"),
])
def test_timestamps_are_normalized_to_what_atif_reads_everywhere(value, expected):
    assert timestamp(value) == expected
    Step.model_validate({"step_id": 1, "source": "user", "message": "", "timestamp": timestamp(value)})


@pytest.mark.parametrize("value", ["2026-11-18T01:02:03", "yesterday", "2026-11-18T25:00:00Z", 1700000000])
def test_a_time_without_a_zone_or_that_isnt_one_is_refused(value):
    with pytest.raises(AtifError):
        timestamp(value)


@pytest.mark.parametrize("arguments,typed", [
    ("*** Begin Patch", {"input": "*** Begin Patch"}), (None, {}), ([1, 2], {"value": [1, 2]}), (7, {"value": 7}),
])
def test_arguments_that_arent_an_object_are_wrapped_as_harbor_wraps_them(arguments, typed):
    run, _ = run_task([{"type": "call", "step": "s1", "id": "c1", "args": arguments}])
    call = run.document()["steps"][1]["tool_calls"][0]
    assert call["arguments"] == typed and call["extra"] == {"raw_arguments": arguments}
    run, _ = run_task([{"type": "call", "step": "s1", "id": "c1", "args": {"command": "ls"}}])
    assert "extra" not in run.document()["steps"][1]["tool_calls"][0]


def test_a_duplicate_call_id_and_a_result_for_no_call_are_refused():
    with pytest.raises(AtifError, match="already"):
        run_task([{"type": "call", "step": "s1", "id": "c1"}, {"type": "call", "step": "s2", "id": "c1"}])
    with pytest.raises(AtifError, match="no call"):
        run_task([{"type": "result", "id": "nope"}])


def test_values_that_arent_json_are_refused_as_they_are_written():
    with pytest.raises(AtifError, match="JSON"):
        run_task([{"type": "call", "step": "s1", "id": "c1", "args": {"x": math.nan}}])
    with pytest.raises(AtifError, match="JSON"):
        run_task([{"type": "call", "step": "s1", "id": "c1", "args": {"x": object()}}])


def test_reads_are_fresh_copies():
    run, _ = run_task([{"type": "say", "step": "s1", "text": "a"}])
    run.document()["steps"].clear()
    run.page()["steps"][0]["message"] = "rewritten"
    assert len(run.document()["steps"]) == 2 and run.page()["steps"][0]["message"] == "hello"


def test_reasoning_blocks_keep_their_boundaries_signatures_and_ciphertext():
    @atif_adapter(native_format="toy-think/1", version="1", input_includes_cache=None)
    class Thinker(Adapter):
        @native(type="think")
        def think(self, r, turn):
            turn.reasoning(step="s1", text=r.get("text"), visibility=r["vis"], signature=r.get("sig"),
                           data=r.get("data"))

    run = AdapterRun(Thinker)
    run.start_task(CTX)
    for r in [{"vis": "raw", "text": "first", "sig": "S1"}, {"vis": "omitted", "sig": "S2"},
              {"vis": "redacted", "data": "Q0lQSEVS"}, {"vis": "summary", "text": "second"}]:
        run.native({"type": "think", **r})
    run.end_task()
    step = run.document()["steps"][1]
    blocks = profile_of(step)["reasoning"]
    assert step["reasoning_content"] == "first\n\nsecond"
    assert [b["chars"] for b in blocks] == [5, 0, 0, 6] and blocks[2]["data"] == "Q0lQSEVS"
    text, pieces = step["reasoning_content"], []
    for block in blocks:
        pieces.append(text[:block["chars"]])
        text = text[block["chars"] + 2:] if block["chars"] else text
    assert pieces == ["first", "", "", "second"]


def test_media_go_to_the_sink_once_and_types_atif_lacks_become_notes():
    @atif_adapter(native_format="toy-media/1", version="1", input_includes_cache=None)
    class Shots(Adapter):
        @native(type="shot")
        def shot(self, r, turn):
            turn.tool_call(step=r["step"], tool_call_id=r["id"], function_name="computer", arguments={})
            turn.tool_result(source_call_id=r["id"], content=[turn.media(r["data"], r["mt"], width=2, height=1)])

    sunk = []
    run = AdapterRun(Shots, media=lambda path, data: sunk.append(path))
    run.start_task(CTX)
    for i, mt in enumerate(["image/png", "image/png", "video/mp4"]):
        run.native({"type": "shot", "step": f"s{i}", "id": f"c{i}", "data": b"same" if i < 2 else b"clip", "mt": mt})
    run.end_task()
    document = run.document()
    check(document)
    png, mp4 = (r["observation"]["results"][0] for r in document["steps"][1::2])
    path = png["content"][0]["source"]["path"]
    assert path.startswith("media/") and path.endswith(".png") and len(sunk) == 2 and run.media_files == {}
    assert profile_of(png)["media"][path] == {"sha256": path[6:-4], "bytes": 4, "width": 2, "height": 1}
    assert mp4["content"].startswith("[video: media/") and mp4["content"].endswith(", 4 B]")
    assert list(profile_of(mp4)["media"].values())[0]["media_type"] == "video/mp4"


def test_records_no_method_takes_are_counted_by_type():
    run, _ = run_task([{"type": "system"}, {"type": "system"}, {"kind": "no type"}])
    assert profile_of(run.document())["unmatched"] == {"dict": 1, "system": 2}


def test_a_sub_agent_is_an_embedded_trajectory_its_spawning_result_points_at():
    @atif_adapter(native_format="toy-sub/1", version="1", input_includes_cache=True)
    class Parent(Adapter):
        @native(type="go")
        def go(self, r, turn):
            turn.tool_call(step="p1", tool_call_id="task", function_name="Task", arguments={})
            turn.subagent("sub", spawned_by="task", name="explore", model_name="small")
            turn.message(step="x1", text="looking", agent="sub")
            turn.step(key="x1", input_tokens=5, output_tokens=1, agent="sub")
            turn.step(key="p1", input_tokens=10, output_tokens=2)
            turn.tool_result(source_call_id="task", text="found it")

    run = AdapterRun(Parent)
    run.start_task(CTX)
    run.native({"type": "go"})
    run.end_task()
    document = run.document()
    check(document)
    sub = document["subagent_trajectories"][0]
    assert sub["trajectory_id"] == "sub" and sub["agent"] == {"name": "explore", "version": "unknown",
                                                             "model_name": "small"}
    result = document["steps"][1]["observation"]["results"][0]
    assert result["subagent_trajectory_ref"] == [{"trajectory_id": "sub"}]
    assert document["final_metrics"]["total_prompt_tokens"] == 15 and document["final_metrics"]["total_steps"] == 2


def test_a_step_with_no_model_call_takes_no_usage():
    @atif_adapter(native_format="toy-dispatch/1", version="1", input_includes_cache=True)
    class Dispatch(Adapter):
        @native(type="d")
        def d(self, r, turn):
            turn.step(key="s1", llm_call_count=0, input_tokens=3)

    with pytest.raises(AtifError):
        run_task([{"type": "d"}], cls=Dispatch)


def test_finish_raising_still_closes_the_task():
    @atif_adapter(native_format="toy-fail/1", version="1", input_includes_cache=None)
    class Fails(Adapter):
        @finish
        def close(self, turn):
            raise RuntimeError("boom")

    run = AdapterRun(Fails)
    run.start_task(CTX)
    with pytest.raises(RuntimeError):
        run.end_task(state="failed")
    assert profile_of(run.document()["steps"][-1])["state"] == "failed"
    run.start_task(TurnContext(index=1, context_id="ctx", task_id="t2", inbound=["again"]))


def test_replaying_a_contexts_stored_tasks_builds_what_it_built_live():
    records = [{"type": "call", "step": "s1", "id": "c1"}, {"type": "usage", "step": "s1", "in": 1, "out": 1},
               {"type": "result", "id": "c1"}, {"type": "say", "step": "s2", "text": "ok"}]
    live, _ = run_task(records, final=["ok"])
    stored = [RecordedTask(context=CTX, records=[Fed(r) for r in records], final=["ok"],
                           ended_at="2026-11-18T10:05:00Z")]
    assert replay(adapter(), stored) == live.document()


def test_declarations_are_checked_at_import():
    with pytest.raises(AtifError):
        atif_adapter(native_format="atif/1", version="1", input_includes_cache=True)
    with pytest.raises(AtifError):
        atif_adapter(native_format="no-version", version="1", input_includes_cache=True)
    with pytest.raises(AtifError):
        atif_adapter(native_format="toy/1", version="", input_includes_cache=True)
    with pytest.raises(AtifError):
        atif_adapter(native_format="toy/1", version="1", input_includes_cache="yes")
    with pytest.raises(AtifError):
        @atif_adapter(native_format="toy/1", version="1", input_includes_cache=True)
        class TwoFinishes(Adapter):
            @finish
            def a(self, turn):
                pass

            @finish
            def b(self, turn):
                pass
    with pytest.raises(AtifError):
        AdapterRun(Adapter)


def test_the_most_specific_handler_takes_a_record_and_a_tie_is_refused():
    @atif_adapter(native_format="toy-pick/1", version="1", input_includes_cache=None)
    class Pick(Adapter):
        @native(type="e")
        def general(self, r, turn):
            turn.message(step="g", text="general")

        @native(type="e", where={"kind": "retry"})
        def retry(self, r, turn):
            turn.message(step="r", text="retry")

    run, _ = run_task([{"type": "e", "kind": "retry"}, {"type": "e", "kind": "other"}], cls=Pick)
    assert [s["message"] for s in run.document()["steps"][1:]] == ["retry", "general"]

    @atif_adapter(native_format="toy-tie/1", version="1", input_includes_cache=None)
    class Tie(Adapter):
        @native(where={"a": 1})
        def one(self, r, turn):
            pass

        @native(where={"b": 1})
        def two(self, r, turn):
            pass

    with pytest.raises(AtifError, match="equally"):
        run_task([{"a": 1, "b": 1}], cls=Tie)


def test_from_atif_is_each_adapters_own():
    with pytest.raises(NotImplementedError):
        adapter().from_atif({})
    assert Native(log=[{"type": "x"}]).session is None


def test_a_time_out_of_range_in_utc_is_refused_and_early_years_keep_four_digits():
    with pytest.raises(AtifError, match="out of range"):
        timestamp("0001-01-01T00:00:00+05:00")
    with pytest.raises(AtifError, match="out of range"):
        timestamp("9999-12-31T23:59:59-05:00")
    assert timestamp("0099-01-01T00:00:00Z") == "0099-01-01T00:00:00Z"


def test_a_refused_task_start_leaves_no_task_open():
    run = AdapterRun(adapter())
    with pytest.raises(AtifError):
        run.start_task(TurnContext(index=0, context_id="ctx", task_id="t1", started_at="yesterday"))
    assert run.document() is None
    run.start_task(CTX)
    run.end_task()


@pytest.mark.parametrize("record", [
    {"type": "call", "step": "s1", "id": "c1", "args": {}, "name": 3},
    {"type": "call", "step": ["s1"], "id": "c1"},
    {"type": "say", "step": "s1", "text": 1.5},
])
def test_a_value_of_the_wrong_type_is_refused_without_touching_the_document(record):
    @atif_adapter(native_format="toy-types/1", version="1", input_includes_cache=None)
    class Raw(Adapter):
        @native(type="call")
        def call(self, r, turn):
            turn.tool_call(step=r["step"], tool_call_id=r["id"], function_name=r.get("name", "Bash"),
                           arguments=r.get("args"))

        @native(type="say")
        def say(self, r, turn):
            turn.message(step=r["step"], text=r["text"])

    run = AdapterRun(Raw)
    run.start_task(CTX)
    before = run.document()
    with pytest.raises(AtifError):
        run.native(record)
    assert run.document() == before


@pytest.mark.parametrize("call", [
    lambda turn: turn.step(key="s1", llm_call_count=[1]),
    lambda turn: turn.step(key="s1", reasoning_effort=float("nan")),
    lambda turn: turn.message(step="s1", content=1.5),
    lambda turn: turn.message(step="s1", text="x", agent={"id": "sub"}),
    lambda turn: turn.media(b"x", "image/png", pages=1.5),
    lambda turn: turn.media(b"x", "png"),
    lambda turn: turn.tool_result(source_call_id="none", text="x"),
    lambda turn: turn.error(message=""),
    lambda turn: turn.attempt(True),
])
def test_the_builder_refuses_values_atif_or_the_profile_would_not_read(call):
    @atif_adapter(native_format="toy-call/1", version="1", input_includes_cache=True)
    class Calls(Adapter):
        @native(type="go")
        def go(self, r, turn):
            turn.message(step="s1", text="open")
            r["call"](turn)

    run = AdapterRun(Calls)
    run.start_task(CTX)
    with pytest.raises(AtifError):
        run.native({"type": "go", "call": call})


def test_a_call_takes_one_result_and_none_once_its_task_has_ended():
    run = AdapterRun(adapter())
    run.start_task(CTX)
    for record in [{"type": "call", "step": "A", "id": "c1"}, {"type": "result", "id": "c1"},
                   {"type": "call", "step": "B", "id": "c2"}, {"type": "say", "step": "C", "text": "x"}]:
        run.native(record)
    served = run.page()
    with pytest.raises(AtifError, match="already has its result"):
        run.native({"type": "result", "id": "c1"})
    run.end_task()
    served = run.page()
    run.start_task(TurnContext(index=1, context_id="ctx", task_id="t2", inbound=["more"]))
    with pytest.raises(AtifError, match="task has ended"):
        run.native({"type": "result", "id": "c2"})
    assert run.document()["steps"][:len(served["steps"])] == served["steps"]


def test_a_step_with_no_model_call_takes_no_reasoning():
    @atif_adapter(native_format="toy-zero/1", version="1", input_includes_cache=True)
    class Zero(Adapter):
        @native(type="z")
        def z(self, r, turn):
            turn.reasoning(step="s1", text="thinking")
            turn.step(key="s1", llm_call_count=0)

    with pytest.raises(AtifError, match="no model call"):
        run_task([{"type": "z"}], cls=Zero)


@pytest.mark.parametrize("includes,expected", [(False, 160), (True, 100)])
def test_usage_given_over_several_records_keeps_every_count(includes, expected):
    run, pages = run_task([{"type": "usage", "step": "s1", "read": 50}, {"type": "usage", "step": "s1", "in": 100},
                           {"type": "usage", "step": "s1", "write": 10, "out": 7}], cls=adapter(includes))
    metrics = run.document()["steps"][1]["metrics"]
    assert metrics["prompt_tokens"] == expected and metrics["cached_tokens"] == 50 and metrics["completion_tokens"] == 7
    assert metrics["extra"] == {"agentenv": {"cache_write_tokens": 10}}
    early = AdapterRun(adapter(includes))
    early.start_task(CTX)
    early.native({"type": "usage", "step": "s1", "read": 50})
    assert "prompt_tokens" not in early.document()["steps"][1]["metrics"]


def test_cache_counts_over_an_input_count_that_includes_them_are_refused():
    run = AdapterRun(adapter(True))
    run.start_task(CTX)
    run.native({"type": "usage", "step": "s1", "in": 40, "read": 30})
    before = run.document()
    with pytest.raises(AtifError, match="exceed the input count"):
        run.native({"type": "usage", "step": "s1", "write": 20})
    assert run.document() == before


def test_a_total_is_left_out_when_a_model_step_has_no_count():
    run, _ = run_task([{"type": "say", "step": "s1", "text": "a"}, {"type": "usage", "step": "s1", "in": 100, "out": 5},
                       {"type": "say", "step": "s2", "text": "b"}], final=["done"])
    totals = run.document()["final_metrics"]
    assert "total_prompt_tokens" not in totals and "total_completion_tokens" not in totals
    measured, _ = run_task([{"type": "say", "step": "s1", "text": "a"}, {"type": "usage", "step": "s1", "in": 100}],
                           final=["done"])
    assert measured.document()["final_metrics"]["total_prompt_tokens"] == 100


def _spawner():
    @atif_adapter(native_format="toy-spawn/1", version="1", input_includes_cache=True)
    class Spawner(Adapter):
        @native(type="task")
        def task(self, r, turn):
            turn.tool_call(step=r["step"], tool_call_id=r["id"], function_name="Task", arguments={})

        @native(type="spawn")
        def spawn(self, r, turn):
            turn.subagent(r["agent"], spawned_by=r["id"])

        @native(type="sub")
        def sub(self, r, turn):
            turn.message(step=r["step"], text="working", agent=r["agent"])

        @native(type="result")
        def result(self, r, turn):
            turn.tool_result(source_call_id=r["id"], text="done")

        @native(type="say")
        def say(self, r, turn):
            turn.message(step=r["step"], text=r["text"])
    return Spawner


def _refs(read):
    return [ref["trajectory_id"] for step in read["steps"] for result in (step.get("observation") or {}).get("results")
            or [] for ref in result.get("subagent_trajectory_ref") or []]


def test_a_sub_agent_that_wrote_no_step_is_left_out_with_its_pointer():
    run, _ = run_task([{"type": "task", "step": "p1", "id": "c1"}, {"type": "spawn", "id": "c1", "agent": "sub"},
                       {"type": "result", "id": "c1"}], final=["done"], cls=_spawner())
    document = run.document()
    check(document)
    assert _refs(document) == [] and "subagent_trajectories" not in document


def test_a_live_page_shows_a_pointer_only_with_a_step_of_its_sub_agent():
    run = AdapterRun(_spawner())
    run.start_task(CTX)
    pages = []
    for record in [{"type": "task", "step": "p1", "id": "c1"}, {"type": "spawn", "id": "c1", "agent": "sub"},
                   {"type": "sub", "step": "x1", "agent": "sub"}, {"type": "result", "id": "c1"},
                   {"type": "say", "step": "p2", "text": "next"}, {"type": "say", "step": "p3", "text": "more"},
                   {"type": "sub", "step": "x2", "agent": "sub"}]:
        run.native(record)
        pages.append(run.page())
    run.end_task()
    pages.append(run.page())
    assert [len(p["steps"]) if p else 0 for p in pages] == [1, 1, 1, 1, 1, 1, 3, 4]
    for page in filter(None, pages):
        trajectories = {sub["trajectory_id"] for sub in page.get("subagent_trajectories") or []}
        assert set(_refs(page)) <= trajectories and is_prefix(page, run.document())
    assert _refs(pages[-1]) == ["sub"]


def test_a_sub_agent_is_spawned_by_a_call_of_its_task_waiting_on_its_result_and_writes_only_in_that_task():
    run = AdapterRun(_spawner())
    run.start_task(CTX)
    for record in [{"type": "task", "step": "p1", "id": "c1"}, {"type": "result", "id": "c1"}]:
        run.native(record)
    with pytest.raises(AtifError, match="still waiting on its result"):
        run.native({"type": "spawn", "id": "c1", "agent": "late"})
    with pytest.raises(AtifError, match="still waiting on its result"):
        run.native({"type": "spawn", "id": "nothing", "agent": "late"})
    run.native({"type": "task", "step": "p2", "id": "c2"})
    run.native({"type": "spawn", "id": "c2", "agent": "sub"})
    with pytest.raises(AtifError, match="already spawned"):
        run.native({"type": "spawn", "id": "c2", "agent": "other"})
    run.end_task()
    run.start_task(TurnContext(index=1, context_id="ctx", task_id="t2", inbound=["more"]))
    with pytest.raises(AtifError, match="earlier task"):
        run.native({"type": "sub", "step": "x1", "agent": "sub"})


def test_a_file_the_sink_failed_to_take_is_handed_to_it_again():
    taken = []

    def sink(path, data):
        if not taken:
            taken.append(None)
            raise OSError("disk full")
        taken.append(path)

    run = AdapterRun(adapter(), media=sink)
    run.start_task(CTX)
    with pytest.raises(OSError):
        run.media(b"shot", "image/png")
    media = run.media(b"shot", "image/png")
    run.media(b"shot", "image/png")
    assert taken == [None, media.path]


def test_a_document_names_the_trajectory_id_it_was_given():
    run = AdapterRun(adapter(), trajectory_id="doc-1")
    run.start_task(CTX)
    run.end_task(final=["done"])
    document = run.document()
    assert (document["session_id"], document["trajectory_id"]) == ("ctx", "doc-1")
    assert check(document)
    assert "trajectory_id" not in replay(adapter(), [RecordedTask(context=CTX, final=["done"])])


def test_a_document_through_an_ended_task_is_the_document_as_it_stood_then():
    run = AdapterRun(adapter())
    run.start_task(CTX)
    run.native({"type": "say", "step": "a", "text": "first"})
    run.native({"type": "usage", "step": "a", "in": 5, "out": 2})
    run.end_task(final=["first"])
    then = run.document()
    run.start_task(TurnContext(index=1, context_id="ctx", task_id="t2", inbound=["again"]))
    run.native({"type": "say", "step": "b", "text": "second"})
    run.native({"something": "unhandled"})
    assert run.document(through=0) == then
    run.end_task(final=["second"])
    assert run.document(through=0) == then
    assert run.document(through=1) == run.document()
    with pytest.raises(AtifError, match="hasn't ended"):
        run.document(through=2)


def test_a_tasks_diagnostics_and_error_go_on_its_last_step():
    run = AdapterRun(adapter())
    run.start_task(CTX)
    run.native({"type": "say", "step": "a", "text": "trying"})
    run.end_task(state="failed", final=["it broke"], data=[{"data": {"tool_call_count": 0}, "metadata": {"k": 1}}],
                 error={"kind": "agent_error", "message": "it broke", "code": "toy.broke"})
    profile = profile_of(run.document()["steps"][-1])
    assert profile["state"] == "failed"
    assert profile["data"] == [{"data": {"tool_call_count": 0}, "metadata": {"k": 1}}]
    assert profile["error"] == {"kind": "agent_error", "message": "it broke", "code": "toy.broke"}
    assert check(run.document())


@pytest.mark.parametrize("data, error", [
    ([{"tool_call_count": 0}], None),
    ({"data": 1}, None),
    (None, {"kind": "mystery", "message": "x"}),
    (None, {"message": ""}),
    (None, {"message": "x", "extra": 1}),
])
def test_a_tasks_diagnostics_and_error_are_checked_before_anything_is_written(data, error):
    run = AdapterRun(adapter())
    run.start_task(CTX)
    with pytest.raises(AtifError):
        run.end_task(data=data, error=error)


def test_a_spawn_that_never_got_its_result_is_answered_canceled_when_its_task_ends():
    @atif_adapter(native_format="toy-events/1", version="0.1.0", input_includes_cache=False)
    class Spawning(adapter()):
        @native(type="spawn")
        def spawn(self, r, turn):
            turn.subagent("sub", spawned_by=r["id"], name="explorer")

        @native(type="sub")
        def sub(self, r, turn):
            turn.message(step=r["step"], text=r["text"], agent="sub")

    run = AdapterRun(Spawning)
    run.start_task(CTX)
    for record in ({"type": "call", "step": "a", "id": "c1"}, {"type": "spawn", "id": "c1"},
                   {"type": "sub", "step": "s", "text": "looking"}, {"type": "call", "step": "a", "id": "c2"}):
        run.native(record)
    run.end_task(state="canceled")
    document = run.document()
    results = document["steps"][1]["observation"]["results"]
    assert results == [{"source_call_id": "c1", "content": "", "subagent_trajectory_ref": [{"trajectory_id": "sub"}],
                        "extra": {"agentenv": {"status": "canceled"}}}]
    assert check(document)


def test_a_served_turn_gives_the_recorded_task_back():
    run = AdapterRun(adapter())
    shot = run.media(b"\x89PNG", "image/png", width=2, height=2)
    context = TurnContext(index=3, context_id="ctx", task_id="t1", message_id="m1", inbound=["look", shot],
                          agent_name="toy", agent_version="1.0", started_at="2026-11-18T10:00:00.5Z")
    records = [{"type": "say", "step": "a", "text": "a picture"}]
    turn = served_turn(context, state="completed", final=["a picture"], ended_at="2026-11-18T10:00:01Z",
                       data=[{"data": {"n": 1}}])
    assert turn["started_at"] == "2026-11-18T10:00:00.500Z"
    assert turn["inbound"][1] == {"media": {"path": shot.path, "media_type": "image/png", "info": dict(shot.info)}}
    recorded = RecordedTask.from_served(turn, records, [{"name": "hooks", "record": {"x": 1}, "after": 1}])
    assert recorded.context.inbound[1] == shot
    assert recorded.records == [Fed(records[0]), Fed({"x": 1}, aux="hooks")]
    assert (recorded.state, recorded.final, recorded.data) == ("completed", ["a picture"], [{"data": {"n": 1}}])
