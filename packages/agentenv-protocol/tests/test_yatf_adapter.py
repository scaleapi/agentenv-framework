"""yatf/1 adapters: declared with decorators, fed a context's native and aux records a turn at a time, and stamped by
the turn they write through, so a replay of the stored records makes the same events."""
import jsonschema
import pytest

from agentenv_protocol import yatf
from agentenv_protocol.yatf import Adapter, AdapterRun, Fed, RecordedTurn, TurnContext, aux, finish, native

VALIDATOR = jsonschema.Draft202012Validator(yatf.load_schema())
CREATED = []


@yatf.yatf_adapter(native_format="toy-events/1", version="0.1.0", usage_input_includes_cache=True,
                   tested_with=("toy 1.0",), lossy=("system prompt",))
class Toy(Adapter):
    """A toy CLI whose calls get their times from an aux clock file, so a call waits for its tick."""

    def __init__(self, turn):
        CREATED.append(self)
        self.held = {}
        self.ticks = {}
        self.turns = 0

    @native(type="say")
    def say(self, record, turn):
        turn.message(role="assistant", step=record["id"], text=record["text"])

    @native(type="call")
    def call(self, record, turn):
        self.held[record["id"]] = (record, turn.source)
        self.emit_ready(turn)

    @aux("clock", type="tick")
    def tick(self, record, turn):
        self.ticks[record["id"]] = record["at"]
        self.emit_ready(turn)

    @aux("clock", type="drift")
    def drift(self, record, turn):
        turn.note("the clock drifted")

    def emit_ready(self, turn):
        for call_id in [call_id for call_id in self.held if call_id in self.ticks]:
            record, source = self.held.pop(call_id)
            turn.tool_call(tool_call_id=call_id, function_name=record["name"], arguments=record["args"],
                           step=record["step"], kind="builtin", executor="agent", time=self.ticks[call_id],
                           derived_from=source)

    @native(type="result")
    def result(self, record, turn):
        turn.tool_result(source_call_id=record["id"], text=record["out"], time=record["at"])

    @native(type="usage")
    def usage(self, record, turn):
        turn.step(key=record["step"], model_name="toy-model",
                  metrics={"prompt_tokens": record["prompt"], "completion_tokens": record["completion"],
                           "cost_usd": None, "cost_status": "unpriced"})

    @native(type="event")
    def event(self, record, turn):
        turn.note(record["kind"])

    @native(type="event", where={"kind": "retry"})
    def retry(self, record, turn):
        turn.attempt(record["n"], reason="api error")

    @finish
    def close(self, turn):
        for call_id, (record, source) in self.held.items():
            turn.tool_call(tool_call_id=call_id, function_name=record["name"], arguments=record["args"],
                           step=record["step"], derived_from=source)
        self.held, self.ticks = {}, {}
        self.turns += 1


RECORDS = (
    Fed({"type": "say", "id": "m1", "text": "listing"}),
    Fed({"type": "call", "id": "c1", "step": "m1", "name": "Bash", "args": {"command": "ls"}}),
    Fed({"type": "tick", "id": "c1", "at": "2026-11-17T10:00:01Z"}, aux="clock"),
    Fed({"type": "noise"}, aux="clock"),
    Fed({"type": "result", "id": "c1", "out": "a b", "at": "2026-11-17T10:00:02Z"}),
    Fed({"type": "usage", "step": "m1", "prompt": 10, "completion": 2}),
    Fed({"type": "debug", "detail": 1}),
    Fed({"type": "event", "kind": "compacted"}),
    Fed({"type": "drift"}, aux="clock"),
)


def context(index=0):
    return TurnContext(index=index, context_id="ctx", task_id=f"t{index}", message_id=f"m-{index}",
                       inbound=[{"type": "text", "text": "list files"}], started_at="2026-11-17T10:00:00Z")


def recorded(index=0, records=RECORDS):
    return RecordedTurn(context=context(index), records=records, final=[{"type": "text", "text": "a b"}],
                        ended_at="2026-11-17T10:00:03Z")


@pytest.fixture(autouse=True)
def fresh():
    CREATED.clear()


def test_a_turn_is_framed_stamped_and_valid():
    events = yatf.replay(Toy, [recorded()])
    assert [event["type"] for event in events] == [
        "turn.start", "message", "message", "tool_call", "tool_result", "step", "toy-events/raw", "note", "note",
        "final_response", "turn.end"]
    assert [(event["seq"], event["id"], event["turn"], event["observer"]) for event in events] == [
        (seq, f"0.{seq}", 0, "agent") for seq in range(len(events))]
    for event in events:
        VALIDATOR.validate(event)
    start, inbound, end = events[0], events[1], events[-1]
    assert start["native_format"] == "toy-events/1" and start["usage_input_includes_cache"] is True
    assert start["adapter"] == {"name": "toy-events-yatf", "version": "0.1.0", "lossy": ["system prompt"]}
    assert start["actors"] == [{"id": "agent", "kind": "agent"}, {"id": "caller", "kind": "framework"}]
    assert (inbound["actor"], inbound["step"], inbound["role"]) == ("caller", "in-0", "user")
    assert end["state"] == "completed" and end["tool_call_count"] == 1
    assert end["time"] == {"at": "2026-11-17T10:00:03Z"}


def test_every_event_names_the_record_it_came_from():
    by_type = {}
    for event in yatf.replay(Toy, [recorded()]):
        by_type.setdefault(event["type"], []).append(event.get("derived_from"))
    assert by_type["turn.start"] == [None] and by_type["final_response"] == [None]
    assert by_type["tool_call"] == [{"index": 1}]
    assert by_type["tool_result"] == [{"index": 2}] and by_type["step"] == [{"index": 3}]
    assert by_type["toy-events/raw"] == [{"index": 4}]
    assert by_type["note"] == [{"index": 5}, {"index": 2, "aux": "clock"}]


def test_unmatched_native_records_are_kept_and_unmatched_aux_records_dropped():
    events = yatf.replay(Toy, [recorded()])
    raw = [event for event in events if event["type"] == "toy-events/raw"]
    assert [event["record"] for event in raw] == [{"type": "debug", "detail": 1}]
    assert not any("noise" in str(event) for event in events)


def test_the_most_specific_handler_takes_a_record():
    records = (Fed({"type": "event", "kind": "retry", "n": 2}), Fed({"type": "say", "id": "m2", "text": "again"}))
    events = yatf.replay(Toy, [recorded(records=records)])
    assert [event["type"] for event in events[2:4]] == ["attempt", "message"]
    assert events[2]["number"] == 2 and events[2]["attempt"] == 2 and events[3]["attempt"] == 2
    assert "attempt" not in events[0] and events[-1]["attempt"] == 2


def test_equally_specific_handlers_that_both_match_are_refused():
    @yatf.yatf_adapter(native_format="toy-events/1", version="0.1.0", usage_input_includes_cache=None)
    class Ambiguous(Adapter):
        @native(type="x")
        def by_type(self, record, turn):
            pass

        @native(where={"kind": "y"})
        def by_kind(self, record, turn):
            pass

    with pytest.raises(yatf.YatfError, match="by_kind, by_type"):
        yatf.replay(Ambiguous, [recorded(records=(Fed({"type": "x", "kind": "y"}),))])


def test_one_adapter_serves_the_whole_context():
    events = yatf.replay(Toy, [recorded(0), recorded(1)])
    assert len(CREATED) == 1 and CREATED[0].turns == 2
    second = [event for event in events if event["turn"] == 1]
    assert second[0]["type"] == "turn.start" and second[0]["task_id"] == "t1"
    assert [event["id"] for event in second] == [f"1.{seq}" for seq in range(len(second))]


def test_a_live_run_and_a_replay_of_its_records_make_the_same_events():
    run = AdapterRun(Toy)
    live = run.start_turn(context())
    for fed in RECORDS:
        live += run.native(fed.record) if fed.aux is None else run.aux(fed.aux, fed.record)
    live += run.end_turn(final=[{"type": "text", "text": "a b"}], ended_at="2026-11-17T10:00:03Z")
    assert live == run.events == yatf.replay(Toy, [recorded()])


def test_a_call_still_held_at_turn_end_is_written_without_a_time():
    records = (Fed({"type": "call", "id": "c9", "step": "m9", "name": "Read", "args": {}}),)
    events = yatf.replay(Toy, [recorded(records=records)])
    call = next(event for event in events if event["type"] == "tool_call")
    assert call["time"] == {"at": None} and call["derived_from"] == {"index": 0}


def test_the_frame_is_written_when_finish_raises():
    @yatf.yatf_adapter(native_format="toy-events/1", version="0.1.0", usage_input_includes_cache=None)
    class Broken(Adapter):
        @finish
        def close(self, turn):
            raise RuntimeError("boom")

    run = AdapterRun(Broken)
    run.start_turn(context())
    with pytest.raises(RuntimeError, match="boom"):
        run.end_turn()
    assert run.events[-1]["type"] == "turn.end"
    run.start_turn(context(1))


def test_records_are_fed_only_inside_a_turn_and_only_to_declared_aux_files():
    run = AdapterRun(Toy)
    with pytest.raises(yatf.YatfError, match="no turn is open"):
        run.native({"type": "say"})
    run.start_turn(context())
    with pytest.raises(yatf.YatfError, match="still open"):
        run.start_turn(context(1))
    with pytest.raises(yatf.YatfError, match="no aux file 'rollout'"):
        run.aux("rollout", {})


def test_extension_events_are_namespaced_outside_yatf():
    turn = yatf.YatfTurn(context(), Toy.spec)
    body = {"note": "kept", "model": 1, "event_type": 2, "nulls": 3, "kind": 4}
    assert turn.extension("toy-events/marker", body).model_extra == body
    for kind in ("delegation", "yatf/thing", "note"):
        with pytest.raises(yatf.YatfError, match="isn't an extension type"):
            turn.extension(kind, {})
    with pytest.raises(yatf.YatfError, match="envelope fields: seq"):
        turn.extension("toy-events/marker", {"seq": 3})
    with pytest.raises(yatf.YatfError, match="either content or text"):
        turn.message(role="assistant", text="a", content=[])
    with pytest.raises(yatf.YatfError, match="message event"):
        turn.message(role="assistant", content=[{"type": "text"}])


def test_explicit_nulls_an_adapter_writes_are_kept():
    turn = yatf.YatfTurn(context(), Toy.spec)
    turn.extension("toy-events/marker", {"value": None})
    turn.raw(None)
    turn.reasoning(visibility="omitted", step="s1", opaque=[{"kind": "signature", "value": "sig"}])
    marker, raw, reasoning = turn.events
    assert "value" in marker and marker["value"] is None
    assert "record" in raw and raw["record"] is None
    assert "text" in reasoning and reasoning["text"] is None


def test_an_event_that_is_not_json_is_refused_as_it_is_written():
    turn = yatf.YatfTurn(context(), Toy.spec)
    for arguments in ({"x": float("inf")}, {"x": {1, 2}}, {"x": b"raw"}):
        with pytest.raises(yatf.YatfError, match="isn't JSON"):
            turn.tool_call(tool_call_id="c1", function_name="Bash", arguments=arguments)
    assert turn.events == []
    assert turn.note("next").seq == 0


def test_what_a_run_hands_out_is_a_copy():
    run = AdapterRun(Toy)
    page = run.start_turn(context())
    page[0]["tampered"] = True
    page[1]["content"][0]["text"] = "changed"
    assert "tampered" not in run.events[0] and run.events[1]["content"][0]["text"] == "list files"
    run.events[0]["also"] = 1
    assert "also" not in run.events[0]


def adapter_class(**declared):
    options = {"native_format": "toy-events/1", "version": "0.1.0", "usage_input_includes_cache": None, **declared}

    @yatf.yatf_adapter(**options)
    class Declared(Adapter):
        pass
    return Declared


@pytest.mark.parametrize("declared,message", [
    ({"native_format": "toy"}, "<name>/<version>"),
    ({"native_format": "yatf/1"}, "not a native one"),
    ({"version": ""}, "version"),
    ({"usage_input_includes_cache": "yes"}, "true, false or None"),
    ({"namespace": "Toy"}, "can't name extension types"),
])
def test_a_bad_declaration_is_refused_at_import(declared, message):
    with pytest.raises(yatf.YatfError, match=message):
        adapter_class(**declared)


def test_conflicting_handlers_are_refused_at_import():
    declare = yatf.yatf_adapter(native_format="toy-events/1", version="0.1.0", usage_input_includes_cache=None)
    with pytest.raises(TypeError, match="Adapter subclass"):
        declare(type("Plain", (), {}))

    class Twice(Adapter):
        @native(type="say")
        def one(self, record, turn):
            pass

        @native(type=["say"])
        def two(self, record, turn):
            pass

    class TwoFinishes(Adapter):
        @finish
        def one(self, turn):
            pass

        @finish
        def two(self, turn):
            pass

    class TwoMediaTypes(Adapter):
        @aux("log", media_type="application/jsonl", type="a")
        def one(self, record, turn):
            pass

        @aux("log", media_type="text/plain", type="b")
        def two(self, record, turn):
            pass

    for cls, message in ((Twice, "take the same records"), (TwoFinishes, "more than one @finish"),
                         (TwoMediaTypes, "declared as both")):
        with pytest.raises(yatf.YatfError, match=message):
            declare(cls)
    with pytest.raises(yatf.YatfError, match="isn't declared"):
        AdapterRun(type("Undeclared", (Toy,), {}))


def test_a_subclass_inherits_and_overrides_handlers():
    @yatf.yatf_adapter(native_format="toy-events/2", version="0.2.0", usage_input_includes_cache=True)
    class Louder(Toy):
        @native(type="say")
        def say(self, record, turn):
            turn.message(role="assistant", step=record["id"], text=record["text"].upper())

    events = yatf.replay(Louder, [recorded()])
    assert events[2]["content"] == [{"type": "text", "text": "LISTING"}]
    assert events[0]["native_format"] == "toy-events/2" and events[3]["type"] == "tool_call"


def test_converting_back_is_the_adapters_to_write():
    assert Toy.spec.aux == {"clock": "application/jsonl"} and Toy.spec.finish == "close"
    with pytest.raises(NotImplementedError, match="Toy"):
        Toy.from_yatf([])
    assert Toy.normal_form([{"a": 1}]) == [{"a": 1}]
    assert yatf.Native(log=[[{"type": "say"}]]).session is None
