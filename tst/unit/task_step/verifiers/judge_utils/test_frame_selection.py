"""frame_selection: transcript extraction, the action log, criterion matching, the frame budget and labels."""
from __future__ import annotations

import json
import re

from agent_env.task_step.task_steps.verifiers.judge_utils import frame_selection as FS

_PNG = "iVBORw0KGgo" + "A" * 40


def _span(tool: str, args: dict, shot: str | None = _PNG) -> dict:
    comp = {"result": "ok"}
    if shot:
        comp["screenshot"] = shot
    return {"name": tool, "attributes": {"gen_ai.operation.name": "execute_tool",
                                         "gen_ai.prompt": json.dumps({"tool": tool, "input": args}),
                                         "gen_ai.completion": json.dumps(comp)}}


def _raw(spans: list) -> str:
    return json.dumps([{"attributes": {"gen_ai.operation.name": "chat"}}] + spans)   # a non-tool span too


def _taps(n: int) -> list[dict]:
    return [_span("ios_tap", {"x": i}) for i in range(1, n + 1)]      # spans[k] is action k + 1


def _labels(frames: list[FS.LabeledFrame]) -> list[str]:
    return [f.label for f in frames]


def _indices(labels: list[str]) -> list[int]:
    return [int(re.search(r"action (\d+)$", label).group(1)) for label in labels]


# What a task pins (TrajectoryFilter.always_show_actions, compiled by the verifier): every ios_type call.
_PIN = {"typed": re.compile(r"^ios_type\b", re.I)}
_NAMES = ["alpha", "bravo", "charlie", "delta", "echo"]


def _five_criteria() -> list[dict]:
    return [{"id": f"{n}-open", "description": f"The {n} page is open"} for n in _NAMES]


def _pinned_run() -> list[FS.TrajectoryAction]:
    """30 taps; ios_type at actions 2, 5, ..., 29 (ten pins); the five criteria's matches at taps 4, 7, ..., 16."""
    spans = _taps(30)
    for j in range(2, 30, 3):
        spans[j - 1] = _span("ios_type", {"text": f"t{j}"})
    for k, name in enumerate(_NAMES):
        spans[3 + 3 * k] = _span("ios_tap_element", {"label": f"Open {name}"})
    return FS.transcript_from_raw(_raw(spans)).actions


def test_transcript_from_raw_reads_tool_args_and_screenshot():
    acts = FS.transcript_from_raw(_raw([_span("ios_tap", {"x": 1, "y": 2}),
                                        _span("ios_type", {"text": "latte"}, shot=None)])).actions
    assert [(a.index, a.tool, a.args, a.screenshot is not None) for a in acts] == [
        (1, "ios_tap", {"x": 1, "y": 2}, True), (2, "ios_type", {"text": "latte"}, False)]
    assert FS.transcript_from_raw("not json") == FS.Transcript([], []) == FS.transcript_from_raw(json.dumps({"a": 1}))


def test_action_log_is_one_line_per_call_with_the_result_and_keeps_head_and_tail_over_the_cap():
    acts = FS.transcript_from_raw(_raw([_span(f"t{i}", {"i": i}) for i in range(5)])).actions
    lines = FS.action_log(acts, max_lines=3).splitlines()
    assert lines[0] == '1. t0 {"i": 0} -> {"result": "ok"}'                     # the screenshot is not in the result
    assert lines == [lines[0], '2. t1 {"i": 1} -> {"result": "ok"}', "... (2 lines not shown) ...",
                     '5. t4 {"i": 4} -> {"result": "ok"}']                         # 3 kept: 2 head + 1 tail
    assert FS.action_log(acts, max_lines=5).count("\n") == 4 and "not shown" not in FS.action_log(acts, max_lines=5)


def test_long_args_are_cut_with_an_ellipsis_in_the_log_and_for_matching():
    acts = FS.transcript_from_raw(_raw([_span("ios_tap_element", {"pad": "x" * 200, "label": "zebra"})])).actions
    args_part = FS.action_log(acts).splitlines()[0].split(" -> ")[0]
    assert args_part.endswith("…") and len(args_part) == len("1. ios_tap_element ") + FS.MAX_ARGS_CHARS + 1
    # matching and the pins read the same cut text: a word past the cut is not evidence, whatever the argument
    assert FS.criterion_matches([{"id": "z", "description": "the zebra"}], acts) == []
    typed = FS.transcript_from_raw(_raw([_span("ios_type", {"text": "y" * 200 + " zebra"})])).actions
    assert FS.rendered_action(typed[0]) == "ios_type " + FS._render_json(typed[0].args, FS.MAX_ARGS_CHARS)
    assert not re.search(r"zebra", FS.rendered_action(typed[0]))
    assert FS.criterion_matches([{"id": "z", "description": "the zebra"}], typed) == []
    short = FS.transcript_from_raw(_raw([_span("ios_tap_element", {"label": "zebra"})])).actions
    assert FS.action_log(short).splitlines()[0].startswith('1. ios_tap_element {"label": "zebra"} -> ')
    assert [a.index for _, a in FS.criterion_matches([{"id": "z", "description": "the zebra"}], short)] == [1]


# --------------------------------------------------------------------------- results + agent messages

def _chat(*texts: str, bare_list: bool = False, thinking: str = "") -> dict:
    blocks = ([{"type": "thinking", "thinking": thinking}] if thinking else []) + [{"type": "text", "text": t} for t in texts]
    return {"attributes": {"gen_ai.operation.name": "chat",
                           "gen_ai.completion": json.dumps(blocks if bare_list else {"content": blocks})}}


def _result_span(tool: str, result: object, args: dict | None = None) -> dict:
    span = _span(tool, args or {})
    span["attributes"]["gen_ai.completion"] = json.dumps(result)
    return span


def test_results_are_rendered_without_screenshots_and_bracketed_text_in_a_result_survives():
    note = "Typed 'latte' [verified: false — field could not be read back]"
    acts = FS.transcript_from_raw(json.dumps([
        _result_span("ios_type", {"result": note, "screenshot": _PNG, "meta": {"screenshot": _PNG, "ms": 12}},
                     {"text": "latte"}),
        _result_span("ios_screenshot", {"result": "/9j/" + "B" * 400}),         # a frame returned as text
        _result_span("ios_tap", {"screenshot": _PNG}),                           # nothing but the frame
        _result_span("ios_wait", "done"),                                        # a bare-string result
    ])).actions
    lines = FS.action_log(acts).splitlines()
    assert lines[0] == f'1. ios_type {{"text": "latte"}} -> {{"result": "{note}", "meta": {{"ms": 12}}}}'
    assert lines[1] == '2. ios_screenshot {} -> {"result": "<image>"}'
    assert lines[2] == "3. ios_tap {}"                                           # no result left -> no arrow
    assert lines[3] == '4. ios_wait {} -> "done"'
    assert acts[0].screenshot == _PNG and acts[0].result == {"result": note, "meta": {"ms": 12}}
    assert acts[2].result is None
    assert "iVBOR" not in "\n".join(lines) and "/9j/" not in "\n".join(lines)


def test_long_results_are_cut_at_the_named_cap():
    acts = FS.transcript_from_raw(json.dumps([_result_span("ios_tap", {"result": "x" * 500})])).actions
    result_part = FS.action_log(acts).splitlines()[0].split(" -> ")[1]
    assert result_part.endswith("…") and len(result_part) == FS.MAX_RESULT_CHARS + 1


def test_nested_empty_values_in_a_result_are_kept_and_only_the_top_level_collapses_to_none():
    acts = FS.transcript_from_raw(json.dumps([
        _result_span("ios_tap", {"items": ["", None, 0, "x"], "meta": {"screenshot": _PNG}}),
        _result_span("ios_tap", ""),
        _result_span("ios_tap", {"screenshot": _PNG}),
    ])).actions
    assert acts[0].result == {"items": ["", None, 0, "x"], "meta": {}}          # "" in a list is what the tool said
    assert acts[1].result is None and acts[2].result is None                    # nothing left at the top -> None
    lines = FS.action_log(acts).splitlines()
    assert lines[0] == '1. ios_tap {} -> {"items": ["", null, 0, "x"], "meta": {}}'
    assert lines[1:] == ["2. ios_tap {}", "3. ios_tap {}"]


def test_base64_in_tool_args_and_agent_messages_is_replaced_before_the_cut():
    blob = "iVBORw0KGgo" + "A" * 400
    acts = FS.transcript_from_raw(_raw([_span("ios_tap_element", {"image": blob, "label": "zebra"})])).actions
    line = FS.action_log(acts).splitlines()[0]
    assert line.startswith('1. ios_tap_element {"image": "<image>", "label": "zebra"} -> ') and "iVBOR" not in line
    # the placeholder, not the blob, is what gets cut — so the label after it is still evidence for matching
    assert [a.index for _, a in FS.criterion_matches([{"id": "z", "description": "the zebra"}], acts)] == [1]
    # agent messages: a bare blob split over lines, and an inline data: URL
    t = FS.transcript_from_raw(json.dumps([_chat("look:\n" + blob[:200] + "\n" + blob[200:]),
                                           _chat(f"see data:image/png;base64,{blob} ok")]))
    assert FS.action_log([], messages=t.messages).splitlines() == [
        'agent said: "look: <image>"', 'agent said: "see <image> ok"']


def test_agent_messages_are_logged_in_run_order_and_cut():
    long_msg = "w" * 400
    spans = [_chat("I will open  the\nmenu first.", thinking="private"), _span("ios_tap", {"x": 1}),
             _chat("Menu is open.", "Now the size.", bare_list=True), _span("ios_tap", {"x": 2}),
             {"attributes": {"gen_ai.operation.name": "initial_state", "gen_ai.completion": json.dumps({"screenshot": _PNG})}},
             {"attributes": {"gen_ai.operation.name": "chain",
                             "gen_ai.completion": json.dumps({"content": [{"type": "text", "text": "FINAL"}]})}},
             _chat(long_msg)]
    t = FS.transcript_from_raw(json.dumps(spans))
    assert [(m.after_action, m.text) for m in t.messages] == [
        (0, "I will open  the\nmenu first."), (1, "Menu is open."), (1, "Now the size."), (2, long_msg)]
    assert [a.index for a in t.actions] == [1, 2]                                # initial_state / chain are neither
    lines = FS.action_log(t.actions, messages=t.messages).splitlines()
    assert lines[0] == 'agent said: "I will open the menu first."'               # whitespace collapsed, one line
    assert lines[1].startswith("1. ios_tap")
    assert lines[2:4] == ['agent said: "Menu is open."', 'agent said: "Now the size."']
    assert lines[4].startswith("2. ios_tap")
    assert lines[5] == f'agent said: "{"w" * FS.MAX_AGENT_MESSAGE_CHARS}…"'       # trailing message kept, cut
    assert len(lines) == 6 and "private" not in "\n".join(lines) and "FINAL" not in "\n".join(lines)


def test_the_line_cap_counts_agent_messages_too():
    spans = [_chat("a"), _span("t", {}), _chat("b"), _span("t", {}), _span("t", {})]
    t = FS.transcript_from_raw(json.dumps(spans))
    lines = FS.action_log(t.actions, messages=t.messages, max_lines=3).splitlines()
    assert lines == ['agent said: "a"', '1. t {} -> {"result": "ok"}',            # 2 head (a message counts)
                     "... (2 lines not shown) ...", '3. t {} -> {"result": "ok"}']   # 1 tail
    assert FS.action_log(t.actions, messages=t.messages).count("\n") == 4        # 5 lines under the default cap


def test_a_long_run_keeps_its_first_and_last_lines_around_one_marker():
    spans = [_chat("start")] + [_span("t", {"i": i}) for i in range(1, 300)]     # 1 message + 299 calls = 300 lines
    t = FS.transcript_from_raw(json.dumps(spans))
    lines = FS.action_log(t.actions, messages=t.messages, max_lines=11).splitlines()
    assert len(lines) == 12 and sum("not shown" in line for line in lines) == 1  # 11 kept + the marker
    assert lines[:2] == ['agent said: "start"', '1. t {"i": 1} -> {"result": "ok"}']
    assert lines[6] == "... (289 lines not shown) ..."                           # 6 head, 5 tail
    assert [line.split(".")[0] for line in lines[7:]] == ["295", "296", "297", "298", "299"]
    assert lines[-1] == '299. t {"i": 299} -> {"result": "ok"}'


def test_whole_run_is_sent_and_still_labelled_when_it_fits():
    spans = _taps(4)
    spans[1] = _span("ios_type", {"text": "latte"})                            # action 2
    acts = FS.transcript_from_raw(_raw(spans)).actions
    frames = FS.select_frames_per_criterion(acts, [{"id": "latte-ordered", "description": "A latte is ordered"}],
                                            max_frames=10, always_show=_PIN)
    assert _labels(frames) == ["action 1", "[latte-ordered] final typed action 2",
                               "[latte-ordered] final action 3", "final action 4"]
    assert frames[0].frame.media_type == "image/png"


def test_budget_is_met_exactly_and_reserved_frames_survive():
    spans = _taps(29)
    spans[4] = _span("ios_type", {"text": "hello there"})                    # action 5 (pinned, no criterion words)
    spans[11] = _span("ios_tap_element", {"label": "Add Spain shirt to cart"})   # action 12 -> criterion evidence
    acts = FS.transcript_from_raw(_raw(spans)).actions
    criteria = [{"id": "spain-shirt-in-cart", "description": "The Spain shirt is added to the cart"},
                {"id": "checkout-open", "description": "The checkout page is open"}]
    frames = FS.select_frames_per_criterion(acts, criteria, max_frames=8, always_show=_PIN)
    labels = _labels(frames)
    assert len(frames) == 8                                                   # filled up to the budget, never past it
    assert "typed action 5" in labels                                         # the pinned frame, reserved
    assert "[spain-shirt-in-cart] action 12" in labels                         # match...
    assert "[spain-shirt-in-cart] action 13" in labels                         # ...and its effect frame
    assert {"final action 29", "final action 28", "final action 27"} <= set(labels)   # end state reserved
    assert _indices(labels) == sorted(_indices(labels))                       # run order


def test_trim_drops_effect_frames_first_then_best_matches_round_robin():
    spans = _taps(40)
    for k, name in enumerate(_NAMES):
        spans[5 + 5 * k] = _span("ios_tap_element", {"label": f"Open {name}"})     # actions 6, 11, 16, 21, 26
    acts = FS.transcript_from_raw(_raw(spans)).actions
    criteria = _five_criteria()
    # 5 best matches + 5 effect frames + 3 final = 13 wanted; budget 8 -> the 5 effect frames go, no best match does
    labels = _labels(FS.select_frames_per_criterion(acts, criteria, max_frames=8))
    assert len(labels) == 8
    assert [label for label in labels if label.startswith("[")] == [
        f"[{n}-open] action {6 + 5 * k}" for k, n in enumerate(_NAMES)]
    # budget 6: after the effect frames, best matches go one criterion per round, first criteria first
    labels = _labels(FS.select_frames_per_criterion(acts, criteria, max_frames=6))
    assert len(labels) == 6
    assert [label for label in labels if label.startswith("[")] == [
        "[charlie-open] action 16", "[delta-open] action 21", "[echo-open] action 26"]
    assert {"final action 38", "final action 39", "final action 40"} <= set(labels)


def test_two_criteria_matching_the_same_action_share_its_frame_label():
    spans = _taps(20)
    spans[9] = _span("ios_tap_element", {"label": "Reserve table for four"})   # action 10
    acts = FS.transcript_from_raw(_raw(spans)).actions
    criteria = [{"id": "table-reserved", "description": "A table is reserved"},
                {"id": "party-size", "description": "Party of four"}]
    labels = _labels(FS.select_frames_per_criterion(acts, criteria, max_frames=8))
    assert "[table-reserved][party-size] action 10" in labels
    assert "[table-reserved][party-size] action 11" in labels                # the shared effect frame too
    assert labels.count("[table-reserved][party-size] action 10") == 1


def test_pinned_frame_that_is_a_match_keeps_both_labels_and_label_ids_rewrite_every_tag():
    spans = _taps(20)
    spans[9] = _span("ios_type", {"text": "latte"})                            # action 10
    acts = FS.transcript_from_raw(_raw(spans)).actions
    criteria = [{"id": "latte-ordered", "description": "A latte is ordered"},
                {"id": "drink-latte", "description": "The drink is a latte"}]
    labels = _labels(FS.select_frames_per_criterion(acts, criteria, max_frames=8, always_show=_PIN,
                                                    label_ids={"latte-ordered": "c1", "drink-latte": "c2"}))
    assert "[c1][c2] typed action 10" in labels
    assert not any("latte-ordered" in label or "drink-latte" in label for label in labels)


def test_pins_match_the_rendered_action_and_several_labels_compose_on_one_frame():
    spans = _taps(20)
    spans[9] = _span("ios_type", {"text": "Search latte"})                     # action 10: tool AND args match
    spans[14] = _span("ios_tap_element", {"label": "Search"})                  # action 15: args match only
    acts = FS.transcript_from_raw(_raw(spans)).actions
    pins = {"typed": re.compile(r"^ios_type\b", re.I), "searched": re.compile(r"search", re.I)}
    labels = _labels(FS.select_frames_per_criterion(acts, [], max_frames=8, always_show=pins))
    assert "typed searched action 10" in labels                                # both labels, configuration order
    assert "searched action 15" in labels                                      # a pin over the arguments alone
    assert not any("typed" in label and "15" in label for label in labels)
    # the rendering the pins see is the one the exclude pattern sees: the tool, then the cut args JSON
    assert FS.rendered_action(acts[9]) == 'ios_type {"text": "Search latte"}'


def test_without_pins_nothing_is_reserved_and_no_pin_label_is_written():
    acts, criteria = _pinned_run(), _five_criteria()
    labels = _labels(FS.select_frames_per_criterion(acts, criteria, max_frames=11))
    assert len(labels) == 11 and not any("typed" in label for label in labels)
    idx = _indices(labels)
    assert 2 not in idx and 20 not in idx                                      # the pins below would have kept these
    assert _labels(FS.select_frames_per_criterion(acts, criteria, max_frames=11, always_show={})) == labels


def test_tiny_budget_shrinks_the_final_reservation_and_still_fits():
    acts = FS.transcript_from_raw(_raw(_taps(20))).actions
    assert _labels(FS.select_frames_per_criterion(acts, [], max_frames=2)) == ["action 1", "final action 20"]
    assert _labels(FS.select_frames_per_criterion(acts, [], max_frames=1)) == ["final action 20"]
    n_final = FS.FINAL_FRAMES_RESERVED
    assert len(FS.select_frames_per_criterion(acts, [], max_frames=n_final)) == n_final


def test_pins_over_the_cap_are_spread_over_the_run_keeping_the_first_and_the_last():
    # The criteria's matches and effect frames soak up every spare frame, so the pinned frames that come
    # back are the RESERVED ones, not context fill that happened to land on a pin.
    acts, criteria = _pinned_run(), _five_criteria()

    def pinned(max_frames: int) -> list[int]:
        labels = _labels(FS.select_frames_per_criterion(acts, criteria, max_frames=max_frames, always_show=_PIN))
        assert len(labels) == max_frames
        return _indices([label for label in labels if "typed" in label])

    # Action 29 is pinned AND a final frame: it is kept (and labelled) as a final frame, so it does not
    # count against the pin cap — the cap is spent on the pins that would otherwise be lost.
    assert pinned(11) == [2, 11, 17, 26, 29]   # cap (11 - 3) // 2 = 4 over 2..26: first, last, two spread between
    assert pinned(23) == list(range(2, 30, 3))  # cap 10 covers them all
    assert pinned(6) == [26, 29]               # cap 1: the last pinned action not already final, plus 29


def test_criterion_matching_prefers_unique_terms_and_skips_excluded_actions():
    acts = FS.transcript_from_raw(_raw([_span("ios_tap_element", {"label": "Home screen search"}),
                                        _span("ios_tap_element", {"label": "Add latte to cart"}),
                                        _span("ios_tap_element", {"label": "Checkout"})])).actions
    criteria = [{"id": "latte-in-cart", "description": "A latte is in the cart"},
                {"id": "checkout", "description": "Checkout is reached"}]
    matches = FS.criterion_matches(criteria, acts, exclude_pattern=re.compile(r"home screen", re.I))
    assert [(cid, a.index) for cid, a in matches] == [("latte-in-cart", 2), ("checkout", 3)]
