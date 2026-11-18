"""The validator: the fixtures and every live page they were served are clean, a negative fixture for each rule fails
with that rule's id at the severity PROFILE.md gives it, and the CLI reports it."""
import copy
import json
import subprocess
import sys
from importlib.resources import files
from typing import Any, Callable

import pytest

from agentenv_protocol.atif import KIT_RULES, RULES, errors, validate, validate_live
from agentenv_protocol.atif.__main__ import main
from atif_cases import CASES, FIXTURES, build

SPEC = files("agentenv_protocol.atif").joinpath("PROFILE.md").read_text(encoding="utf-8")


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _rules(findings, severity: str | None = None) -> set[str]:
    return {f.rule for f in findings if severity is None or f.severity == severity}


def _parts(document: dict[str, Any]) -> list[dict[str, Any]]:
    return [part for step in document["steps"] for result in (step.get("observation") or {}).get("results", [])
            for part in result.get("content") or [] if isinstance(part, dict) and part.get("type") == "image"]


def _duplicate_call_id(document: dict[str, Any]) -> None:
    step = document["steps"][2]
    step["tool_calls"][0]["tool_call_id"] = "toolu_01"
    step["observation"]["results"][0]["source_call_id"] = "toolu_01"


def _dangling_reference(document: dict[str, Any]) -> None:
    for step in document["steps"]:
        for result in (step.get("observation") or {}).get("results", []):
            for ref in result.get("subagent_trajectory_ref") or []:
                ref["trajectory_id"] = "nobody"


def _rename_media(document: dict[str, Any]) -> None:
    _parts(document)[0]["source"]["path"] = "screens/1.png"


def _one_more(field: str) -> Callable[[dict[str, Any]], None]:
    def mutate(document: dict[str, Any]) -> None:
        document["final_metrics"][field] += 1
    return mutate


def _over_cached(document: dict[str, Any]) -> None:
    metrics = document["steps"][1]["metrics"]
    metrics["cached_tokens"] = metrics["prompt_tokens"] + 1


def _text_parts(document: dict[str, Any]) -> None:
    step = document["steps"][-1]
    step["message"] = [{"type": "text", "text": step["message"]}]


#: A fixture and the change that breaks one rule.
NEGATIVE: dict[str, tuple[str, Callable[[dict[str, Any]], None]]] = {
    "json": ("coding", lambda d: d["steps"][1]["tool_calls"][0]["arguments"].update(x=float("nan"))),
    "atif": ("coding", lambda d: d["steps"][1].update(step_id=9)),
    "profile": ("coding", lambda d: d["steps"][0]["extra"]["agentenv"].update(turn="zero")),
    "profile-key": ("coding", lambda d: d["steps"][0]["extra"]["agentenv"].update(later=1)),
    "profile-version": ("coding", lambda d: d["extra"]["agentenv"].update(version="2.0")),
    "vocabulary": ("coding", lambda d: d["steps"][-1]["extra"]["agentenv"].update(state="paused")),
    "outside-key": ("coding", lambda d: d["steps"][0]["extra"].update(note="x")),
    "call-id": ("coding", _duplicate_call_id),
    "reference": ("subagent", _dangling_reference),
    "totals": ("coding", _one_more("total_prompt_tokens")),
    "total-steps": ("coding", _one_more("total_steps")),
    "usage": ("coding", _over_cached),
    "media-path": ("computer_use", _rename_media),
    "timestamp": ("coding", lambda d: d["steps"][0].update(timestamp="2026-11-04T17:02:11.204+00:00")),
    "harbor-save": ("coding", lambda d: d["steps"][1].update(model_name=None)),
    "text-message": ("coding", _text_parts),
}


def _negative(rule: str) -> dict[str, Any]:
    name, mutate = NEGATIVE[rule]
    document = _fixture(name)
    mutate(document)
    return document


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_every_fixture_is_clean_for_a_producer_and_for_a_reader(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    assert validate(document, media=FIXTURES) == []
    assert validate(document, reader=True, media=FIXTURES) == []


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_page_a_case_served_passes_the_live_rule(name):
    document, pages, media = build(name)
    assert validate_live(pages, document, media=media) == []


def test_every_rule_has_a_negative_fixture():
    assert set(NEGATIVE) | {"media-bytes", "live-prefix", "live-unanswered"} == set(RULES)


@pytest.mark.parametrize("rule", sorted(NEGATIVE))
def test_a_negative_fixture_fails_with_its_rules_id_and_the_cli_reports_it(rule, tmp_path, capsys):
    document = _negative(rule)
    assert rule in _rules(errors(validate(document)))
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert main(["validate", str(path)]) == 1
    assert f"error {rule} " in capsys.readouterr().out


@pytest.mark.parametrize("rule", sorted(NEGATIVE))
def test_a_reader_sees_each_rule_at_its_reader_severity(rule):
    found = {f.severity for f in validate(_negative(rule), reader=True) if f.rule == rule}
    assert found == ({RULES[rule].reader} if RULES[rule].reader else set())


def test_an_unknown_profile_key_fails_a_producer_and_only_warns_a_reader(tmp_path, capsys):
    document = _negative("profile-key")
    assert [(f.rule, f.severity, f.where) for f in validate(document)] == [
        ("profile-key", "error", "steps[0].extra.agentenv.later")]
    assert [(f.rule, f.severity) for f in validate(document, reader=True)] == [("profile-key", "warning")]
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert main(["validate", "--reader", str(path)]) == 0
    assert "warning profile-key steps[0].extra.agentenv.later" in capsys.readouterr().out


def test_media_bytes_are_checked_against_their_name_and_profile():
    document = _fixture("computer_use")
    path = _parts(document)[0]["source"]["path"]
    assert validate(document, media=FIXTURES) == []
    assert "media-bytes" in _rules(errors(validate(document, media={})))
    assert "media-bytes" in _rules(errors(validate(document, media={path: b"not the screenshot"})))


def test_totals_sum_the_sub_agents_steps_and_skip_a_count_a_step_lacks():
    document = _fixture("subagent")
    root = sum(s["metrics"]["prompt_tokens"] for s in document["steps"] if s.get("metrics"))
    assert root < document["final_metrics"]["total_prompt_tokens"]
    assert validate(document) == []
    sub_step = next(s for s in document["subagent_trajectories"][0]["steps"] if s.get("metrics"))
    del sub_step["metrics"]["cached_tokens"]
    assert validate(document) == []
    document["final_metrics"]["total_prompt_tokens"] += 1
    assert [f.where for f in validate(document)] == ["final_metrics.total_prompt_tokens"]
    sub_step["metrics"]["prompt_tokens"] = None
    assert "totals" not in _rules(validate(document))


def _open_page(name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    """A case's final document and a page served while its task ran, with two closed steps."""
    document, pages, media = build(name)
    page = next(p for p in pages[:-1] if p is not None and len(p["steps"]) >= 3)
    return document, copy.deepcopy(page), media


def test_the_live_check_flags_a_page_that_differs_from_the_final_read_by_one_step():
    document, page, media = _open_page("coding")
    assert validate_live([page], document, media=media) == []
    page["steps"][1]["message"] = "a step served, then changed"
    assert [(f.rule, f.where) for f in errors(validate_live([page], document, media=media))] == [
        ("live-prefix", "final: ")]


def test_the_live_check_flags_a_page_serving_a_call_with_no_result_before_its_task_ends():
    document, page, media = _open_page("coding")
    del page["steps"][1]["observation"]
    assert "live-unanswered" in _rules(errors(validate_live([page], document, media=media)))


def test_a_page_that_serves_nothing_after_serving_steps_breaks_the_live_rule():
    document, page, media = _open_page("coding")
    assert "live-prefix" in _rules(errors(validate_live([page, None], document, media=media)))


def test_python_m_runs_the_cli(tmp_path):
    clean = subprocess.run([sys.executable, "-m", "agentenv_protocol.atif", "validate", "--media", str(FIXTURES),
                            str(FIXTURES / "computer_use.json")], capture_output=True, text=True)
    assert (clean.returncode, clean.stdout.splitlines()[-1].split(": ")[-1]) == (0, "valid, 0 errors, 0 warnings")
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(_negative("call-id")), encoding="utf-8")
    broken = subprocess.run([sys.executable, "-m", "agentenv_protocol.atif", "validate", str(path)],
                            capture_output=True, text=True)
    assert broken.returncode == 1
    assert "error call-id steps[2].tool_calls[0]: 'toolu_01' is also the id at steps[1].tool_calls[0]" in broken.stdout


def test_the_cli_exits_2_on_a_file_it_cant_read(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "missing.json")]) == 2
    assert "can't read" in capsys.readouterr().err


def test_profile_md_lists_every_rule_with_its_severities():
    for rule in RULES.values():
        assert f"| `{rule.id}` | {rule.holds} | {rule.producer} | {rule.reader or 'not checked'} |" in SPEC
    for rule in KIT_RULES.values():
        assert f"| `{rule.id}` | {rule.holds} | {rule.producer} |" in SPEC
