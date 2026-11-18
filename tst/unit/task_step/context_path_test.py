"""Paths into a run's context: dotted keys, or any id in brackets, as a bundle writes an id it renamed."""
from __future__ import annotations

import pytest

from agent_env.task_step.context import context_path_segments, renamed_context_path

ID = "@local/~/runs/pn.v2/tickets"


@pytest.mark.parametrize("path, segments", [
    ("verifications.cart-check.count", ["verifications", "cart-check", "count"]),
    (f"env_trigger_state[{ID}].statuses.wf-1", ["env_trigger_state", ID, "statuses", "wf-1"]),
    (f"[{ID}].rows", [ID, "rows"]),
    ("items.0", ["items", "0"]),
])
def test_a_path_walks_its_keys_and_a_bracketed_id_is_one_key(path, segments):
    assert context_path_segments(path) == segments


@pytest.mark.parametrize("path", ["", "a..b", "a.", ".a", "a/b", "a[b", "a[]", "a[[b]]", "a b", None, 3])
def test_anything_else_is_no_path(path):
    assert context_path_segments(path) is None


def test_a_renamed_id_is_bracketed_unless_a_dotted_path_can_hold_it():
    rename = {"tickets": ID, "crm": "triage-crm"}.get

    assert renamed_context_path("env_trigger_state.tickets.statuses.wf-1", rename) == (
        f"env_trigger_state[{ID}].statuses.wf-1")
    assert renamed_context_path("env_changes.crm.rows", rename) == "env_changes.triage-crm.rows"
    assert renamed_context_path("tickets.rows", rename) == f"[{ID}].rows"
    assert context_path_segments(renamed_context_path("tickets.rows", rename)) == [ID, "rows"]


@pytest.mark.parametrize("path", ["verifications.judge.score", "a note about tickets", f"env_trigger_state[{ID}].x"])
def test_a_path_naming_no_renamed_id_and_a_string_that_isnt_a_path_are_left_as_they_are(path):
    assert renamed_context_path(path, {"tickets": ID}.get) == path
