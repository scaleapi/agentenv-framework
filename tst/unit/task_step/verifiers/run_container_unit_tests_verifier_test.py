"""Unit tests for the reward-file scoring and per-run command/env resolution of
RunContainerUnitTestsVerifierTaskStep."""

from __future__ import annotations

import pytest

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep as Step,
)


def _step(command="python -m convscrape --url \"$URL\"", **kw):
    return Step(id="scrape", version=None, sandbox_name="h", container_name="c",
                command=command, **kw)


def _ctx(*, seed=None, step_params=None):
    ctx = TaskStepContext()
    if seed is not None:
        ctx.metadata["seed"] = seed
    if step_params is not None:
        ctx.metadata["user_overrides"] = {"step_params": {"scrape": step_params}}
    return ctx


@pytest.mark.parametrize("raw,expected", [
    (1, 1.0), (0, 0.0), (0.5, 0.5), ("1", 1.0), ("0\n", 0.0), (" 0.25 ", 0.25),
    (2.0, 1.0),        # clamp high
    (-1.0, 0.0),       # clamp low
    (None, None),      # missing
    ("FAIL", None),    # non-numeric
    (True, None),      # bool is not a reward
    ({"r": 1}, None),  # structured junk
])
def test_parse_reward(raw, expected):
    assert Step._parse_reward(raw) == expected


def test_reward_path_auto_added_to_result_paths():
    step = Step(
        id="s", version=None, sandbox_name="h", container_name="c",
        command="bash /tests/test.sh", reward_path="/logs/verifier/reward.txt",
    )
    assert "/logs/verifier/reward.txt" in step.result_paths


def test_reward_path_roundtrips():
    step = Step(
        id="s", version=None, sandbox_name="h", container_name="c",
        command="bash /tests/test.sh", reward_path="/logs/verifier/reward.txt",
    )
    restored = Step.from_dict(step.to_dict())
    assert restored.reward_path == "/logs/verifier/reward.txt"


def test_default_reward_path_none():
    step = Step(id="s", version=None, sandbox_name="h", container_name="c", command="x")
    assert step.reward_path is None


def test_resolve_command_no_override_no_seed():
    step = _step(env_vars={"A": "1"})
    command, extra_env = step._resolve_command(_ctx())
    assert command == 'python -m convscrape --url "$URL"'
    assert extra_env == {}


def test_step_params_command_override_wins():
    step = _step()
    command, _ = step._resolve_command(_ctx(step_params={"command": "bash /alt.sh"}))
    assert command == "bash /alt.sh"


def test_step_params_env_vars_are_additive():
    step = _step(env_vars={"KEEP": "yes"})
    _, extra_env = step._resolve_command(_ctx(step_params={"env_vars": {"URL": "https://x/a"}}))
    # The stored env_vars are merged by execute(); the helper returns only the overlay.
    assert extra_env == {"URL": "https://x/a"}


def test_seed_exports_upper_cased_env():
    step = _step()
    _, extra_env = step._resolve_command(_ctx(seed={"url": "https://x/a", "min_turns": 8}))
    assert extra_env == {"URL": "https://x/a", "MIN_TURNS": "8"}


def test_seed_value_with_shell_metacharacters_stays_one_argument():
    """A hostile seed value must not become a second command when substituted."""
    step = _step(command="scrape <url>")
    command, extra_env = step._resolve_command(_ctx(seed={"url": "https://x/a; rm -rf /"}))
    assert command == "scrape 'https://x/a; rm -rf /'"
    # Via the env channel it is never parsed as command text at all.
    assert extra_env["URL"] == "https://x/a; rm -rf /"


def test_unknown_placeholder_is_left_literal():
    """No matching seed key → the placeholder survives and fails loudly in-container,
    rather than silently scraping nothing."""
    step = _step(command="scrape <url>")
    command, _ = step._resolve_command(_ctx(seed={"other": "x"}))
    assert command == "scrape <url>"


def test_non_posix_seed_key_is_skipped_but_still_substitutable():
    step = _step(command="scrape <min turns>")
    command, extra_env = step._resolve_command(_ctx(seed={"min turns": "8"}))
    assert command == "scrape 8"
    assert extra_env == {}


def test_override_command_is_itself_seed_rendered():
    step = _step()
    command, _ = step._resolve_command(
        _ctx(seed={"url": "https://x/a"}, step_params={"command": "scrape <url>"})
    )
    # shlex.quote only quotes when the value needs it; a plain URL passes through bare.
    assert command == "scrape https://x/a"


def test_seed_env_wins_over_step_params_env():
    step = _step()
    _, extra_env = step._resolve_command(
        _ctx(seed={"url": "https://seed"}, step_params={"env_vars": {"URL": "https://override"}})
    )
    assert extra_env["URL"] == "https://seed"


# Keys reach `docker exec ... -e K=v` unquoted, so shell syntax in a key escapes the
# docker invocation and runs on the sandbox VM, outside the container.
# "URL\n" is the `$`-vs-`\Z` case: `$` also matches before a trailing newline, so it
# would pass validation and split the docker exec line in two.
@pytest.mark.parametrize("key", ["X; id #", "A B", "1BAD", "K$(id)", "K`id`", "URL\n"])
def test_non_posix_override_env_key_raises(key):
    step = _step()
    with pytest.raises(ValueError, match="is not a POSIX name"):
        step._resolve_command(_ctx(step_params={"env_vars": {key: "v"}}))


@pytest.mark.parametrize("key", ["X; id #", "A B", "1BAD", "URL\n"])
def test_non_posix_stored_env_key_raises(key):
    with pytest.raises(ValueError, match="is not a POSIX name"):
        _step(env_vars={key: "v"})


@pytest.mark.parametrize("col", ["litellm_base_url", "litellm_api_key", "path", "home"])
def test_seed_key_colliding_with_a_reserved_env_var_is_skipped(col):
    """A CSV column that upper-cases onto a name execute() injects would redirect the
    verifier's LLM traffic or swap its credentials."""
    step = _step()
    _, extra_env = step._resolve_command(_ctx(seed={col: "http://attacker"}))
    assert extra_env == {}


def test_explicit_override_may_still_set_a_reserved_env_var():
    """Only the seed channel is guarded — an override is deliberate, not an odd column."""
    step = _step()
    _, extra_env = step._resolve_command(
        _ctx(step_params={"env_vars": {"LITELLM_BASE_URL": "http://internal"}})
    )
    assert extra_env == {"LITELLM_BASE_URL": "http://internal"}


def test_ordinary_seed_keys_are_unaffected_by_the_reserved_set():
    step = _step()
    _, extra_env = step._resolve_command(_ctx(seed={"url": "https://x/a", "min_turns": 8}))
    assert extra_env == {"URL": "https://x/a", "MIN_TURNS": "8"}


def test_nested_placeholder_in_seed_value_is_not_re_substituted():
    """Key-by-key, this rendered `scrape ''$(id)''` — the pass for `b` rewrote the
    placeholder inside the quotes `shlex.quote("<b>")` added, so they closed each other
    and bash ran the substitution."""
    step = _step(command="scrape <a>")
    command, _ = step._resolve_command(_ctx(seed={"a": "<b>", "b": "$(id)"}))
    assert command == "scrape '<b>'"
    assert "$(id)" not in command


def test_seed_placeholders_render_in_one_pass_regardless_of_order():
    step = _step(command="scrape <url> --out <dest>")
    command, _ = step._resolve_command(
        _ctx(seed={"dest": "/tmp/o", "url": "https://x/a"})
    )
    assert command == "scrape https://x/a --out /tmp/o"


def test_seed_key_with_regex_metacharacters_is_matched_literally():
    step = _step(command="scrape <a.b>")
    command, _ = step._resolve_command(_ctx(seed={"a.b": "v", "axb": "BAD"}))
    assert command == "scrape v"


def test_posix_override_and_stored_env_keys_are_accepted():
    step = _step(env_vars={"STORED": "1"})
    _, extra_env = step._resolve_command(_ctx(step_params={"env_vars": {"_URL2": "x"}}))
    assert step.env_vars == {"STORED": "1"}
    assert extra_env == {"_URL2": "x"}
