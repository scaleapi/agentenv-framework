"""Unit tests for the reward-file scoring and per-run command/env resolution of
RunContainerUnitTestsVerifierTaskStep."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import agent_env.providers.sandbox_providers.sandbox_provider as sandbox_provider
from agent_env.config import configure, get_config
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.verifiers import run_container_unit_tests_verifier as verifier_module
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep as Step,
)
from tst.unit.event_loop_probe import on_event_loop


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


class _Sandbox:
    async def exec_script(self, script: str) -> str:
        return ""

    async def exec_with_output(self, *args) -> tuple[int, str, str]:
        return 0, "ok", ""


def test_stdout_and_stderr_are_kept_under_the_fixture_prefix(local_stores, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_FIXTURE_PREFIX", "fx")
    configure()
    provider = SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=_Sandbox()))
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: provider)
    uploaded: list[str] = []

    def record(text, artifact_id, description, object_url):
        uploaded.append(object_url)
        return SimpleNamespace(id=artifact_id, version=1, object_url=object_url)

    monkeypatch.setattr(Step, "_upload_text_artifact", staticmethod(record))
    ctx = TaskStepContext()
    ctx.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "h"}]
    ctx.deployed_sandboxes.append(DeployedSandbox(sandbox_name="h", sandbox_id="sb-1", sandbox_mode="vm"))

    asyncio.run(_step(command="true").execute(ctx))

    outputs = get_config().get_object_store().object_url("fx/verifier-outputs/scrape/")
    assert [url.rsplit("/", 1)[-1] for url in uploaded] == ["stdout.txt", "stderr.txt"]
    assert all(url.startswith(outputs) for url in uploaded)
    [recorded] = ctx.metadata["verifications"].values()
    assert [(recorded[k]["s3_url"], recorded[k]["object_url"]) for k in ("stdout_artifact", "stderr_artifact")] == [
        (url, url) for url in uploaded
    ]


def test_stdout_and_stderr_upload_off_the_event_loop(local_stores, monkeypatch):
    configure()
    provider = SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=_Sandbox()))
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: provider)
    on_loop: list[bool] = []

    def record(text, artifact_id, description, object_url):
        on_loop.append(on_event_loop())
        return SimpleNamespace(id=artifact_id, version=1, object_url=object_url)

    monkeypatch.setattr(Step, "_upload_text_artifact", staticmethod(record))
    ctx = TaskStepContext()
    ctx.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "h"}]
    ctx.deployed_sandboxes.append(DeployedSandbox(sandbox_name="h", sandbox_id="sb-1", sandbox_mode="vm"))

    asyncio.run(_step(command="true").execute(ctx))

    assert on_loop == [False, False]


@pytest.mark.parametrize(
    "instance_ids", [("i-1", "i-2"), ("i-1", "i-1"), (None, None)], ids=["instance-ids", "retried-attempt", "no-instance-id"]
)
def test_runs_and_retries_in_one_second_keep_their_outputs_apart(local_stores, monkeypatch, instance_ids):
    """A retried step keeps its run's instance id, so the key must also be new for each execution."""
    configure()
    provider = SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=_Sandbox()))
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", lambda: provider)
    monkeypatch.setattr(verifier_module.time, "time", lambda: 1_790_000_000.0)
    uploads: list[tuple[str, str]] = []

    def record(text, artifact_id, description, object_url):
        uploads.append((artifact_id, object_url))
        return SimpleNamespace(id=artifact_id, version=1, object_url=object_url)

    monkeypatch.setattr(Step, "_upload_text_artifact", staticmethod(record))

    def ctx(instance_id):
        c = TaskStepContext(instance_id=instance_id)
        c.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "h"}]
        c.deployed_sandboxes.append(DeployedSandbox(sandbox_name="h", sandbox_id="sb-1", sandbox_mode="vm"))
        return c

    async def both():
        step = _step(command="true")
        await asyncio.gather(*(step.execute(ctx(i)) for i in instance_ids))

    asyncio.run(both())

    assert len({artifact_id for artifact_id, _ in uploads}) == len({url for _, url in uploads}) == 4
