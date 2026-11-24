"""A turn whose agent reports its counts by model (``Usage.models``) is priced model by model, each at its own rate
from LiteLLM's price map, end to end: the echo agent (tst/data/a2a_agent) sends the breakdown over A2A, the host reads
it, the prompt step's entry keeps a row per model, the task instance's ledger totals what the rows cost, and the run's
summary splits it by model as ``agent-env run`` prints it. Builds the echo agent; needs Docker."""

import json
import logging
import shutil
import subprocess
import uuid

import pytest

from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle import run_bundle
from agent_env.cli.usage_output import cost_line
from agent_env.config import configure, reset_config
from agent_env.config.model_pricing import ReportedBy, Tokens, litellm_prices, price
from agent_env.providers.sandbox_providers.sandbox_provider import (
    reset_agent_sandbox_provider,
    reset_env_sandbox_provider,
    reset_sandbox_provider,
)
from agent_env.store.routing import namespace_routing
from agent_env.task.store import get_task_instance_store
from tst.util.a2a_test_agent import put_test_agent
from tst.util.capabilities import missing_capability_reason

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ROWS = [{"model": "gpt-4.1-mini", "calls": 2, "input_tokens": 9000, "output_tokens": 400, "cache_read_tokens": 8000,
         "input_includes_cache": True},
        {"model": "gpt-4o-mini", "calls": 1, "input_tokens": 3000, "output_tokens": 90, "input_includes_cache": True}]


@pytest.fixture
def bundle(monkeypatch, tmp_path):
    """A bundle whose one task prompts the echo agent to report ``ROWS`` as its turn's counts by model, in local
    stores. Whatever ran, no sandbox work folder may be left behind."""
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip(missing_capability_reason("docker_daemon"))
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    config = tmp_path / "config.toml"
    config.write_text('[model]\nbase_url = "http://127.0.0.1:9/v1"\napi_key = "unused"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    _reset()
    configure()
    agent = put_test_agent(f"usage-models-{uuid.uuid4().hex[:8]}")
    root = tmp_path / "by-model"
    (root / "tasks").mkdir(parents=True)
    (root / "tasks" / "by-model.json").write_text(json.dumps([
        {"id": "solver", "type": "deploy_agent", "agent_name": "solver", "env_ids": [], "a2a_agent_id": agent.id},
        {"id": "prompt", "type": "prompt_agent", "agent_name": "solver", "prompt_id": "p", "model": "gpt-4.1-mini",
         "prompt": "usage-models " + json.dumps(ROWS), "depends_on": [{"task_step_id": "solver"}]},
    ], indent=2))
    logging.disable(logging.CRITICAL)
    try:
        yield root
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        _reset()


def _reset():
    for reset in (reset_config, reset_artifact_store, reset_sandbox_provider, reset_env_sandbox_provider,
                  reset_agent_sandbox_provider):
        reset()


def test_a_turns_counts_by_model_are_priced_at_each_models_rate_and_reach_the_ledger(bundle):
    result = run_bundle(bundle)
    [run] = result.runs
    assert run.error is None, run.error or run.failed_steps

    [entry] = run.context.metadata["step_usage"]["prompt"]
    prices = litellm_prices()
    expected = [price(Tokens.of(row), None, model=row["model"], prices=prices, reported_by=ReportedBy.HARNESS).usd
                for row in ROWS]
    assert [(row["model"], row["calls"], row["cost_status"]) for row in entry["models"]] == [
        ("gpt-4.1-mini", 2, "priced"), ("gpt-4o-mini", 1, "priced")]
    assert [row["cost_usd"] for row in entry["models"]] == pytest.approx(expected)
    assert entry["cost_usd"] == pytest.approx(sum(expected)) and "unit_prices" not in entry
    assert (entry["model"], entry["input_tokens"], entry["output_tokens"]) == ("gpt-4.1-mini", 12000, 490)
    with namespace_routing():
        instance = get_task_instance_store().get(run.context.instance_id)
    assert instance.usage_total["cost_usd"] == pytest.approx(sum(expected))
    summary = run.usage_summary
    assert [(share.role, share.model) for share in summary.shares] == [("solver", "gpt-4.1-mini"),
                                                                     ("solver", "gpt-4o-mini")]
    assert summary.cost_usd == pytest.approx(sum(expected)) and not summary.lower_bound
    line = cost_line(summary)
    assert line.startswith("cost: $") and "solver gpt-4.1-mini $" in line and "9.0k in, 400 out" in line
    assert "solver gpt-4o-mini $" in line and "3.0k in, 90 out" in line
