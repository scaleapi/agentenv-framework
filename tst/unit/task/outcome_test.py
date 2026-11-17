"""A failed step's entry in ``context.metadata.failed_steps`` says what role the step played and why it failed."""

from __future__ import annotations

import inspect

import httpx
import pytest

from agent_env.config import set_document_store
from agent_env.store import LocalSqliteDocumentStore
from agent_env.task.outcome import FailureCause, StepRole, classify_failure
from agent_env.task.store import TASK_INSTANCES_COLLECTION, TaskInstanceStore, set_task_instance_store
from agent_env.task.task import Task
from agent_env.task_step.context import TaskStepContext, VerifierNotPassingError
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.run_code import RunCodeTaskStep
from agent_env.task_step.task_steps.snapshot_agent_state import SnapshotAgentStateTaskStep
from agent_env.task_step.task_steps.snapshot_env import SnapshotEnvTaskStep
from agent_env.task_step.task_steps.verifiers import AgentPromptResponseVerifierTaskStep, RubricsVerifierTaskStep

BUDGET_400 = (
    "Judge A2A task failed (state=failed): API Error: 400 Budget exceeded for team-a: "
    "spent $100.54 of $100.00 per 15d"
)


def _chained(outer: BaseException, inner: BaseException, *, explicit: bool = True) -> BaseException:
    try:
        try:
            raise inner
        except BaseException as e:
            if explicit:
                raise outer from e
            raise outer
    except BaseException as caught:
        return caught


@pytest.mark.parametrize(("exc", "cause"), [
    (RuntimeError(BUDGET_400), FailureCause.KEY_BUDGET),
    (Exception("litellm.BudgetExceededError: Budget has been exceeded! Current cost: 50.54, Max budget: 50.0"),
     FailureCause.KEY_BUDGET),
    (_chained(RuntimeError("LLM judge failed after 3 attempts"), RuntimeError(BUDGET_400)), FailureCause.KEY_BUDGET),
    (_chained(TimeoutError("judge timed out"), RuntimeError(BUDGET_400)), FailureCause.KEY_BUDGET),
    (RuntimeError("Error code: 401 - {'error': {'message': 'Authentication Error, Invalid proxy server token passed. "
                  "valid_token=None.', 'type': 'auth_error', 'code': '401'}}"), FailureCause.KEY_AUTH),
    (RuntimeError("Authentication Error: token_not_found_in_db"), FailureCause.KEY_AUTH),
    (RuntimeError("litellm.AuthenticationError: OpenAIException - Incorrect API key provided: sk-...abcd"),
     FailureCause.KEY_AUTH),
    (RuntimeError("Server disconnected without sending a response."), FailureCause.TRANSIENT),
    (TimeoutError(), FailureCause.TRANSIENT),
    (ConnectionResetError(104, "Connection reset by peer"), FailureCause.TRANSIENT),
    (RuntimeError("API Error: 503 Service Unavailable"), FailureCause.TRANSIENT),
    (RuntimeError("Server error '502 Bad Gateway' for url 'https://judge.example/v1/chat'"), FailureCause.TRANSIENT),
    (RuntimeError("sandbox command exited with code 137 (OOMKilled)"), FailureCause.TRANSIENT),
    (RuntimeError("Error code: 429 - Too Many Requests"), FailureCause.TRANSIENT),
    (httpx.ConnectError("All connection attempts failed"), FailureCause.TRANSIENT),
    (_chained(RuntimeError("agent config not sent"), httpx.ConnectError("[Errno -3] Temporary failure in name resolution")),
     FailureCause.TRANSIENT),
    (RuntimeError("Authentication Error - Expired Key. Key Expiry time 2026-11-01 and current time 2026-11-17"),
     FailureCause.KEY_AUTH),
    (RuntimeError("claude CLI exited with code 1: API Error: 400 Budget exceeded for service_identity:litellm-x: "
                  "spent $50,008.81 of $50,000.00 per 15d (key: x)"), FailureCause.KEY_BUDGET),
    (RuntimeError("API Error: Request rejected (429) · Rate limited. Retry in 1s."),
     FailureCause.TRANSIENT),
    (RuntimeError("codex CLI stalled (attempt 1: unexpected status 408 Request Timeout: request body arrived too slowly)"),
     FailureCause.TRANSIENT),
    (VerifierNotPassingError("verifier 'grade' scored 0.5: rows not met: done"), FailureCause.VERDICT),
    # Text from an env, a tool, a script or a model mentions codes and words the proxy also uses; none of it is ours.
    (RuntimeError("tool get_account returned HTTP 401 Unauthorized from env MCP server"), FailureCause.UNKNOWN),
    (RuntimeError("run_code 'reload': exited 1: urllib.error.HTTPError: HTTP Error 401: Unauthorized"),
     FailureCause.UNKNOWN),
    (ValueError("rubric row 401 has no criteria"), FailureCause.UNKNOWN),
    (ValueError("invalid token in template: {{seed}"), FailureCause.UNKNOWN),
    (AssertionError("expected status 500 to not be returned by create_order"), FailureCause.UNKNOWN),
    (RuntimeError("judge said: user exceeded the budget for travel"), FailureCause.UNKNOWN),
    (RuntimeError("review 'review' timed out after 10800s awaiting a decision"), FailureCause.UNKNOWN),
    # An error raised while handling another, without ``from``, is not caused by it.
    (_chained(ValueError("rubric item 3 has no criteria"), TimeoutError(), explicit=False), FailureCause.UNKNOWN),
    (_chained(RuntimeError("LLM judge failed after 3 attempts"), RuntimeError(BUDGET_400), explicit=False),
     FailureCause.UNKNOWN),
    (ValueError("rubric item 3 has no criteria"), FailureCause.UNKNOWN),
    (ValueError("judge returned invalid JSON: Expecting value: line 1 column 1"), FailureCause.UNKNOWN),
])
def test_classify_failure(exc, cause):
    assert classify_failure(exc) == cause


def test_step_types_declare_their_role():
    assert RubricsVerifierTaskStep.step_role == StepRole.GRADER
    assert AgentPromptResponseVerifierTaskStep.step_role == StepRole.GRADER
    assert SnapshotEnvTaskStep.step_role == StepRole.EVIDENCE
    assert SnapshotAgentStateTaskStep.step_role == StepRole.EVIDENCE
    assert CollectArtifactsTaskStep.step_role == StepRole.EVIDENCE
    assert PromptAgentTaskStep.step_role == StepRole.AGENT
    assert RunCodeTaskStep.step_role == StepRole.OTHER


def test_every_step_type_that_writes_verifications_is_a_grader():
    writers = {
        name for name, cls in get_task_step_registry().items()
        if '"verifications"' in inspect.getsource(inspect.getmodule(cls))
    }
    assert writers, "the scan found no verifier, so it no longer reads the steps' source"
    assert {name for name in writers if get_task_step_registry()[name].step_role != StepRole.GRADER} == set()


def _run_code(**extra) -> dict:
    return {"id": "verify", "type": "run_code", "script_artifact_id": "s", "entrypoint": "main.py", **extra}


def test_a_step_can_declare_its_role_in_config():
    task = Task.from_dict({"id": "t", "type": "task", "version": 1, "steps": [_run_code(step_role="grader")]})
    assert task.steps[0].step_role == StepRole.GRADER
    assert task.to_dict()["steps"][0]["step_role"] == "grader"


def test_a_step_without_a_role_override_serializes_as_before():
    task = Task.from_dict({"id": "t", "type": "task", "version": 1, "steps": [_run_code()]})
    assert "step_role" not in task.to_dict()["steps"][0]


def test_an_unknown_role_is_refused_when_the_task_loads():
    with pytest.raises(ValueError, match="judge"):
        Task.from_dict({"id": "t", "type": "task", "version": 1, "steps": [_run_code(step_role="judge")]})


class _OutOfBudgetVerifier(AgentPromptResponseVerifierTaskStep):
    async def execute(self, context):
        raise RuntimeError(BUDGET_400)


@pytest.fixture
def store(tmp_path):
    doc = LocalSqliteDocumentStore(str(tmp_path / "instances.db"))
    doc.ensure_index(TASK_INSTANCES_COLLECTION, ["instance_id"], unique=True)
    set_document_store(doc)
    s = TaskInstanceStore()
    set_task_instance_store(s)
    try:
        yield s
    finally:
        set_task_instance_store(None)


@pytest.mark.asyncio
async def test_a_tolerated_grader_failure_records_its_role_and_cause(store):
    grader = _OutOfBudgetVerifier("grade", None, prompt_id="p", fail_task_on_error=False)
    ctx = TaskStepContext()
    result = await Task(id="t", version=1, steps=[grader]).run(context=ctx)

    [entry] = result.metadata["failed_steps"]
    assert entry == {
        "step_id": "grade", "step_type": "agent_prompt_response_verifier",
        "error": BUDGET_400, "error_type": "RuntimeError",
        "started_at_utc": entry["started_at_utc"], "duration_seconds": entry["duration_seconds"],
        "is_fatal": False, "step_role": "grader", "cause": "key_budget",
    }
    assert store.get(ctx.instance_id).status == "completed"
    assert store.get(ctx.instance_id).context["metadata"]["failed_steps"] == [entry]
