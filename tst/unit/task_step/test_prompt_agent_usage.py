"""A prompt step keeps the usage its agents report: one entry per role, the solver's turns and the user sim's each
summed, with the model, the endpoint's host and the attempt epoch, in the step's prompt response and, as each turn ends,
in ``metadata["step_usage"]``, so a step that times out keeps the turns before it."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from a2a.types import TaskState

import agent_env.a2a_agent as a2a_agent_package
from agent_env.task_step.context import ATTEMPT_EPOCH, DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa_mod
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep


def _turn(input_tokens, output_tokens, **more):
    return {"tool_call_count": 1, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens, **more}


def _wire(monkeypatch, solver_usage: list, user_usage=None):
    """The conversation's HTTP and store boundary. The solver's n-th reply reports ``solver_usage[n]`` (no usage when
    None; a TimeoutError raised instead when it is one); the user sim reports ``user_usage`` and never says done."""
    cfg = MagicMock()
    cfg.get_model_params.return_value = {}
    monkeypatch.setattr(pa_mod, "get_config", lambda: cfg)
    for fn in ("create_conversation", "add_a2a_task", "complete_a2a_task", "mark_closed", "get_conversation"):
        monkeypatch.setattr(pa_mod.conversation_store, fn, lambda *a, **kw: None)
    solver_turns = iter(solver_usage)

    async def post_agent_config(url, fields, headers=None):
        return None

    async def send(url, parts, message_id, context_id, timeout):
        return f"task-{message_id}", None

    async def poll(url, task_id, timeout, interval, **_):
        if "user-sim" in url:
            parts = [{"kind": "text", "text": '{"message": "keep going", "done": false}'}]
            usage = user_usage
        else:
            usage = next(solver_turns)
            if isinstance(usage, Exception):
                raise usage
            parts = [{"kind": "text", "text": "done"}]
        if usage is not None:
            parts.append({"kind": "data", "data": {"usage": usage}})
        return {"status": {"state": TaskState.completed, "message": {"parts": parts}}}

    monkeypatch.setattr(pa_mod.protocol, "post_agent_config", post_agent_config)
    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    monkeypatch.setattr(pa_mod.protocol, "poll_a2a_task", poll)


def _context() -> TaskStepContext:
    ctx = TaskStepContext(instance_id="ti-1")
    ctx.deployed_agents.append(DeployedAgent(agent_name="solver", api_url="http://solver", a2a_url="http://solver",
                                             a2a_card={}, model_endpoint_host="proxy.example"))
    ctx.deployed_agents.append(DeployedAgent(agent_name="usersim", api_url="http://user-sim",
                                             a2a_url="http://user-sim", a2a_card={},
                                             model_endpoint_host="sim-proxy.example:4000"))
    return ctx


def _step(turns: int) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver",
                               max_conversation_turns=turns, user_agent_name="usersim")


@pytest.mark.asyncio
async def test_each_roles_turns_are_summed_into_one_entry(monkeypatch):
    _wire(monkeypatch, [_turn(100, 10, cache_read_tokens=60, input_includes_cache=True, cost_usd=0.01),
                        _turn(200, 20, cache_read_tokens=150, input_includes_cache=True, cost_usd=0.02)],
          user_usage=_turn(30, 5, input_includes_cache=True))
    token = ATTEMPT_EPOCH.set(2)
    try:
        ctx = await _step(2).execute(_context())
    finally:
        ATTEMPT_EPOCH.reset(token)

    solver, user = ctx.prompt_responses[-1].usage
    assert solver == {
        "role": "solver", "model": None, "endpoint_host": "proxy.example", "attempt": 2, "turns": 2,
        "tool_call_count": 2, "input_tokens": 300, "output_tokens": 30, "total_tokens": 330,
        "cache_read_tokens": 210, "cache_write_tokens": None, "input_includes_cache": True, "cost_usd": 0.03,
        "partial": False}
    assert (user["role"], user["endpoint_host"], user["turns"], user["input_tokens"], user["partial"]) == (
        "user_simulator", "sim-proxy.example:4000", 1, 30, False)
    assert ctx.metadata["step_usage"]["solve"] == [solver, user]
    assert ctx.metadata["model_endpoints"] == {
        "solver": {"solve": {"model": None, "endpoint_host": "proxy.example"}},
        "user_simulator": {"solve": {"model": None, "endpoint_host": "sim-proxy.example:4000"}}}


@pytest.mark.asyncio
async def test_an_agent_that_reports_only_tool_calls_gets_no_token_counts(monkeypatch):
    _wire(monkeypatch, [{"tool_call_count": 3}])

    ctx = await _step(1).execute(_context())

    [solver] = ctx.prompt_responses[-1].usage
    assert solver["tool_call_count"] == 3 and ctx.prompt_responses[-1].tool_call_count == 3
    assert [solver[name] for name in ("input_tokens", "output_tokens", "cost_usd")] == [None, None, None]
    assert solver["partial"] is False


@pytest.mark.asyncio
async def test_a_sum_some_turns_or_calls_left_out_is_partial(monkeypatch):
    _wire(monkeypatch, [_turn(100, 10), None, _turn(50, 5, provider_details={"unreported_calls": 2})])

    ctx = await _step(3).execute(_context())

    solver = ctx.prompt_responses[-1].usage[0]
    assert (solver["turns"], solver["input_tokens"], solver["partial"], solver["unreported_calls"]) == (3, 150, True, 2)


@pytest.mark.asyncio
async def test_a_step_that_times_out_keeps_the_turns_before_it(monkeypatch):
    _wire(monkeypatch, [_turn(100, 10), _turn(200, 20), TimeoutError("no answer")], user_usage=_turn(30, 5))
    ctx = _context()

    with pytest.raises(TimeoutError):
        await _step(3).execute(ctx)

    assert not ctx.prompt_responses
    solver, user = ctx.metadata["step_usage"]["solve"]
    assert (solver["turns"], solver["input_tokens"], user["turns"]) == (2, 300, 2)


@pytest.mark.asyncio
async def test_malformed_usage_is_kept_aside_and_the_step_runs(monkeypatch):
    _wire(monkeypatch, [{"tool_call_count": 1, "input_tokens": -5, "output_tokens": 7}, "garbage"])

    ctx = await _step(2).execute(_context())

    solver = ctx.prompt_responses[-1].usage[0]
    assert (solver["input_tokens"], solver["output_tokens"], solver["partial"]) == (None, 7, True)
    assert solver["unparsed"] == [{"input_tokens": -5}, {"usage": "garbage"}]


class _Agent:
    """An agent whose deploy is faked; the endpoint it records is the one the deploy is given, as the real rule says."""

    metadata: dict = {}

    async def deploy(self, **_):
        return SimpleNamespace(a2a_url="http://solver", agent_card={}, sandbox_id="sb-1", sandbox_type="fake",
                               instance_id="i-1", network_policy=None, control_token=None, control_headers={})

    def model_base_url(self, env_vars, config):
        return env_vars.get("LITELLM_BASE_URL")


@pytest.mark.asyncio
async def test_no_key_or_userinfo_reaches_what_a_run_stores(monkeypatch):
    monkeypatch.setattr(a2a_agent_package.A2AAgent, "get", classmethod(lambda cls, *a, **k: _Agent()))
    context = TaskStepContext(instance_id="ti-1")
    context.metadata["user_overrides"] = {"litellm_api_key": "sk-live-secret"}
    await DeployAgentTaskStep(id="deploy", version=None, a2a_agent_id="a", agent_name="solver",
                              litellm_base_url="https://alice:hunter2@gw.example:8443/v1").execute(context)
    _wire(monkeypatch, [_turn(10, 1)])

    await PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver").execute(context)

    assert context.prompt_responses[-1].usage[0]["endpoint_host"] == "gw.example:8443"
    stored = json.dumps(context.to_safe_dict(), default=str)
    assert "gw.example:8443" in stored
    assert not [secret for secret in ("alice", "hunter2", "sk-live-secret") if secret in stored]
