"""SetEnvClockRateTaskStep: changes an armed env clock's rate in place, so a task can hold the clock through setup and
start it with the agent, or stop it when the agent stops."""
import httpx
import pytest

from agent_env.env.gateway.clock import Clock, ClockError
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.task_step.context import StepContextError, TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.set_env_clock_rate import SetEnvClockRateTaskStep

T0 = "2026-07-01T22:00:00Z"


class FakeNow:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _GatewayEnv:
    """A deployed env whose clock/v1 calls reach a real gateway Clock, answering a refused write as the gateway does."""

    def __init__(self, clock: Clock, env_id: str = "e1"):
        self.env_id = env_id
        self.clock = clock
        self.calls: list[tuple[str, dict]] = []

    async def invoke(self, uri, method, params=None, timeout=None):
        assert uri == EXT_CLOCK_URI
        self.calls.append((method, params))
        try:
            if method == "set_rate":
                return self.clock.set_rate(params["virtual_seconds_per_real_second"])
            if method == "state":
                return self.clock.state()
        except ClockError as e:
            request = httpx.Request("PUT", "http://gw/clock/set-rate")
            raise httpx.HTTPStatusError("400", request=request, response=httpx.Response(400, text=str(e), request=request))
        raise AssertionError(method)


def _step(rate, **kw):
    return SetEnvClockRateTaskStep(id="clock-rate", version=1, env_id="e1", virtual_seconds_per_real_second=rate,
                                   depends_on=[], **kw)


def _armed(rate):
    now = FakeNow()
    clock = Clock(now_fn=now)
    clock.set_time(T0, rate=rate)
    return clock, now


def test_round_trip_and_registered():
    d = _step(0, timeout_seconds=12).to_dict()
    s = SetEnvClockRateTaskStep.from_dict(d)
    assert d["type"] == "set_env_clock_rate"
    assert (s.env_id, s.virtual_seconds_per_real_second, s.timeout_seconds) == ("e1", 0, 12)
    assert get_task_step_registry()["set_env_clock_rate"] is SetEnvClockRateTaskStep


@pytest.mark.asyncio
async def test_a_clock_held_through_setup_starts_at_its_armed_time():
    clock, now = _armed(rate=0)
    now.t += 600
    await _step(1.0).execute(TaskStepContext(deployed_envs=[_GatewayEnv(clock)]))
    assert clock.read() == {"virtual_time": T0}
    now.t += 60
    assert clock.read() == {"virtual_time": "2026-07-01T22:01:00Z"}


@pytest.mark.asyncio
async def test_stopping_the_clock_holds_virtual_time_and_records_it():
    clock, now = _armed(rate=1.0)
    now.t += 90
    context = await _step(0).execute(TaskStepContext(deployed_envs=[_GatewayEnv(clock)]))
    now.t += 3600
    assert clock.read() == {"virtual_time": "2026-07-01T22:01:30Z"}
    assert context.metadata["clock_configurations"] == [{
        "step_id": "clock-rate", "env_id": "e1", "virtual_seconds_per_real_second": 0,
        "virtual_time": "2026-07-01T22:01:30Z"}]


@pytest.mark.asyncio
async def test_an_unarmed_clock_fails_the_step():
    with pytest.raises(RuntimeError, match="not armed"):
        await _step(0).execute(TaskStepContext(deployed_envs=[_GatewayEnv(Clock(now_fn=FakeNow()))]))


@pytest.mark.asyncio
async def test_an_env_not_deployed_fails_the_step():
    with pytest.raises(StepContextError, match="e1"):
        await _step(0).execute(TaskStepContext(deployed_envs=[_GatewayEnv(Clock(), env_id="other")]))


def test_a_negative_rate_is_refused_at_load():
    with pytest.raises(ValueError, match="non-negative"):
        _step(-1)
