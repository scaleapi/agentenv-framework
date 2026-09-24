"""Behavior tests for ApplyServerConfigStep: config directives are invoked against
the right per-service gateway-proxy URL + advertised extension endpoint, results
land in context.metadata, and bad targets / unadvertised extensions raise typed
errors. Only the HTTP boundary (the gateway proxy) is mocked; the step logic, the
directive schema, and the agentenv_protocol card helpers run for real."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.apply_server_config import (
    ApplyServerConfigError,
    ApplyServerConfigStep,
    ConfigDirective,
)

_SET_ERRORS_URI = "urn:agentenv:set-errors/v1"
_SET_ASYNC_WAIT_URI = "urn:agentenv:set-async-wait/v1"
_SET_ACTING_USER_URI = "urn:agentenv:set-acting-user/v1"

_GATEWAY = "http://gw:18765"


def _resp(status: int, url: str, body: dict) -> httpx.Response:
    """httpx.Response with a request attached so raise_for_status() works
    (real responses always carry one; hand-built ones must be given it)."""
    return httpx.Response(status, json=body, request=httpx.Request("GET", url))


def _card(*extensions: dict) -> dict:
    """Minimal EnvironmentCard as served at /.well-known/agent-env.json."""
    return {
        "name": "slack",
        "protocolVersion": "1.0",
        "capabilities": {"extensions": list(extensions)},
    }


def _ext(uri: str, endpoint: str, method: str = "POST") -> dict:
    """A card extension advertised the way the @extension decorator emits it."""
    op = endpoint.rsplit("/", 1)[-1]
    return {
        "uri": uri,
        "description": "test extension",
        "params": {"endpoint": endpoint, "methods": {op: {"method": method, "request": {}}}},
    }


def _make_context(*, env_id="env-x", gateway_url=_GATEWAY, metadata=None):
    env = DeployedEnv(
        env_id=env_id, env_version=1, gateway_url=gateway_url, mcp_url="",
        db_web_url=None, sandbox_id="sb-env",
    )
    return TaskStepContext(deployed_envs=[env], metadata={} if metadata is None else metadata)


def _step(**overrides):
    base = {
        "id": "cfg-1",
        "version": None,
        "env_id": "env-x",
        "directives": [
            {
                "service": "slack",
                "uri": _SET_ERRORS_URI,
                "args": {"tool_name": "slack_send_message", "error_rate": 0.5, "error_type": "rate_limit"},
            }
        ],
    }
    base.update(overrides)
    return ApplyServerConfigStep(**base)


class _MockGateway:
    """Stands in for the gateway REST proxy. Serves a per-service card at
    /svc/mcp-<service>/.well-known/agent-env.json and echoes extension POSTs.
    Records every request so tests can assert on URL + body."""

    def __init__(self, cards: dict[str, dict]):
        self.cards = cards  # service -> card dict
        self.get_calls: list[str] = []
        self.post_calls: list[tuple[str, dict]] = []

    async def get(self, url, *, timeout=None, params=None):
        self.get_calls.append(url)
        for service, card in self.cards.items():
            if f"/svc/mcp-{service}/.well-known/agent-env.json" in url:
                return _resp(200, url, card)
        return _resp(404, url, {"error": "no card"})

    async def request(self, method, url, *, json=None, timeout=None):
        self.post_calls.append((url, json))
        return _resp(200, url, {"ok": True, "echo": json})

    # invoke_extension uses client.get for GET verbs and client.request otherwise;
    # our extensions are POST, so request() handles them.
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _patch_httpx(gateway: _MockGateway):
    """Patch httpx.AsyncClient(...) so both get_card and invoke_extension talk to
    our in-memory gateway. AsyncClient is a context manager, so return the mock."""
    return patch("httpx.AsyncClient", return_value=gateway)


@pytest.mark.asyncio
async def test_applies_directive_to_correct_service_url_and_records_metadata():
    gw = _MockGateway({"slack": _card(_ext(_SET_ERRORS_URI, "/agentenv/ext/set_errors"))})
    ctx = _make_context()
    with _patch_httpx(gw):
        out = await _step().execute(ctx)

    # Card fetched through the per-service proxy prefix.
    assert any(f"{_GATEWAY}/svc/mcp-slack/.well-known/agent-env.json" == u for u in gw.get_calls)
    # Extension POSTed to the advertised endpoint under that same prefix.
    assert len(gw.post_calls) == 1
    url, body = gw.post_calls[0]
    assert url == f"{_GATEWAY}/svc/mcp-slack/agentenv/ext/set_errors"
    assert body == {"tool_name": "slack_send_message", "error_rate": 0.5, "error_type": "rate_limit"}

    changes = out.metadata["server_config_changes"]
    assert len(changes) == 1
    assert changes[0]["step_id"] == "cfg-1"
    assert changes[0]["service"] == "slack"
    assert changes[0]["uri"] == _SET_ERRORS_URI
    assert changes[0]["result"]["ok"] is True


@pytest.mark.asyncio
async def test_multiple_directives_same_service_fetches_card_once():
    gw = _MockGateway({
        "slack": _card(
            _ext(_SET_ERRORS_URI, "/agentenv/ext/set_errors"),
            _ext(_SET_ASYNC_WAIT_URI, "/agentenv/ext/set_async_wait"),
        )
    })
    step = _step(directives=[
        {"service": "slack", "uri": _SET_ERRORS_URI, "args": {"tool_name": "a", "error_rate": 1.0}},
        {"service": "slack", "uri": _SET_ASYNC_WAIT_URI, "args": {"tool_name": "b", "min_seconds": 5, "max_seconds": 10}},
    ])
    with _patch_httpx(gw):
        out = await step.execute(_make_context())

    # One card GET (cached), two extension POSTs.
    assert len(gw.get_calls) == 1
    assert len(gw.post_calls) == 2
    assert {u for u, _ in gw.post_calls} == {
        f"{_GATEWAY}/svc/mcp-slack/agentenv/ext/set_errors",
        f"{_GATEWAY}/svc/mcp-slack/agentenv/ext/set_async_wait",
    }
    assert len(out.metadata["server_config_changes"]) == 2


@pytest.mark.asyncio
async def test_unadvertised_extension_raises():
    # Card served, but the requested extension URI is not on it.
    gw = _MockGateway({"slack": _card(_ext(_SET_ASYNC_WAIT_URI, "/agentenv/ext/set_async_wait"))})
    with _patch_httpx(gw):
        with pytest.raises(ApplyServerConfigError, match="not advertised on card"):
            await _step().execute(_make_context())
    # Nothing was invoked.
    assert gw.post_calls == []


@pytest.mark.asyncio
async def test_tolerate_unadvertised_skips_instead_of_raising():
    # Broadcast set_acting_user against a service that hasn't opted into the
    # extension: with tolerate_unadvertised it is skipped + recorded, not raised.
    gw = _MockGateway({"slack": _card(_ext(_SET_ERRORS_URI, "/agentenv/ext/set_errors"))})
    ctx = _make_context()
    step = _step(
        tolerate_unadvertised=True,
        directives=[
            {"service": "slack", "uri": _SET_ACTING_USER_URI, "args": {"user_email": "a@x.com"}},
        ],
    )
    with _patch_httpx(gw):
        out = await step.execute(ctx)

    assert gw.post_calls == []  # nothing invoked
    assert out.metadata.get("server_config_changes", []) == []
    skipped = out.metadata["server_config_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["service"] == "slack"
    assert skipped[0]["uri"] == _SET_ACTING_USER_URI
    assert skipped[0]["reason"] == "extension_not_advertised"


@pytest.mark.asyncio
async def test_every_diagnostic_entry_carries_its_environment_twin():
    """The additive dual-write covers the diagnostics this step appends to task_instances.

    Nothing else asserts it: every surrounding test reads ``entry["service"]``, so
    deleting the twin from any of the three append sites leaves the suite green while
    silently reverting the migration for the one collection the destructive pass's
    backfill gate cannot re-derive from anywhere else.

    Drives all three sites in one run — applied, extension_not_advertised, and
    no_env_card — so a fourth append site inherits the check instead of needing its
    own test."""
    gw = _MockGateway({
        "slack": _card(_ext(_SET_ERRORS_URI, "/agentenv/ext/set_errors")),
        "email": _card(_ext(_SET_ACTING_USER_URI, "/agentenv/ext/set_acting_user")),
    })
    ctx = _make_context()
    step = _step(
        tolerate_unadvertised=True,
        directives=[
            {"service": "slack", "uri": _SET_ERRORS_URI, "args": {"tool_name": "t"}},
            {"service": "email", "uri": _SET_ERRORS_URI, "args": {"tool_name": "t"}},
            {"service": "ghost", "uri": _SET_ERRORS_URI, "args": {"tool_name": "t"}},
        ],
    )
    with _patch_httpx(gw):
        out = await step.execute(ctx)

    changes = out.metadata["server_config_changes"]
    skipped = out.metadata["server_config_skipped"]
    assert {s["reason"] for s in skipped} == {"extension_not_advertised", "no_env_card"}

    entries = changes + skipped
    assert len(entries) == 3
    for e in entries:
        assert e["environment"] == e["service"], f"twin missing or diverged: {e}"


@pytest.mark.asyncio
async def test_tolerate_unadvertised_applies_where_advertised_skips_where_not():
    # Fan-out across two services: one advertises set_acting_user (applied), the
    # other does not (skipped). Partial opt-in must not fail the step.
    gw = _MockGateway({
        "slack": _card(_ext(_SET_ACTING_USER_URI, "/agentenv/ext/set_acting_user")),
        "email": _card(_ext(_SET_ERRORS_URI, "/agentenv/ext/set_errors")),
    })
    ctx = _make_context()
    step = _step(
        tolerate_unadvertised=True,
        directives=[
            {"service": "slack", "uri": _SET_ACTING_USER_URI, "args": {"user_email": "a@x.com"}},
            {"service": "email", "uri": _SET_ACTING_USER_URI, "args": {"user_email": "a@x.com"}},
        ],
    )
    with _patch_httpx(gw):
        out = await step.execute(ctx)

    changes = out.metadata["server_config_changes"]
    assert len(changes) == 1 and changes[0]["service"] == "slack"
    assert len(gw.post_calls) == 1
    assert gw.post_calls[0][0] == f"{_GATEWAY}/svc/mcp-slack/agentenv/ext/set_acting_user"
    skipped = out.metadata["server_config_skipped"]
    assert len(skipped) == 1 and skipped[0]["service"] == "email"


@pytest.mark.asyncio
async def test_tolerate_unadvertised_still_fails_on_rejected_args():
    # tolerate_unadvertised only tolerates a missing extension. A service that
    # DOES advertise but rejects the args (e.g. unresolvable persona) still
    # fails loud — that's the fail-loud-on-unknown-persona contract.
    gw = _MockGateway({"slack": _card(_ext(_SET_ACTING_USER_URI, "/agentenv/ext/set_acting_user"))})

    async def rejecting_request(method, url, *, json=None, timeout=None):
        gw.post_calls.append((url, json))
        return _resp(500, url, {"error": "user_not_found: nobody@x.com"})

    gw.request = rejecting_request
    with _patch_httpx(gw):
        with pytest.raises(ApplyServerConfigError, match="HTTP 500"):
            await _step(
                tolerate_unadvertised=True,
                directives=[
                    {"service": "slack", "uri": _SET_ACTING_USER_URI, "args": {"user_email": "nobody@x.com"}},
                ],
            ).execute(_make_context())


def _fake_env(*environment_names: str):
    """A MultiEnv-shaped stand-in for Env.get: only .mcp_server_envs[].environment_name
    is read by _expand_directives."""
    return SimpleNamespace(
        mcp_server_envs=[SimpleNamespace(environment_name=s) for s in environment_names]
    )


@pytest.mark.asyncio
async def test_broadcast_expands_to_every_env_service():
    # A single service="*" directive fans out to one directive per MCP server
    # in the env (resolved from Env.get), all carrying the same args.
    gw = _MockGateway({
        "slack": _card(_ext(_SET_ACTING_USER_URI, "/agentenv/ext/set_acting_user")),
        "email": _card(_ext(_SET_ACTING_USER_URI, "/agentenv/ext/set_acting_user")),
    })
    step = _step(directives=[
        {"service": "*", "uri": _SET_ACTING_USER_URI, "args": {"user_email": "a@x.com"}},
    ])
    with _patch_httpx(gw), patch(
        "agent_env.env.env.Env.get", return_value=_fake_env("slack", "email")
    ):
        out = await step.execute(_make_context())

    assert {c["service"] for c in out.metadata["server_config_changes"]} == {"slack", "email"}
    assert len(gw.post_calls) == 2
    assert {u for u, _ in gw.post_calls} == {
        f"{_GATEWAY}/svc/mcp-slack/agentenv/ext/set_acting_user",
        f"{_GATEWAY}/svc/mcp-email/agentenv/ext/set_acting_user",
    }


@pytest.mark.asyncio
async def test_broadcast_env_get_failure_wraps_as_step_error():
    # Env.get failing during broadcast expansion must surface as
    # ApplyServerConfigError, same contract as the per-directive HTTP calls.
    gw = _MockGateway({})
    step = _step(directives=[
        {"service": "*", "uri": _SET_ACTING_USER_URI, "args": {"user_email": "a@x.com"}},
    ])

    def _boom(*a, **k):
        raise RuntimeError("mongo unreachable")

    with _patch_httpx(gw), patch("agent_env.env.env.Env.get", side_effect=_boom):
        with pytest.raises(ApplyServerConfigError, match="expand broadcast directives"):
            await step.execute(_make_context())


@pytest.mark.asyncio
async def test_broadcast_with_tolerate_applies_and_skips_per_service():
    # Broadcast set_acting_user across an env where only one server opts in.
    gw = _MockGateway({
        "slack": _card(_ext(_SET_ACTING_USER_URI, "/agentenv/ext/set_acting_user")),
        "email": _card(_ext(_SET_ERRORS_URI, "/agentenv/ext/set_errors")),  # no acting-user
    })
    step = _step(
        tolerate_unadvertised=True,
        directives=[
            {"service": "*", "uri": _SET_ACTING_USER_URI, "args": {"user_email": "a@x.com"}},
        ],
    )
    with _patch_httpx(gw), patch(
        "agent_env.env.env.Env.get", return_value=_fake_env("slack", "email")
    ):
        out = await step.execute(_make_context())

    assert {c["service"] for c in out.metadata["server_config_changes"]} == {"slack"}
    assert {s["service"] for s in out.metadata["server_config_skipped"]} == {"email"}


@pytest.mark.asyncio
async def test_broadcast_with_tolerate_skips_service_without_card():
    # The incident repro: broadcast set_acting_user across an env where one
    # server serves a card and another (a pre-card image) serves none (404).
    # The card-less server is skipped, not fatal.
    gw = _MockGateway({
        "oms": _card(_ext(_SET_ACTING_USER_URI, "/agentenv/ext/set_acting_user")),
        # "blackline" has no card entry -> the gateway returns 404
    })
    step = _step(
        tolerate_unadvertised=True,
        directives=[
            {"service": "*", "uri": _SET_ACTING_USER_URI, "args": {"user_email": "a@x.com"}},
        ],
    )
    with _patch_httpx(gw), patch(
        "agent_env.env.env.Env.get", return_value=_fake_env("oms", "blackline")
    ):
        out = await step.execute(_make_context())

    assert {c["service"] for c in out.metadata["server_config_changes"]} == {"oms"}
    skipped = out.metadata["server_config_skipped"]
    assert {s["service"] for s in skipped} == {"blackline"}
    assert skipped[0]["reason"] == "no_env_card"


@pytest.mark.asyncio
async def test_broadcast_expands_to_single_server_env_via_service_name():
    # A single-server env (a bare MCPServerEnv) exposes environment_name and no
    # mcp_server_envs; broadcast must still fan out to that one service rather
    # than silently matching nothing.
    gw = _MockGateway({
        "slack": _card(_ext(_SET_ACTING_USER_URI, "/agentenv/ext/set_acting_user")),
    })
    step = _step(directives=[
        {"service": "*", "uri": _SET_ACTING_USER_URI, "args": {"user_email": "a@x.com"}},
    ])
    with _patch_httpx(gw), patch(
        "agent_env.env.env.Env.get",
        return_value=SimpleNamespace(environment_name="slack"),
    ):
        out = await step.execute(_make_context())

    assert {c["service"] for c in out.metadata["server_config_changes"]} == {"slack"}
    assert len(gw.post_calls) == 1


@pytest.mark.asyncio
async def test_broadcast_zero_services_raises():
    # A broadcast that resolves to no services must fail loud, not silently
    # succeed. Even tolerate_unadvertised (about advertisement) doesn't excuse
    # an env with no MCP servers to apply to.
    gw = _MockGateway({})
    step = _step(
        tolerate_unadvertised=True,
        directives=[
            {"service": "*", "uri": _SET_ACTING_USER_URI, "args": {"user_email": "a@x.com"}},
        ],
    )
    with _patch_httpx(gw), patch(
        "agent_env.env.env.Env.get", return_value=SimpleNamespace()
    ):
        with pytest.raises(ApplyServerConfigError, match="matched no MCP servers"):
            await step.execute(_make_context())


@pytest.mark.asyncio
async def test_tolerate_unadvertised_skips_service_without_card():
    # A server that serves no card at all (404 on the well-known path) predates
    # env-card support — the same "hasn't opted in" case as a missing extension.
    # Under tolerate_unadvertised it is skipped + recorded, not raised.
    gw = _MockGateway({})  # no card for any service -> 404
    ctx = _make_context()
    step = _step(
        tolerate_unadvertised=True,
        directives=[
            {"service": "slack", "uri": _SET_ACTING_USER_URI, "args": {"user_email": "a@x.com"}},
        ],
    )
    with _patch_httpx(gw):
        out = await step.execute(ctx)

    assert gw.post_calls == []  # nothing invoked
    assert out.metadata.get("server_config_changes", []) == []
    skipped = out.metadata["server_config_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["service"] == "slack"
    assert skipped[0]["uri"] == _SET_ACTING_USER_URI
    assert skipped[0]["reason"] == "no_env_card"


@pytest.mark.asyncio
async def test_non_404_card_fetch_still_raises_under_tolerate():
    # tolerate_unadvertised excuses only a genuinely absent card (404). A card
    # endpoint that errors (500) is a real fault and must still fail loud.
    gw = _MockGateway({})

    async def failing_get(url, *, timeout=None, params=None):
        gw.get_calls.append(url)
        return _resp(500, url, {"error": "boom"})

    gw.get = failing_get
    step = _step(
        tolerate_unadvertised=True,
        directives=[
            {"service": "slack", "uri": _SET_ACTING_USER_URI, "args": {"user_email": "a@x.com"}},
        ],
    )
    with _patch_httpx(gw):
        with pytest.raises(ApplyServerConfigError, match="HTTP 500"):
            await step.execute(_make_context())


@pytest.mark.asyncio
async def test_server_rejects_args_raises_with_status():
    gw = _MockGateway({"slack": _card(_ext(_SET_ERRORS_URI, "/agentenv/ext/set_errors"))})

    async def rejecting_request(method, url, *, json=None, timeout=None):
        gw.post_calls.append((url, json))
        return _resp(400, url, {"error": "error_rate must be between 0.0 and 1.0"})

    gw.request = rejecting_request
    with _patch_httpx(gw):
        with pytest.raises(ApplyServerConfigError, match="HTTP 400"):
            await _step(directives=[
                {"service": "slack", "uri": _SET_ERRORS_URI, "args": {"tool_name": "x", "error_rate": 9}},
            ]).execute(_make_context())


@pytest.mark.asyncio
async def test_partial_failure_still_records_applied_directives():
    # First directive advertised (succeeds + mutates the server), second not
    # advertised (raises). The audit trail must reflect the first so a partially
    # armed server is never invisible in context.metadata.
    gw = _MockGateway({"slack": _card(_ext(_SET_ERRORS_URI, "/agentenv/ext/set_errors"))})
    ctx = _make_context()
    step = _step(directives=[
        {"service": "slack", "uri": _SET_ERRORS_URI, "args": {"tool_name": "a", "error_rate": 1.0}},
        {"service": "slack", "uri": _SET_ASYNC_WAIT_URI, "args": {"tool_name": "b", "min_seconds": 5}},
    ])
    with _patch_httpx(gw):
        with pytest.raises(ApplyServerConfigError, match="not advertised"):
            await step.execute(ctx)
    changes = ctx.metadata.get("server_config_changes", [])
    assert len(changes) == 1
    assert changes[0]["uri"] == _SET_ERRORS_URI
    assert changes[0]["result"]["ok"] is True


@pytest.mark.asyncio
async def test_missing_env_is_clear_error():
    gw = _MockGateway({})
    with _patch_httpx(gw):
        with pytest.raises(ApplyServerConfigError, match="not found in context.deployed_envs"):
            await _step(env_id="ghost").execute(_make_context())


@pytest.mark.asyncio
async def test_card_404_raises():
    gw = _MockGateway({})  # no card for any service → 404
    with _patch_httpx(gw):
        with pytest.raises(ApplyServerConfigError):
            await _step().execute(_make_context())


# --- schema / validation (pure, no I/O) ---


def test_config_round_trips_through_to_dict_from_dict():
    step = _step(
        timeout_seconds=45,
        tolerate_unadvertised=True,
        depends_on=[{"task_step_id": "deploy-env-1"}],
        directives=[
            {"service": "slack", "uri": _SET_ERRORS_URI, "args": {"tool_name": "t", "error_rate": 0.3}},
            {"service": "email", "uri": _SET_ASYNC_WAIT_URI, "args": {"tool_name": "send", "min_seconds": 2}},
        ],
    )
    d = step.to_dict()
    assert d["type"] == "apply_server_config"
    assert d["env_id"] == "env-x"
    assert d["timeout_seconds"] == 45
    assert d["tolerate_unadvertised"] is True
    assert len(d["directives"]) == 2
    assert ApplyServerConfigStep.from_dict(d).to_dict() == d
    # JSON-serializable (it gets persisted to Mongo + sent through Temporal).
    assert json.loads(json.dumps(d)) == d


def test_empty_directives_rejected():
    with pytest.raises(ValueError, match="directives must be a non-empty list"):
        _step(directives=[])


def test_missing_env_id_rejected():
    with pytest.raises(ValueError, match="env_id must be a non-empty string"):
        _step(env_id="")


def test_directive_validation():
    with pytest.raises(ValueError, match="directive.service"):
        ConfigDirective(service="", uri=_SET_ERRORS_URI, args={})
    with pytest.raises(ValueError, match="directive.uri"):
        ConfigDirective(service="slack", uri="", args={})
    with pytest.raises(ValueError, match="directive.args must be a dict"):
        ConfigDirective(service="slack", uri=_SET_ERRORS_URI, args=["not", "a", "dict"])


# --- Read side / write side: reads either spelling, writes both ---


def test_directive_from_dict_reads_service_key():
    """The frozen spelling: every delivered bundle and stored task on disk says "service"."""
    d = ConfigDirective.from_dict({"service": "slack", "uri": _SET_ERRORS_URI, "args": {"tool_name": "t"}})
    assert d.service == "slack"
    assert d.uri == _SET_ERRORS_URI
    assert d.args == {"tool_name": "t"}


def test_directive_from_dict_reads_environment_key():
    d = ConfigDirective.from_dict({"environment": "slack", "uri": _SET_ERRORS_URI, "args": {"tool_name": "t"}})
    assert d.service == "slack"
    assert d.to_dict() == {
        "service": "slack", "environment": "slack", "uri": _SET_ERRORS_URI, "args": {"tool_name": "t"},
    }


def test_directive_from_dict_prefers_service_when_both_present():
    d = ConfigDirective.from_dict({"service": "slack", "environment": "email", "uri": _SET_ERRORS_URI})
    assert d.service == "slack"


def test_directive_from_dict_requires_one_of_the_two_keys():
    with pytest.raises(ValueError, match="requires a 'service' \\(or 'environment'\\) key"):
        ConfigDirective.from_dict({"uri": _SET_ERRORS_URI, "args": {}})


def test_directive_from_dict_present_but_empty_service_does_not_fall_through():
    """Presence-based, not truthiness: an explicit empty "service" is malformed input and must
    still raise, exactly as it did before the rename — it must NOT silently resolve to "environment".
    Matches the presence-based dual-reads in mcp_server.from_dict / website.from_dict."""
    with pytest.raises(ValueError, match="directive.service must be a non-empty string"):
        ConfigDirective.from_dict({"service": "", "environment": "slack", "uri": _SET_ERRORS_URI})


def test_directive_from_dict_still_validates_the_resolved_name():
    with pytest.raises(ValueError, match="directive.service must be a non-empty string"):
        ConfigDirective.from_dict({"environment": 7, "uri": _SET_ERRORS_URI})
    with pytest.raises(ValueError, match="directive.uri"):
        ConfigDirective.from_dict({"environment": "slack", "uri": ""})


def test_directive_to_dict_always_emits_both_keys():
    """``to_dict`` writes the `tasks` / `task_steps` documents, so the additive
    dual-write covers it: a task authored today must still load once the destructive
    pass drops the legacy key, and the backfill gates on every directive having both.

    This is NOT the frozen bundle surface. The frozen surface is the *Harbor applier
    input*, which the plugin-hosted Harbor exporter builds from the
    attributes directly and never routes through here — see the export
    producer test."""
    for key in ("service", "environment"):
        d = ConfigDirective.from_dict({key: "slack", "uri": _SET_ERRORS_URI, "args": {}}).to_dict()
        assert d["service"] == "slack"
        assert d["environment"] == "slack"


def test_step_with_environment_directives_serializes_to_both_keys():
    step = _step(directives=[{"environment": "slack", "uri": _SET_ERRORS_URI, "args": {"tool_name": "t"}}])
    d = step.to_dict()
    assert d["directives"] == [
        {"service": "slack", "environment": "slack", "uri": _SET_ERRORS_URI, "args": {"tool_name": "t"}},
    ]
    assert ApplyServerConfigStep.from_dict(d).to_dict() == d
