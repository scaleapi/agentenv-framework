"""The one caller value: its binding, the `_meta` stamp it produces, and the wire names it shares with the protocol."""
from __future__ import annotations

import asyncio
import contextvars

import agentenv_protocol
import pytest

from agent_env.env.gateway.caller import Caller
from agent_env.env.gateway.constants import (
    AGENT_ENV_ROLE_HEADER,
    AGENT_ENV_ROLE_META_KEY,
    AGENT_ENV_SESSION_META_KEY,
    DEFAULT_ROLE,
)


def test_the_wire_names_are_the_protocols():
    """The gateway container ships without agentenv_protocol, so these are literals; they must stay the protocol's."""
    assert (AGENT_ENV_ROLE_HEADER, AGENT_ENV_ROLE_META_KEY, AGENT_ENV_SESSION_META_KEY, DEFAULT_ROLE) == (
        agentenv_protocol.ROLE_HEADER, agentenv_protocol.ROLE_META_KEY, agentenv_protocol.SESSION_META_KEY,
        agentenv_protocol.DEFAULT_ROLE)


def test_an_unforwarded_caller_is_stamped_default_with_no_session():
    assert Caller.current() == Caller(DEFAULT_ROLE, "")
    assert Caller().meta() == {AGENT_ENV_ROLE_META_KEY: DEFAULT_ROLE}, "the role entry is what marks a call as proxied"


def test_the_session_rides_along_only_when_there_is_one():
    assert Caller("alice@example.com").meta() == {AGENT_ENV_ROLE_META_KEY: "alice@example.com"}
    assert Caller("alice@example.com", "s1").meta() == {
        AGENT_ENV_ROLE_META_KEY: "alice@example.com", AGENT_ENV_SESSION_META_KEY: "s1"}


def test_binding_is_per_context_and_restorable():
    token = Caller("bob@example.com", "s2").bind()
    try:
        assert Caller.current() == Caller("bob@example.com", "s2")
        inner = contextvars.copy_context()
        inner.run(Caller("carol@example.com").bind)
        assert inner.run(Caller.current) == Caller("carol@example.com")
        assert Caller.current() == Caller("bob@example.com", "s2"), "a copied context binds on its own"
    finally:
        Caller.reset(token)
    assert Caller.current() == Caller()


@pytest.mark.asyncio
async def test_a_task_inherits_the_caller_bound_when_it_was_spawned():
    token = Caller("dave@example.com", "s3").bind()
    try:
        seen = await asyncio.create_task(_current_in_task())
    finally:
        Caller.reset(token)
    assert seen == Caller("dave@example.com", "s3")


async def _current_in_task() -> Caller:
    return Caller.current()
