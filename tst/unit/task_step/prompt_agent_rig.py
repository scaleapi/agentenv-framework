"""What tests of a prompt step against an agent served in process share: the conversation store kept quiet, every
HTTP client sent to the agent, and the agent the step prompts, deployed from its own card."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from agent_env.task_step.context import DeployedAgent
from agent_env.task_step.task_steps import prompt_agent as pa

#: The conversation store calls a prompt step makes, which these tests keep from reaching a store.
CONVERSATION_STORE_CALLS = ("create_conversation", "add_a2a_task", "complete_a2a_task", "mark_closed",
                            "get_conversation")
AGENT_URL = "http://agent.test"


def quiet_conversations(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in CONVERSATION_STORE_CALLS:
        monkeypatch.setattr(pa.conversation_store, name, lambda *args, **kwargs: None)


def route_clients(monkeypatch: pytest.MonkeyPatch, transport: httpx.AsyncBaseTransport, **options: Any) -> None:
    """Send every ``httpx.AsyncClient`` in this process through ``transport``, with ``options`` such as event hooks."""
    real_client = httpx._client.AsyncClient  # the class itself, so routing twice replaces rather than nests
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda *args, **kwargs: real_client(*args, **{**kwargs, "transport": transport, **options}))


def card_of(app: Any) -> dict:
    """An SDK agent app's card, as its well-known route serves it."""
    return app.state.agentenv_a2a.card.model_dump(mode="json", exclude_none=True)


def deployed(card: dict, *, name: str = "solver", url: str = AGENT_URL) -> DeployedAgent:
    return DeployedAgent(agent_name=name, api_url=url, a2a_url=url, a2a_card=card)
