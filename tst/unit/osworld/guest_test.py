"""The guest server's address comes from the computer's card: the endpoint of the osworld-guest/v1 extension one of its
apps advertises, resolved against the computer's address, from the stored card or, while the app's server isn't
answering yet, the card the computer serves now; a computer without the app is refused."""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from agent_env.osworld import guest
from tst.unit.osworld.helpers import ENV_URL, card, computer


def test_the_endpoint_is_the_one_an_app_advertises_for_the_guest_extension():
    assert guest.guest_endpoint(card("/apps/7/osworld")) == "/apps/7/osworld"
    assert guest.guest_endpoint(card()["children_environments"][0]) == "/svc/osworld-server/osworld"
    assert guest.guest_endpoint(card(uri="urn:agentenv:other/v1")) is None
    assert guest.guest_endpoint(None) is None


@pytest.mark.asyncio
async def test_the_stored_card_is_read_first(monkeypatch):
    live = AsyncMock()
    monkeypatch.setattr(guest.protocol_v1, "get_card", live)
    assert await guest.resolve_guest_url(computer(card("/apps/7/osworld/"))) == f"{ENV_URL}/apps/7/osworld"
    live.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_live_card_is_read_again_until_the_app_answers(monkeypatch):
    bare = {"children_environments": [{"name": "osworld-server", "url": "/svc/osworld-server/agentenv"}]}
    live = AsyncMock(side_effect=[httpx.ConnectError("down"), bare, card("https://elsewhere.example/g/")])
    monkeypatch.setattr(guest.protocol_v1, "get_card", live)
    monkeypatch.setattr(guest.asyncio, "sleep", AsyncMock())
    assert await guest.resolve_guest_url(computer(bare), wait_s=60) == "https://elsewhere.example/g"
    assert live.await_count == 3


@pytest.mark.asyncio
async def test_a_computer_without_the_app_is_refused(monkeypatch):
    monkeypatch.setattr(guest.protocol_v1, "get_card", AsyncMock(return_value={}))
    with pytest.raises(RuntimeError, match="compose the osworld-server computer app"):
        await guest.resolve_guest_url(computer({}), wait_s=0)


@pytest.mark.asyncio
async def test_an_env_with_no_address_is_refused():
    env = computer()
    env.environment_card_url = None
    with pytest.raises(RuntimeError, match="no address"):
        await guest.resolve_guest_url(env)


def test_the_controller_url_is_the_one_recorded():
    assert guest.controller_url(computer(osworld_controller_url="https://c")) == "https://c"
    with pytest.raises(RuntimeError, match="run osworld_deploy_controller"):
        guest.controller_url(computer())
