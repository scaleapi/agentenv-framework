"""The gateway's roles (urn:agentenv:roles/v1): roles declared none or token, members kept as their agent token's
digest, and the one rule a role's name follows."""
from __future__ import annotations

import httpx
import pytest

from agent_env.env.gateway.gateway import Gateway
from agent_env.env.gateway.roles import Computer, Resolution, RoleError, Roles, digest, role_name

TOKEN = "agent-token-of-brooke"
OTHER = "agent-token-of-denise"


def test_create_is_idempotent_and_a_conflicting_create_names_the_existing_auth():
    roles = Roles()
    first = roles.create("brooke@example.com", {"type": "token"})
    again = roles.create("brooke@example.com", {"type": "token"})
    with pytest.raises(RoleError) as conflict:
        roles.create("brooke@example.com", {"type": "none"})

    assert first == {"role": "brooke@example.com", "auth": {"type": "token"}, "created": True}
    assert again == {"role": "brooke@example.com", "auth": {"type": "token"}, "created": False}
    assert (conflict.value.status, conflict.value.fields) == (409, {"role": "brooke@example.com",
                                                                   "auth": {"type": "token"}})


def test_a_role_created_without_auth_is_open():
    roles = Roles()
    assert roles.create("denise@example.com")["auth"] == {"type": "none"}
    assert roles.auth_of("denise@example.com") == roles.auth_of("never-declared") == "none"


def test_adding_a_member_twice_gives_the_same_id_and_the_answer_carries_the_roles_auth():
    roles = Roles()
    roles.create("brooke@example.com", {"type": "token"})
    first = roles.add_member("brooke@example.com", TOKEN)
    again = roles.add_member("brooke@example.com", TOKEN)
    other = roles.add_member("brooke@example.com", OTHER)

    assert first == again and first["auth"] == {"type": "token"}
    assert first["member_id"].startswith("mem_") and other["member_id"] != first["member_id"]


def test_a_member_add_on_an_undeclared_role_is_a_404():
    with pytest.raises(RoleError) as undeclared:
        Roles().add_member("brooke@example.com", TOKEN)
    assert undeclared.value.status == 404


def test_delete_makes_the_role_undeclared_again_and_forgets_its_members():
    roles = Roles()
    roles.create("brooke@example.com", {"type": "token"})
    roles.add_member("brooke@example.com", TOKEN)

    assert roles.delete("brooke@example.com") == {"role": "brooke@example.com", "deleted": True}
    assert roles.delete("brooke@example.com") == {"role": "brooke@example.com", "deleted": False}
    assert roles.auth_of("brooke@example.com") == "none" and roles.state() == {"roles": []}
    assert roles.roles_of(digest(TOKEN)) == frozenset()
    assert roles.create("brooke@example.com", {"type": "none"})["created"]


def test_removing_a_member_keeps_the_role_and_its_other_members():
    roles = Roles()
    roles.create("brooke@example.com", {"type": "token"})
    brooke = roles.add_member("brooke@example.com", TOKEN)["member_id"]
    denise = roles.add_member("brooke@example.com", OTHER)["member_id"]

    assert roles.remove_member("brooke@example.com", brooke) == {"role": "brooke@example.com", "member_id": brooke,
                                                                "removed": True}
    assert roles.remove_member("brooke@example.com", brooke)["removed"] is False
    assert roles.remove_member("never-declared", brooke)["removed"] is False
    assert roles.state()["roles"] == [{"role": "brooke@example.com", "auth": {"type": "token"}, "members": [denise],
                                       "startup": False}]
    assert roles.roles_of(digest(TOKEN)) == frozenset()
    assert roles.roles_of(digest(OTHER)) == {"brooke@example.com"}


def test_an_agent_in_two_roles_is_indexed_under_both():
    roles = Roles()
    roles.create("brooke@example.com", {"type": "token"})
    roles.create("denise@example.com")
    roles.add_member("brooke@example.com", TOKEN)
    roles.add_member("denise@example.com", TOKEN)
    assert roles.roles_of(digest(TOKEN)) == {"brooke@example.com", "denise@example.com"}


def test_a_role_is_the_same_role_however_its_name_is_cased():
    """The services behind the gateway match a persona's address case-insensitively, so the gateway does too."""
    roles = Roles()
    roles.create("brooke@example.com", {"type": "token"})
    member = roles.add_member("BROOKE@example.com", TOKEN)["member_id"]
    with pytest.raises(RoleError) as respelled:
        roles.create("Brooke@Example.com", {"type": "token"})

    assert (respelled.value.status, respelled.value.fields) == (
        409, {"role": "brooke@example.com", "auth": {"type": "token"}})
    assert str(respelled.value) == ("role 'Brooke@Example.com' is already declared as 'brooke@example.com' with auth "
                                    "'token'")
    assert roles.auth_of("BROOKE@EXAMPLE.COM") == "token"
    assert roles.roles_of(digest(TOKEN)) == {"brooke@example.com"}
    assert roles.state()["roles"] == [{"role": "brooke@example.com", "auth": {"type": "token"}, "members": [member],
                                       "startup": False}]
    assert roles.remove_member("Brooke@example.com", member) == {"role": "brooke@example.com", "member_id": member,
                                                                "removed": True}
    assert roles.delete("BROOKE@EXAMPLE.COM") == {"role": "brooke@example.com", "deleted": True}
    assert roles.state() == {"roles": []}


def test_startup_roles_are_declared_and_marked():
    roles = Roles({"brooke@example.com": "token", "default": "none"})
    assert roles.state() == {"roles": [
        {"role": "brooke@example.com", "auth": {"type": "token"}, "members": [], "startup": True},
        {"role": "default", "auth": {"type": "none"}, "members": [], "startup": True}]}


def test_state_and_every_refusal_carry_no_token_or_digest():
    roles = Roles()
    roles.create("brooke@example.com", {"type": "token"})
    member = roles.add_member("brooke@example.com", TOKEN)["member_id"]
    errors = []
    for attempt in (lambda: roles.create("brooke@example.com", {"type": "none"}),
                    lambda: roles.add_member("denise@example.com", TOKEN),
                    lambda: roles.add_member("brooke@example.com", ""),
                    lambda: roles.add_member(TOKEN * 10, TOKEN),
                    lambda: roles.create(f"{TOKEN}\t{TOKEN}", {"type": "token"}),
                    lambda: roles.create("brooke@example.com", {"type": TOKEN})):
        with pytest.raises(RoleError) as refused:
            attempt()
        errors.append(f"{refused.value} {refused.value.fields}")

    text = " ".join([str(roles.state()), *errors])
    assert TOKEN not in text and digest(TOKEN) not in text and digest(TOKEN)[:12] not in text
    assert member in text


@pytest.mark.parametrize("value, name", [("brooke@example.com", "brooke@example.com"), ("  default ", "default"),
                                         ("x" * 100, "x" * 100)])
def test_a_role_name_is_stripped(value, name):
    assert role_name(value) == name


@pytest.mark.parametrize("value", ["", "   ", "*", " * ", "x" * 101, "bro\nke", "bröoke", "tab\there", None, 7, ["a"]],
                         ids=["empty", "blank", "wildcard", "padded-wildcard", "too-long", "control", "non-ascii",
                              "tab", "none", "int", "list"])
def test_a_role_name_is_printable_ascii_of_at_most_100_characters_and_not_the_wildcard(value):
    with pytest.raises(RoleError, match="must name a role"):
        role_name(value)


@pytest.mark.parametrize("auth", [{"type": "password"}, {"type": "token", "extra": 1}, "token", {}, ["token"]],
                         ids=["unknown-type", "extra-field", "bare-string", "empty", "list"])
def test_auth_is_a_typed_object_naming_none_or_token(auth):
    with pytest.raises(RoleError, match="auth must be"):
        Roles().create("brooke@example.com", auth)


@pytest.mark.parametrize("token", ["", "  ", None, 7], ids=["empty", "blank", "none", "int"])
def test_a_member_is_added_by_a_non_empty_agent_token(token):
    roles = Roles()
    roles.create("brooke@example.com", {"type": "token"})
    with pytest.raises(RoleError, match="agent_token must be"):
        roles.add_member("brooke@example.com", token)


# ---------------------------------------------------------------- the routes

CONTROL = "s3cret"
AUTHORIZED = {"Authorization": f"Bearer {CONTROL}"}


def _client(control_token: str | None = CONTROL) -> httpx.AsyncClient:
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[], control_token=control_token)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw._asgi_app()), base_url="http://gateway",
                             headers=AUTHORIZED if control_token else {})


@pytest.mark.asyncio
async def test_roles_are_created_joined_listed_left_and_deleted_at_fixed_paths():
    async with _client() as client:
        created = await client.post("/roles", json={"role": "brooke@example.com", "auth": {"type": "token"}})
        again = await client.post("/roles", json={"role": "brooke@example.com", "auth": {"type": "token"}})
        conflict = await client.post("/roles", json={"role": "brooke@example.com"})
        joined = await client.post("/roles/members", json={"role": "brooke@example.com", "agent_token": TOKEN})
        rejoined = await client.post("/roles/members", json={"role": "brooke@example.com", "agent_token": TOKEN})
        undeclared = await client.post("/roles/members", json={"role": "denise@example.com", "agent_token": TOKEN})
        listed = await client.get("/roles")
        left = await client.request("DELETE", "/roles/members",
                                    json={"role": "brooke@example.com", "member_id": joined.json()["member_id"]})
        deleted = await client.request("DELETE", "/roles", json={"role": "brooke@example.com"})
        after = await client.get("/roles")

    assert (created.status_code, created.json()["created"], again.status_code, again.json()["created"]) == (
        200, True, 200, False)
    assert conflict.status_code == 409 and conflict.json()["auth"] == {"type": "token"}
    assert joined.status_code == 200 and joined.json() == rejoined.json()
    assert joined.json() == {"member_id": joined.json()["member_id"], "auth": {"type": "token"}}
    assert undeclared.status_code == 404
    assert listed.json() == {"roles": [{"role": "brooke@example.com", "auth": {"type": "token"},
                                        "members": [joined.json()["member_id"]], "startup": False}]}
    assert TOKEN not in listed.text and digest(TOKEN) not in listed.text
    assert (left.status_code, left.json()["removed"]) == (200, True)
    assert (deleted.status_code, deleted.json()["deleted"]) == (200, True)
    assert after.json() == {"roles": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", [("POST", "/roles"), ("DELETE", "/roles"), ("POST", "/roles/members"),
                                          ("DELETE", "/roles/members")])
@pytest.mark.parametrize("body", [b'{"role": "brooke", "agent_token": "' + TOKEN.encode() + b'"', b"[1, 2]"],
                         ids=["not-json", "not-an-object"])
async def test_a_bad_body_is_a_400_that_never_echoes_it(method, path, body):
    async with _client() as client:
        response = await client.request(method, path, content=body, headers={"content-type": "application/json"})
    assert response.status_code == 400 and TOKEN not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", [("POST", "/roles"), ("GET", "/roles"), ("DELETE", "/roles"),
                                          ("POST", "/roles/members"), ("DELETE", "/roles/members")])
async def test_a_gateway_without_a_control_token_serves_no_roles(method, path):
    async with _client(None) as client:
        response = await client.request(method, path, json={"role": "brooke@example.com", "agent_token": TOKEN})
    assert response.status_code in (404, 405)


# ---------------------------------------------------------------- resolution, one case per row of the design's table

def _office() -> Roles:
    """brooke is token-auth with one member, denise is declared open, and the member also belongs to denise."""
    roles = Roles()
    roles.create("brooke@example.com", {"type": "token"})
    roles.create("denise@example.com", {"type": "none"})
    roles.add_member("brooke@example.com", TOKEN)
    return roles


def test_agent_token_and_role_header_a_member_plays_the_named_role():
    resolved = _office().resolve("brooke@example.com", digest(TOKEN))
    assert (resolved.role, resolved.source, resolved.refusal, resolved.member) == (
        "brooke@example.com", "agent_token", None, digest(TOKEN))


def test_agent_token_and_role_header_a_non_member_is_refused_a_token_role():
    roles = _office()
    roles.add_member("denise@example.com", OTHER)
    resolved = roles.resolve("brooke@example.com", digest(OTHER))
    assert (resolved.role, resolved.source) == ("brooke@example.com", "header")
    assert resolved.refusal == "this agent token isn't a member of role 'brooke@example.com'"


def test_agent_token_and_role_header_any_agent_plays_an_open_role_through_the_header():
    resolved = _office().resolve("denise@example.com", digest(TOKEN))
    assert (resolved.role, resolved.source, resolved.refusal) == ("denise@example.com", "header", None)


def test_agent_token_without_a_role_header_resolves_to_the_agents_only_role():
    resolved = _office().resolve("", digest(TOKEN))
    assert (resolved.role, resolved.source, resolved.refusal) == ("brooke@example.com", "agent_token", None)


def test_agent_token_without_a_role_header_resolves_to_default_for_an_agent_in_several_roles():
    roles = _office()
    roles.add_member("denise@example.com", TOKEN)
    resolved = roles.resolve("", digest(TOKEN))
    assert (resolved.role, resolved.source, resolved.refusal) == ("default", "default", None)
    roles.create("default", {"type": "token"})
    assert roles.resolve("", digest(TOKEN)).refusal == "this agent token isn't a member of role 'default'"
    roles.add_member("default", TOKEN)
    assert roles.resolve("", digest(TOKEN)).refusal is None


def test_an_agent_in_two_roles_plays_either_chosen_by_the_header():
    roles = _office()
    roles.create("carol@example.com", {"type": "token"})
    roles.add_member("carol@example.com", TOKEN)
    for role in ("brooke@example.com", "carol@example.com"):
        resolved = roles.resolve(role, digest(TOKEN))
        assert (resolved.role, resolved.source, resolved.refusal) == (role, "agent_token", None)


@pytest.mark.parametrize("header, role", [("brooke@example.com", "brooke@example.com"), ("", "default")])
def test_the_control_token_plays_any_role_and_without_a_header_default(header, role):
    roles = _office()
    roles.create("default", {"type": "token"})
    resolved = roles.resolve(header, control=True)
    assert (resolved.role, resolved.source, resolved.refusal) == (role, "control_token", None)


def test_a_plain_header_is_refused_a_token_role_and_plays_an_open_or_undeclared_one():
    roles = _office()
    refused = roles.resolve("brooke@example.com")
    assert (refused.role, refused.source, refused.member) == ("brooke@example.com", "header", None)
    assert refused.refusal == "role 'brooke@example.com' takes the agent token of one of its members"
    for role in ("denise@example.com", "never-declared"):
        resolved = roles.resolve(role)
        assert (resolved.role, resolved.source, resolved.refusal) == (role, "header", None)


@pytest.mark.parametrize("header", ["BROOKE@EXAMPLE.COM", "Brooke@Example.com"])
def test_any_case_of_a_token_roles_name_takes_a_members_token(header):
    roles = _office()
    refused = roles.resolve(header)
    played = roles.resolve(header, digest(TOKEN))
    assert (refused.role, refused.source) == (header, "header")
    assert refused.refusal == f"role {header!r} takes the agent token of one of its members"
    assert (played.role, played.source, played.refusal) == (header, "agent_token", None)
    assert not roles.may_play(header, "header") and roles.may_play(header, "agent_token", digest(TOKEN))


def test_a_name_that_case_folds_to_a_token_roles_name_is_that_role():
    """A header decodes as latin-1, so it can hold a character such as 'ß', which folds to 'ss'."""
    roles = Roles()
    roles.create("strasse@example.com", {"type": "token"})
    assert roles.resolve("STRAßE@example.com").refusal == (
        "role 'STRAßE@example.com' takes the agent token of one of its members")


def test_an_unrecognised_agent_token_is_treated_as_no_token_but_its_digest_is_kept():
    """Kept so a session that presented it plays its role once the token becomes a member."""
    roles = _office()
    unknown = digest("never-a-member")
    for header in ("brooke@example.com", "denise@example.com", ""):
        resolved, without = roles.resolve(header, unknown), roles.resolve(header)
        assert (resolved.role, resolved.source, resolved.refusal) == (without.role, without.source, without.refusal)
        assert resolved.member == unknown
    roles.add_member("brooke@example.com", "never-a-member")
    assert roles.may_play("brooke@example.com", "header", roles.resolve("brooke@example.com", unknown).member)


def test_a_computer_with_its_token_takes_its_driver_and_is_never_refused():
    roles = _office()
    roles.create("default", {"type": "token"})
    driven = roles.resolve("", computer=Computer("brooke@example.com", verified=True))
    idle = roles.resolve("", computer=Computer(None, verified=True))
    assert (driven.role, driven.source, driven.refusal) == ("brooke@example.com", "computer_token", None)
    assert (idle.role, idle.source, idle.refusal) == ("default", "computer_token", None)


def test_a_computer_without_its_token_claims_only_a_driver_whose_role_is_open():
    roles = _office()
    open_driver = roles.resolve("", computer=Computer("denise@example.com", verified=False))
    protected_driver = roles.resolve("", computer=Computer("brooke@example.com", verified=False))
    assert (open_driver.role, open_driver.source, open_driver.refusal) == ("denise@example.com", "header", None)
    assert protected_driver == roles.resolve("")


def test_a_role_header_beats_a_computer():
    resolved = _office().resolve("denise@example.com", computer=Computer("brooke@example.com", verified=True))
    assert (resolved.role, resolved.source) == ("denise@example.com", "header")


def test_nothing_resolves_to_default_refused_once_default_is_token_auth():
    roles = _office()
    assert roles.resolve("") == Resolution("default", "default", None, None)
    roles.create("default", {"type": "token"})
    assert roles.resolve("").refusal == "role 'default' takes the agent token of one of its members"


def test_may_play_follows_membership_as_it_changes():
    roles = Roles()
    roles.create("brooke@example.com")
    assert roles.may_play("brooke@example.com", "header", None)
    roles.delete("brooke@example.com")
    roles.create("brooke@example.com", {"type": "token"})
    assert not roles.may_play("brooke@example.com", "header", None)
    member = roles.add_member("brooke@example.com", TOKEN)["member_id"]
    assert roles.may_play("brooke@example.com", "agent_token", digest(TOKEN))
    roles.remove_member("brooke@example.com", member)
    assert not roles.may_play("brooke@example.com", "agent_token", digest(TOKEN))
    assert roles.may_play("brooke@example.com", "control_token", None)
    assert roles.may_play("brooke@example.com", "computer_token", None)


def test_source_now_follows_membership_as_it_changes():
    """A token that joins its session's role counts from then on, and one that leaves an open role counts as the
    header; the control and computer tokens stay as they were."""
    roles = Roles()
    roles.create("denise@example.com")
    assert roles.source_now("denise@example.com", "header", digest(TOKEN)) == "header"
    member = roles.add_member("denise@example.com", TOKEN)["member_id"]
    assert roles.source_now("denise@example.com", "header", digest(TOKEN)) == "agent_token"
    roles.remove_member("denise@example.com", member)
    assert roles.source_now("denise@example.com", "agent_token", digest(TOKEN)) == "header"
    assert roles.source_now("default", "default", digest(TOKEN)) == "default"
    roles.add_member("denise@example.com", TOKEN)
    for source in ("control_token", "computer_token"):
        assert roles.source_now("denise@example.com", source, digest(TOKEN)) == source


def test_a_resolution_never_shows_its_digest():
    assert digest(TOKEN) not in repr(_office().resolve("brooke@example.com", digest(TOKEN)))
