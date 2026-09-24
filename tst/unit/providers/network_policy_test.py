"""NetworkPolicy value semantics — no provider imports, no I/O."""

import json

import pytest

from agent_env.providers.sandbox import NetworkMode, NetworkPolicy


def test_round_trip_preserves_mode_and_lists():
    policy = NetworkPolicy(
        mode=NetworkMode.ALLOWLIST,
        allow_hosts=("llm-proxy.example.com",),
        allow_cidrs=("10.0.0.0/8",),
    )
    wire = policy.to_dict()
    assert NetworkPolicy.from_dict(wire) == policy
    # bson cannot encode tuples or enum members.
    assert json.dumps(wire)
    assert wire["mode"] == "allowlist" and type(wire["mode"]) is str
    assert isinstance(wire["allow_hosts"], list)
    assert isinstance(wire["allow_cidrs"], list)


def test_from_dict_defaults_to_allow_all():
    assert NetworkPolicy.from_dict({}) == NetworkPolicy()
    assert NetworkPolicy().restricts_egress is False


@pytest.mark.parametrize("mode", ["no-network", "deny"])
def test_from_dict_rejects_unknown_mode(mode):
    """"deny" is deliberately included: it was a mode during review and is not one now,
    so a task carrying it must fail loudly rather than resolve to something else."""
    with pytest.raises(ValueError, match="Unknown network policy mode"):
        NetworkPolicy.from_dict({"mode": mode})


def test_string_mode_is_coerced_to_the_enum():
    # Constructs either way, since NetworkMode subclasses str.
    policy = NetworkPolicy(mode="allowlist")
    assert policy.mode is NetworkMode.ALLOWLIST
    assert policy.restricts_egress is True


def test_lists_are_coerced_to_tuples_so_the_policy_stays_hashable():
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=["a.com"], allow_cidrs=["10.0.0.0/8"])
    assert policy.allow_hosts == ("a.com",)
    assert hash(policy)


def test_with_hosts_unions_into_an_allowlist_order_stable_and_deduped():
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("a.com",))
    assert policy.with_hosts(["b.com", "a.com"]).allow_hosts == ("a.com", "b.com")


def test_with_hosts_is_a_noop_for_allow_all():
    assert NetworkPolicy().with_hosts(["a.com"]) == NetworkPolicy()


def test_an_empty_allowlist_still_restricts():
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST)
    assert policy.restricts_egress is True
    assert policy.allow_hosts == () and policy.allow_cidrs == ()


def test_a_policy_that_names_hosts_must_name_a_mode():
    """The mode-omission trap: {"allow_hosts": [...]} would otherwise build an ALLOW_ALL
    that carries hosts -- unrestricted at the wire, restricted-looking in the record."""
    with pytest.raises(ValueError, match="allow_all permits everything"):
        NetworkPolicy.from_dict({"allow_hosts": ["only-this.example.com"]})


def test_a_bare_star_is_rejected_from_the_caller_too():
    """The union already drops a provider-supplied '*'; the caller's own list is the
    likelier source of one, and it silently makes a labelled-restricted run open."""
    with pytest.raises(ValueError, match="bare '\\*'"):
        NetworkPolicy.from_dict({"mode": "allowlist", "allow_hosts": ["*"]})


def test_a_wildcard_prefix_is_not_a_bare_star():
    assert NetworkPolicy(
        mode=NetworkMode.ALLOWLIST, allow_hosts=("*.modal.host",)
    ).allow_hosts == ("*.modal.host",)


@pytest.mark.parametrize("field", ["allow_hosts", "allow_cidrs"])
def test_a_bare_string_is_rejected_rather_than_split_per_character(field):
    with pytest.raises(ValueError, match=f"{field} must be a sequence"):
        NetworkPolicy.from_dict({"mode": "allowlist", field: "api.example.com"})


def test_from_dict_rejects_a_non_mapping():
    """The user_overrides path feeds this unvalidated data."""
    with pytest.raises(ValueError, match="must be a mapping"):
        NetworkPolicy.from_dict("allowlist")


def test_iterables_other_than_strings_are_still_accepted():
    assert NetworkPolicy(
        mode=NetworkMode.ALLOWLIST, allow_hosts=(h for h in ["a.com"])
    ).allow_hosts == ("a.com",)
