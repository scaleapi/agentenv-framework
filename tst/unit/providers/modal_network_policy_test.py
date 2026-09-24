"""Modal lowering, asserted against what was measured on live vm_runtime sandboxes."""

import pytest

from agent_env.providers.modal_sandbox import _modal_network_kwargs
from agent_env.providers.sandbox import NetworkMode, NetworkPolicy


def test_allow_all_contributes_no_kwargs_at_all():
    """The default must be indistinguishable from not passing the parameter.

    Measured: passing outbound_domain_allowlist at all -- even ["*"] -- flips Modal from
    NetworkAccess OPEN to ALLOWLIST and drops public raw-IP egress. A sandbox created with
    no policy has to reach Modal exactly as it did before this feature existed.
    """
    assert _modal_network_kwargs(NetworkPolicy()) == {}


def test_an_empty_allowlist_lowers_to_two_empty_lists():
    """An empty list means deny-all to Modal, not 'unrestricted'. This is the raw
    lowering; provider paths add the platform floor first (see egress_hosts_test)."""
    assert _modal_network_kwargs(NetworkPolicy(mode=NetworkMode.ALLOWLIST)) == {
        "outbound_domain_allowlist": [],
        "outbound_cidr_allowlist": [],
    }


def test_allowlist_sends_both_lists_because_modal_unions_them():
    """Measured: domain and cidr allowlists compose as OR, so passing both is additive."""
    policy = NetworkPolicy(
        mode=NetworkMode.ALLOWLIST, allow_hosts=("a.com", "b.com"), allow_cidrs=("10.0.0.0/8",)
    )
    assert _modal_network_kwargs(policy) == {
        "outbound_domain_allowlist": ["a.com", "b.com"],
        "outbound_cidr_allowlist": ["10.0.0.0/8"],
    }


@pytest.mark.parametrize(
    "policy",
    [
        NetworkPolicy(),
        NetworkPolicy(mode=NetworkMode.ALLOWLIST),
        NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("a.com",)),
    ],
)
def test_block_network_is_never_emitted(policy):
    """Mutually exclusive with encrypted_ports, which both Modal backends always pass."""
    assert "block_network" not in _modal_network_kwargs(policy)


def test_lowering_emits_plain_lists_not_tuples():
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("a.com",))
    kwargs = _modal_network_kwargs(policy)
    assert isinstance(kwargs["outbound_domain_allowlist"], list)
