"""The per-service data-load timeout scales with payload size instead of being flat.

It is an httpx read timeout, not a policy deadline -- the server does the whole ingest
inside one blocking JSON-RPC call, so "time to first response byte" happens to equal "total
load time". A flat 600s therefore killed healthy work: a 409MB service measured 502s on a
1-vCPU VM, 16% short of being severed while functioning correctly.

Size is a deliberately weak proxy (measured cost tracks record count, not bytes), so these
tests pin the properties that must hold rather than exact magic numbers.
"""

from __future__ import annotations

import pytest

from agent_env.env.gateway.constants import (
    DATA_PLANE_LOAD_TIMEOUT_MAX_S,
    DATA_PLANE_LOAD_TIMEOUT_S,
    data_plane_load_timeout_s,
)
from agent_env.env.gateway.gateway import Gateway

MB = 1_000_000


def test_unknown_size_gets_the_floor():
    """An unmeasurable payload must not silently get an unbounded wait."""
    assert data_plane_load_timeout_s(None) == DATA_PLANE_LOAD_TIMEOUT_S
    assert data_plane_load_timeout_s(0) == DATA_PLANE_LOAD_TIMEOUT_S


def test_negative_size_gets_the_floor():
    assert data_plane_load_timeout_s(-1) == DATA_PLANE_LOAD_TIMEOUT_S


def test_a_tiny_service_is_unchanged_from_the_old_flat_value():
    """Sub-MB services loaded in ~5s; nothing about them needed fixing."""
    assert data_plane_load_timeout_s(30_000) == DATA_PLANE_LOAD_TIMEOUT_S


def test_the_timeout_is_monotonic_in_size():
    sizes = [0, 1 * MB, 100 * MB, 500 * MB, 2000 * MB, 10_000 * MB]
    values = [data_plane_load_timeout_s(s) for s in sizes]
    assert values == sorted(values)


def test_it_never_exceeds_the_ceiling():
    """A genuinely hung load must still surface in bounded time rather than holding a
    worker slot and a billable VM for hours."""
    assert data_plane_load_timeout_s(10_000_000 * MB) == DATA_PLANE_LOAD_TIMEOUT_MAX_S


def test_it_is_never_below_the_floor():
    assert min(data_plane_load_timeout_s(s * MB) for s in range(0, 5000, 250)) >= DATA_PLANE_LOAD_TIMEOUT_S


@pytest.mark.parametrize("name,mb,measured_s", [
    # Measured on a 1-vCPU VM, the configuration where the flat cap actually bit.
    ("gmail", 409.1, 502),
    ("slack", 86.6, 294),
    ("github", 3495.7, 142),
    ("gdrive", 1984.3, 77),
    ("linear", 2491.1, 84),
])
def test_every_measured_load_now_has_real_headroom(name, mb, measured_s):
    """The regression this prevents: gmail sat 16% under a flat 600s while healthy."""
    budget = data_plane_load_timeout_s(int(mb * MB))
    assert budget > measured_s * 1.4, (
        f"{name}: {measured_s}s measured against a {budget}s budget is too tight"
    )


def test_gmail_specifically_gains_headroom_over_the_flat_cap():
    after = data_plane_load_timeout_s(int(409.1 * MB))
    assert after - 502 > 400, "gmail should now have comfortable headroom"


def test_the_gateway_proxy_outlasts_every_client_timeout():
    """LOCKED ordering (see constants.py): client add_data <= gateway REST proxy. If the
    proxy gave up first it would sever a load the client was still waiting on, and the
    size-aware client timeout would be silently useless."""
    assert Gateway.REST_PROXY_TIMEOUT_S >= DATA_PLANE_LOAD_TIMEOUT_MAX_S
    assert Gateway.REST_PROXY_TIMEOUT_S >= max(
        data_plane_load_timeout_s(s * MB) for s in range(0, 20_000, 500)
    )
