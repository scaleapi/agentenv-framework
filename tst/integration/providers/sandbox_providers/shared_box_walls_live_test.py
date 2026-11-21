"""A new shared box on Modal passes its walls self-test and keeps the tag across a reattach, and the checks only a real
box can answer hold: the iptables tools the walls need are there, the walls' own drop rules counted the probes they
blocked, so ``-i aes+`` matched routed and DNAT'd slot traffic, strict reverse-path filtering leaves the box's own
egress and its tunnel working, no interface but a slot's bridge is named ``aes*``, and a container under gVisor can't
open a raw socket, so a slot can't forge a reply into another's flow. It deploys nothing else, and ends the box.

Skipped unless ``AGENTENV_LIVE_MODAL=1``. Modal credentials come from ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``, else
the active profile in ``~/.modal.toml``; ``MODAL_ENVIRONMENT`` picks the Modal environment. Each probe's result prints
as a ``walls-probe`` JSON line, and the live checks as one ``walls-live`` line; run with ``-s`` to see them pass."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shlex
import time

import httpx
import pytest

from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandbox, ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.shared_box_selftest import PROBE_IMAGE, WALLS_PROBE_EVENT
from agent_env.providers.sandbox_providers.shared_box_walls import (
    FORWARD_CHAIN,
    INPUT_CHAIN,
    RP_FILTER_KNOBS,
    SLOT_BRIDGE_PREFIX,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.environ.get("AGENTENV_LIVE_MODAL") != "1", reason="set AGENTENV_LIVE_MODAL=1 to run on Modal"),
]

TUNNEL_PORT = 8000
BOX_CPU, BOX_MEMORY_MB, BOX_LIFETIME_S = 2, 4096, 1800
TUNNEL_WITHIN_S = 60
CONNECTIVITY_URL = "https://www.google.com/generate_204"

#: Run on the box with its own python3: what each live-only check found, as one JSON object.
LIVE_CHECKS = r'''
import json, os, subprocess, sys

def sh(command):
    done = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    return {"code": done.returncode, "out": done.stdout.strip(), "err": done.stderr.strip()[-300:]}

forward, inbound, knobs, prefix, image, url = sys.argv[1:7]
print(json.dumps({
    "tools": {tool: sh(f"command -v {tool}")["code"] == 0
              for tool in ("iptables", "iptables-restore", "ip6tables", "ip6tables-restore")},
    "iptables_version": sh("iptables --version")["out"],
    "ip6tables_version": sh("ip6tables --version")["out"],
    "br_netfilter": os.path.isdir("/sys/module/br_netfilter"),
    "bridge_nf_call_iptables": sh("cat /proc/sys/net/bridge/bridge-nf-call-iptables")["out"] or "absent",
    "rp_filter": {knob: sh(f"cat {knob}")["out"] for knob in knobs.split(",")},
    "slot_named_interfaces": [name for name in os.listdir("/sys/class/net") if name.startswith(prefix)],
    "forward_counters": sh(f"iptables -w -L {forward} -v -x -n")["out"],
    "input_counters": sh(f"iptables -w -L {inbound} -v -x -n")["out"],
    "ipv6_raw_counters": sh("ip6tables -w -t raw -L PREROUTING -v -x -n")["out"],
    "box_egress": sh(f"curl -sS -o /dev/null -w '%{{http_code}}' --max-time 15 {url}")["out"],
    "raw_socket_under_gvisor": sh(
        f"docker run --rm --network none {image} python -c "
        "'import socket; socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)'"),
}))
'''


def _counted(listing: str, *, target: str, in_iface: str, out_iface: str | None = None,
             destination_in: tuple[str, ...] | None = None) -> int:
    """The packets the rules of an ``iptables -L -v -x -n`` listing with ``target``, ``in_iface`` and, when given,
    ``out_iface`` and a destination among ``destination_in``, counted."""
    total = 0
    for line in listing.splitlines()[2:]:
        fields = line.split()
        if len(fields) < 9 or fields[2] != target or fields[5] != in_iface:
            continue
        if out_iface is not None and fields[6] != out_iface:
            continue
        if destination_in is not None and fields[8] not in destination_in:
            continue
        total += int(fields[0])
    return total


async def _tunnel_answers(box: ModalVmSandbox) -> int | str:
    """The status a GET through the box's tunnel gets from a web server run on the box for the length of the check."""
    serving = asyncio.ensure_future(box.exec_with_output(
        "bash", "-c", f"timeout {TUNNEL_WITHIN_S + 30} python3 -m http.server {TUNNEL_PORT} --bind 0.0.0.0"))
    deadline, last = time.monotonic() + TUNNEL_WITHIN_S, "no answer"
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            while time.monotonic() < deadline:
                try:
                    status = (await client.get(box.tunnel_urls[TUNNEL_PORT])).status_code
                    if status == 200:
                        return status
                    last = status
                except httpx.HTTPError as e:
                    last = f"{type(e).__name__}: {e}"
                await asyncio.sleep(3)
        return last
    finally:
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)


async def test_a_new_shared_box_verifies_its_walls_and_the_live_only_checks_hold(caplog):
    provider = ModalVmSandboxProvider(environment_name=os.environ.get("MODAL_ENVIRONMENT"))
    box_id = f"walls-live-{secrets.token_hex(4)}"
    box = None
    caplog.set_level(logging.INFO, logger="agent_env.providers.sandbox_providers.shared_box_selftest")
    try:
        try:
            box = await provider.create_shared_box(box_id, cpu=BOX_CPU, memory=BOX_MEMORY_MB, timeout=BOX_LIFETIME_S,
                                                   exposed_ports=[TUNNEL_PORT])
        finally:
            for record in caplog.records:
                if getattr(record, "event", None) == WALLS_PROBE_EVENT:
                    print("walls-probe", json.dumps({key: getattr(record, key) for key in
                                                     ("probe", "passed", "expected", "observed", "seconds")}))
        reattached = await provider.get_shared_box(box.sandbox_id, box_id)
        assert reattached.walls_verified

        argv = ["python3", "-c", LIVE_CHECKS, FORWARD_CHAIN, INPUT_CHAIN, ",".join(RP_FILTER_KNOBS),
                SLOT_BRIDGE_PREFIX, PROBE_IMAGE, CONNECTIVITY_URL]
        code, stdout, stderr = await reattached.exec_with_output("bash", "-c", shlex.join(argv))
        assert code == 0, stderr
        found = json.loads(stdout.strip().splitlines()[-1])
        found["tunnel"] = await _tunnel_answers(reattached)
        found["counted"] = {
            "slot_to_slot_drops": _counted(found["forward_counters"], target="DROP", in_iface=f"{SLOT_BRIDGE_PREFIX}+",
                                           out_iface=f"{SLOT_BRIDGE_PREFIX}+"),
            "private_range_drops": _counted(found["forward_counters"], target="DROP", in_iface=f"{SLOT_BRIDGE_PREFIX}+",
                                            out_iface="*", destination_in=(
                                                "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16",
                                                "100.64.0.0/10")),
            "box_drops": _counted(found["input_counters"], target="DROP", in_iface=f"{SLOT_BRIDGE_PREFIX}+"),
            "same_bridge_returns": _counted(found["forward_counters"], target="RETURN", in_iface="aes0env",
                                            out_iface="aes0env"),
        }
        print("walls-live", json.dumps(found))

        assert all(found["tools"].values()), found["tools"]
        assert found["rp_filter"] == {knob: "1" for knob in RP_FILTER_KNOBS}
        assert found["slot_named_interfaces"] == []
        assert found["box_egress"] == "204"
        assert found["tunnel"] == 200
        assert found["counted"]["slot_to_slot_drops"] > 0
        assert found["counted"]["private_range_drops"] > 0
        assert found["counted"]["box_drops"] > 0
        raw = found["raw_socket_under_gvisor"]
        assert raw["code"] != 0 and "PermissionError" in raw["err"], raw
    finally:
        if box is not None:
            await provider.terminate_shared_box(box.sandbox_id, box_id)
