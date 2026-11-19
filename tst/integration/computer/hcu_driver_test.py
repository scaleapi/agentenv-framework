"""The shipped HCU release reads and activates a GTK control in an app started before its first AX call."""

import json
import platform
import re
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent_env.bundle import parse_bundle, prepare_bundle
from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agentenv_protocol.computer import COMPUTER_READY
from tst.util.capabilities import missing_capability_reason

pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.skipif(platform.machine().lower() not in {"x86_64", "amd64"},
                       reason=missing_capability_reason("hcu_linux_amd64")),
]
BUNDLE = Path(__file__).parent / "hcu"


@pytest.fixture
def hcu_desktop(state):
    bundle = parse_bundle(BUNDLE)
    with namespace_routing():
        prepare_bundle(BUNDLE, sandbox="local")
        yield Env.get(f"{bundle.id_root}/desktop")


def _text(result):
    assert not result.isError, result
    return "\n".join(item.text for item in result.content if item.type == "text")


async def _record(deployed, name):
    command = ("from pathlib import Path; "
               f"print((Path.home() / '.hcu-accessibility' / {name!r}).read_text())")
    result = await inside.run(deployed, ["/usr/bin/python3", "-c", command], timeout_s=10)
    assert result["exit_code"] == 0, result
    return json.loads(result["stdout"])


@pytest.mark.asyncio
async def test_an_app_started_with_accessibility_is_read_and_acted_on_through_the_pinned_hcu(hcu_desktop):
    with namespace_routing():
        deployed = await Env.get(hcu_desktop.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            startup = await _record(deployed, "startup.json")
            assert startup["toolkit_accessibility"] is True
            async with streamable_http_client(f"{deployed.environment_url}/mcp") as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    apps = _text(await session.call_tool("AX", {"mode": "apps"}))
                    window = re.search(r'(w\d+) "HCU accessibility fixture"', apps).group(1)
                    tree = _text(await session.call_tool("AX", {"mode": "observe", "window_ref": window}))
                    button = next(line for line in tree.splitlines() if '"Confirm HCU action"' in line)
                    ref = re.search(r"\[ref=(r\d+)\]", button).group(1)
                    changed = _text(await session.call_tool("AX", {
                        "mode": "act", "actions": [{"type": "click", "ref": ref},
                                                  {"type": "wait", "text": "HCU action confirmed", "timeout_ms": 5000}],
                        "observe": "tree",
                    }))
            saved = await _record(deployed, "result.json")
            ready = await deployed.invoke(COMPUTER_READY, "status", None)
        finally:
            report = await teardown_run(context)

    assert saved == {"clicked": True, "pid": startup["pid"]}
    assert "HCU action confirmed" in changed
    checks = {check["name"]: check for check in ready["checks"]}
    assert ready["ready"] and checks["hcu"]["ok"] and checks["hcu tools"]["ok"]
    assert not report.failed
