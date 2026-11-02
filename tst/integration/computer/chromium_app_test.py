"""agentenv's chromium app on the ubuntu-24.04 base, as the chromium test bundle's desktop lists it: gate 3 makes this
machine's arch ready with it; deployed on two start_urls, it shows one Chromium window, on the first, with no first-run,
sign-in or permission window beside it, and the second a tab away, which an agent reaches with Control+Tab; its managed
policies are where Chrome for Testing reads them; and a reset of its data clears its profile and opens it again. Needs
Docker and the local registry on localhost:5000."""

import asyncio
import json
import platform
import re

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agentenv_protocol import client as protocol_v1

from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
ALPHA = "data:text/html,<title>Alpha page</title><h1>Alpha</h1>"
BRAVO = "data:text/html,<title>Bravo page</title><h1>Bravo</h1>"
# Each window the window manager manages, its class and its title.
WINDOWS = ('for w in $(xprop -root _NET_CLIENT_LIST | sed "s/.*# //; s/,//g"); do '
           'xprop -id "$w" WM_CLASS _NET_WM_NAME; done')


def test_gate_3_makes_this_machines_arch_ready_with_chromium(chromium_desktop):
    results = chromium_desktop.metadata["composed"]["container"]

    assert results[ARCH]["status"] == "ready" and results[ARCH]["probe"]["passed"], results[ARCH]
    assert [app.environment_name for app in chromium_desktop.composed_apps()] == ["chromium"]


@pytest.mark.asyncio
async def test_a_deploy_opens_its_start_urls_unprompted_and_a_reset_clears_its_profile(chromium_desktop):
    with namespace_routing():
        deployed = await Env.get(chromium_desktop.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [ALPHA, BRAVO]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            opened = await _chromium_windows(deployed, "Alpha page")
            async with streamablehttp_client(f"{deployed.environment_url}/mcp") as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    keyed = await session.call_tool("computer_key", {"keys": ["Control", "Tab"]})
            tabbed = await _chromium_windows(deployed, "Bravo page")
            policies = await inside.run(
                deployed, ["cat", "/etc/opt/chrome_for_testing/policies/managed/agentenv.json"], timeout_s=30)
            await inside.run(deployed, ["sh", "-c", "touch ~/.config/chromium/marker"], timeout_s=30)
            await protocol_v1.reset_data(f"{deployed.environment_url}/svc/chromium", headers=deployed.control_headers)
            marker = await inside.run(deployed, ["sh", "-c", "test -e ~/.config/chromium/marker"], timeout_s=30)
            reopened = await _chromium_windows(deployed, "Alpha page")
        finally:
            report = await teardown_run(context)

    assert opened == ["Alpha page - Google Chrome for Testing"]
    assert not keyed.isError and tabbed == ["Bravo page - Google Chrome for Testing"]
    assert {"BrowserSignin": 0, "CommandLineFlagSecurityWarningsEnabled": False}.items() <= json.loads(
        policies["stdout"]).items()
    assert marker["exit_code"] != 0 and reopened == ["Alpha page - Google Chrome for Testing"]
    assert not report.failed


async def _chromium_windows(deployed, title: str) -> list[str]:
    """The titles of the computer's Chromium windows once one of them starts with ``title``."""
    for _ in range(60):
        listed = (await inside.run(deployed, ["sh", "-c", WINDOWS], timeout_s=30))["stdout"]
        classes = re.findall(r'WM_CLASS\(STRING\) = .*"([^"]*)"\n_NET_WM_NAME\(UTF8_STRING\) = "(.*)"', listed)
        titles = [name for cls, name in classes if cls == "Chromium"]
        if any(name.startswith(title) for name in titles):
            return titles
        await asyncio.sleep(1)
    raise TimeoutError(f"no Chromium window titled {title!r}: {listed}")
