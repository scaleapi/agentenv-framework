"""The browser task steps on a deployed chromium, inside a computer or alone, each made from a task's JSON as the step
registry reads it: each passes preflight on its env; a cookie and local storage set by steps sign in a tab opened in
the background, behind the active tab, which stays active; steps bring that tab to the front and close it by its URL,
delete the cookie and clear the storage; a tab step naming a URL no tab shows fails, listing the tabs open; and a tab at
a host in another script, which chromium spells by UTS 46, is closed by the URL as written."""

import asyncio
import time
from typing import Callable

from agent_env.env.envs.computer.utils import inside
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep
from tst.integration.computer.browser_data import ORIGIN, POLL_S, SERVE_S, SERVE_SITE, SITE, WAIT_S, _client

SIGNED_IN = "signed in as ada theme dark"
#: A page at a host in another script, on loopback as chromium sends any .localhost host: what's checked is how chromium
#: spells its URL, which holds whether or not the page loads.
IDN_PAGE = SITE.replace("127.0.0.1", "faß.localhost") + "other"


async def check_browser_steps(deployed, env_id: str, op_url: Callable[[str], str]) -> None:
    """The steps' effects, as the module says, read back through browser/v1 at ``op_url(op)``."""
    call, titled = _client(deployed, op_url)
    context = TaskStepContext(deployed_envs=[deployed])

    def step(type: str, **fields) -> TaskStep:
        made = get_task_step_registry()[type].from_dict({"id": type, "version": None, "type": type, "env_id": env_id,
                                                          **fields})
        assert made.config_problems() + made.preflight() == [], type
        return made

    async def tab_until(tab: str, holds: Callable[[dict], bool], what: str) -> dict:
        last, deadline = None, time.monotonic() + WAIT_S
        while time.monotonic() < deadline:
            last = next((t for t in (await call("tabs/list", method="GET"))["tabs"] if t["id"] == tab), None)
            if last is not None and holds(last):
                return last
            await asyncio.sleep(POLL_S)
        raise AssertionError(f"tab {tab} wasn't {what} within {WAIT_S:g} s: {last!r}")

    serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_SITE, str(SERVE_S)],
                                               timeout_s=SERVE_S + 60))
    try:
        await asyncio.sleep(3)
        await call("navigate", {"url": SITE})
        start = await titled("login")
        context = await step("browser_set_cookies", cookies=[
            {"name": "sid", "value": "ada", "url": SITE, "max_age": 3600}]).execute(context)
        context = await step("browser_set_local_storage", origin=ORIGIN, items={"theme": "dark"}).execute(context)
        context = await step("browser_open_tab", url=SITE, background=True).execute(context)
        opened = context.metadata["browser_steps"]["browser_open_tab"]["tab"]
        behind = await tab_until(opened, lambda tab: tab["title"] == SIGNED_IN, f"titled {SIGNED_IN!r}")
        after_open = [(tab["id"], tab["active"]) for tab in (await call("tabs/list", method="GET"))["tabs"]]
        context = await step("browser_activate_tab", url=SITE).execute(context)
        in_front = await titled(SIGNED_IN)
        context = await step("browser_close_tab", url=SITE).execute(context)
        after_close = [tab["url"] for tab in (await call("tabs/list", method="GET"))["tabs"]]
        try:
            await step("browser_close_tab", url=f"{SITE}nowhere").execute(context)
            missing = None
        except RuntimeError as e:
            missing = str(e)
        context = await step("browser_open_tab", url=IDN_PAGE, background=True).execute(context)
        idn_tab = context.metadata["browser_steps"]["browser_open_tab"]["tab"]
        idn_shown = await tab_until(idn_tab, lambda tab: tab["url"] not in ("", "about:blank"), "past about:blank")
        context = await step("browser_close_tab", url=IDN_PAGE).execute(context)
        idn_closed = context.metadata["browser_steps"]["browser_close_tab"]["tab"]
        after_idn = [tab["id"] for tab in (await call("tabs/list", method="GET"))["tabs"]]
        context = await step("browser_delete_cookies", name="sid").execute(context)
        context = await step("browser_clear_local_storage", origin=ORIGIN).execute(context)
        cookies = (await call("cookies/get", {"urls": [SITE]}))["cookies"]
        storage = (await call("local_storage/get", {"origins": [ORIGIN]}))["origins"]
    finally:
        serving.cancel()

    assert start["url"] == f"{SITE}login" and behind["url"] == SITE
    assert after_open == [(start["id"], True), (opened, False)], after_open
    assert in_front["id"] == opened and after_close == [f"{SITE}login"], after_close
    assert missing is not None and f"no tabs are at {SITE}nowhere" in missing and f"'{SITE}login'" in missing, missing
    assert idn_shown["url"] == SITE.replace("127.0.0.1", "xn--fa-hia.localhost") + "other", idn_shown
    assert idn_closed == idn_tab and after_idn == [start["id"]], (idn_closed, after_idn)
    assert context.metadata["browser_steps"]["browser_delete_cookies"]["deleted"] == 1
    assert cookies == [] and storage == [{"origin": ORIGIN, "localStorage": []}], (cookies, storage)
