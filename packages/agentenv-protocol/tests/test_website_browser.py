"""WebsiteBrowserEnvironment: agentenv's flags after a subclass's command, in a computer and alone, and the proxy rule
for each; a subclass refused for carrying one of them; the start tabs, the time zone and the window from its settings;
its launch run with no desktop when alone; a CDP connection over a new debugging pipe at each launch, which its ready
check asks, and a browser launched again when its server restarts; its shared memory in files only on a small
/dev/shm; an empty cache and the deploy's prompt answers in its profile at each launch; its policies, card params and
skill; its record's titles read from the browser, which sends no event for a page's; and an app that isn't a browser
launched and declared as before."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import sys

import pytest

from agentenv_protocol.computer import APP_SETTINGS_ENV, ComputerAppEnvironment, Launch, Ready, app, linux, session
from agentenv_protocol import website_browser
from agentenv_protocol.computer import app_runtime
from agentenv_protocol.computer.errors import DeviceUnavailable, InvalidArgument
from agentenv_protocol.computer.platform import Platform
from agentenv_protocol.website_browser import (ALONE_BYPASS, BROWSER, COMPUTER_BYPASS, POLICIES, SITE_PROXY_ENV, SKILL,
                                               BrowserControl, DevToolsError, WebsiteBrowserEnvironment)
from agentenv_protocol.website_browser.control import STOP

PROXY = "http://127.0.0.1:3128"
OWNED = ["--user-data-dir", "--disk-cache-dir", "--remote-debugging-pipe", "--no-first-run",
         "--no-default-browser-check", "--lang=en-US"]
# A browser that answers CDP on its debugging pipe, fds 3 and 4, as a Chromium does: Browser.getVersion with its
# product; its first tab, "start", made known and attached to; Target.createTarget by writing the tab's URL, or a new
# window's after "window ", to tabs.txt beside it and attaching to the tab; Page.navigate by writing "navigate <tab>
# <url>" there, then committing and loading it; the calls agentenv makes to follow a page with nothing; anything else
# with an error.
FAKE_BROWSER = """import json, os
buffer, loaders = b"", 0
def send(method, params, session=None):
    os.write(4, json.dumps({"method": method, "params": params, **({"sessionId": session} if session else {})}).encode()
             + b"\\0")
def page(target, url="about:blank"):
    return {"targetId": target, "type": "page", "url": url, "title": ""}
while chunk := os.read(3, 65536):
    buffer += chunk
    while b"\\0" in buffer:
        raw, buffer = buffer.split(b"\\0", 1)
        message = json.loads(raw)
        method, params, session = message["method"], message.get("params") or {}, message.get("sessionId")
        events = []
        if method == "Browser.getVersion":
            answer = {"result": {"product": "Fake/1"}}
        elif method == "Target.setDiscoverTargets":
            answer = {"result": {}}
            events = [("Target.targetCreated", {"targetInfo": page("start")}, None)]
        elif method == "Target.setAutoAttach":
            answer = {"result": {}}
            events = [("Target.attachedToTarget", {"sessionId": "s-start", "targetInfo": page("start"),
                                                   "waitingForDebugger": True}, None)]
        elif method == "Target.createTarget":
            with open("tabs.txt", "a") as tabs:
                tabs.write(("window " if params.get("newWindow") else "") + params["url"] + "\\n")
            target = f"t{len(open('tabs.txt').read().splitlines())}"
            answer = {"result": {"targetId": target}}
            events = [("Target.targetCreated", {"targetInfo": page(target)}, None),
                      ("Target.attachedToTarget", {"sessionId": "s-" + target, "targetInfo": page(target),
                                                   "waitingForDebugger": True}, None)]
        elif method == "Page.navigate":
            target, loaders = session[2:], loaders + 1
            with open("tabs.txt", "a") as tabs:
                tabs.write(f"navigate {target} {params['url']}\\n")
            answer = {"result": {"frameId": target, "loaderId": f"L{loaders}"}}
            events = [("Page.frameNavigated", {"frame": {"id": target, "loaderId": f"L{loaders}",
                                                         "url": params["url"]}}, session),
                      ("Page.loadEventFired", {}, session)]
        elif method == "Target.getTargets":
            answer = {"result": {"targetInfos": []}}
        elif method in ("Browser.setDownloadBehavior", "Page.enable", "Emulation.setDeviceMetricsOverride",
                        "Page.setInterceptFileChooserDialog", "Runtime.runIfWaitingForDebugger", "Network.enable",
                        "Network.disable", "Target.activateTarget", "Page.bringToFront"):
            answer = {"result": {}}
        else:
            answer = {"error": {"code": -32601, "message": "unknown method"}}
        os.write(4, json.dumps({"id": message["id"], **answer}).encode() + b"\\0")
        for event in events:
            send(*event)
"""


@linux(launch=Launch(command=["chromium/chrome", "--no-sandbox"], window={"class": "Chromium"},
                     ready=Ready(window=True, timeout_s=30)))
class Browser(WebsiteBrowserEnvironment):
    version = "156.0.8078.4"
    policies = {"DefaultGeolocationSetting": 2}


@linux(launch=Launch(command=["mail"], window={"class": "Mail"}, ready=Ready(window=True)))
class Mail(ComputerAppEnvironment):
    pass


@linux(launch=Launch(at="demand", command=["chromium/chrome"], window={"class": "Chromium"}, ready=Ready(window=True)))
class OnDemandBrowser(WebsiteBrowserEnvironment):
    pass


@pytest.fixture(autouse=True)
def browser_home(monkeypatch, tmp_path):
    """A browser on Linux, its session user's home under ``tmp_path``, inside a computer with no site proxy and 1 GiB
    of /dev/shm."""
    monkeypatch.setattr(app, "detect", lambda: Platform(os="linux"))
    monkeypatch.setattr(website_browser.environment, "_shm_bytes", lambda path="/dev/shm": 1 << 30)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("ENVIRONMENT_NAME", "chromium")
    for name in (SITE_PROXY_ENV, APP_SETTINGS_ENV):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "home"


def _settings(monkeypatch, **values) -> None:
    monkeypatch.setenv(APP_SETTINGS_ENV, json.dumps({"chromium": values}))


def _alone(monkeypatch) -> None:
    """On a computer with no display, a browser computer's."""
    monkeypatch.setattr(app, "detect", lambda: Platform(os="linux", display="none"))


def test_in_a_computer_its_flags_follow_its_command_then_its_start_tabs_and_its_window_is_waited_for(browser_home):
    launch = Browser().launch

    profile = f"--user-data-dir={browser_home / '.config/chromium'}"
    cache = f"--disk-cache-dir={browser_home / '.cache/chromium'}"
    assert launch.command == ["chromium/chrome", "--no-sandbox", profile, cache, *OWNED[2:], "about:blank"]
    assert launch.env == {"TZ": "UTC"}
    assert launch.window == {"class": "Chromium"} and launch.ready == Ready(window=True, timeout_s=30)


def test_in_a_computer_with_a_site_proxy_only_loopback_bypasses_it(monkeypatch):
    monkeypatch.setenv(SITE_PROXY_ENV, PROXY)

    command = Browser().launch.command

    assert [f"--proxy-server={PROXY}", f"--proxy-bypass-list={COMPUTER_BYPASS}",
            "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"] == command[8:11]
    assert not any(flag.startswith(("--headless", "--window-size")) for flag in command)


def test_alone_it_goes_through_its_proxy_headless_at_its_window_and_its_launch_waits_on_no_window(monkeypatch):
    _alone(monkeypatch)
    monkeypatch.setenv(SITE_PROXY_ENV, PROXY)
    _settings(monkeypatch, window={"width": 1024, "height": 768})

    browser = Browser()
    launch = browser.launch

    assert f"--proxy-bypass-list={ALONE_BYPASS}" in launch.command
    assert launch.command[-3:] == ["--window-size=1024,768", "--headless=new", "about:blank"]
    assert launch.window is None and launch.ready is None
    assert Launch.model_validate(launch.model_dump()) == launch
    assert browser._computer_app.desktop is False and browser._computer_app.timeout_s == 30


def test_alone_its_launch_keeps_the_port_and_command_it_waits_on_and_their_timeout(monkeypatch):
    @linux(launch=Launch(command=["chromium/chrome"], window={"class": "Chromium"},
                         ready=Ready(window=True, port=9000, command=["true"], timeout_s=45)))
    class Waiting(WebsiteBrowserEnvironment):
        pass

    _alone(monkeypatch)
    monkeypatch.setenv(SITE_PROXY_ENV, PROXY)

    waiting = Waiting()

    assert waiting.launch.ready == Ready(window=False, port=9000, command=["true"], timeout_s=45)
    assert Launch.model_validate(waiting.launch.model_dump()) == waiting.launch
    assert waiting._computer_app.timeout_s == 45


def test_alone_it_refuses_to_launch_without_a_site_proxy(monkeypatch):
    _alone(monkeypatch)

    with pytest.raises(ValueError, match=SITE_PROXY_ENV):
        Browser()


@pytest.mark.parametrize("flag", ["--remote-debugging-port=9222", "--remote-allow-origins=*", "--headless",
                                  "--proxy-server=http://elsewhere:3128", "--user-data-dir=/tmp/profile",
                                  "--disk-cache-dir=/tmp/cache", "--disable-dev-shm-usage"])
def test_a_subclass_whose_command_carries_a_flag_agentenv_owns_is_refused_when_declared(flag):
    @linux(launch=Launch(command=["chromium/chrome", flag]))
    class Widened(WebsiteBrowserEnvironment):
        pass

    with pytest.raises(ValueError, match="agentenv sets for every website browser"):
        Widened.declaration()


@pytest.mark.parametrize("mib, in_files", [(64, True), (1024, False)], ids=["docker-default", "sized"])
def test_it_keeps_its_shared_memory_in_files_only_when_dev_shm_is_under_1_gib(monkeypatch, mib, in_files):
    monkeypatch.setattr(website_browser.environment, "_shm_bytes", lambda path="/dev/shm": mib * 2**20)

    assert ("--disable-dev-shm-usage" in Browser().launch.command) is in_files


@pytest.mark.asyncio
@pytest.mark.parametrize("settings, prompts", [
    ({}, {"geolocation": 2, "notifications": 2}),
    ({"notifications": "ask"}, {"geolocation": 2}),
], ids=["deny", "ask"])
async def test_each_launch_starts_with_an_empty_cache_and_the_deploys_prompt_answers_in_its_profile(
        browser_home, monkeypatch, settings, prompts):
    _settings(monkeypatch, **settings)
    preferences = browser_home / ".config/chromium/Default/Preferences"
    preferences.parent.mkdir(parents=True)
    preferences.write_text(json.dumps({"homepage": "about:blank", "profile": {"default_content_setting_values": {
        "notifications": 1, "cookies": 1}}}))
    (browser_home / ".cache/chromium/Cache").mkdir(parents=True)
    browser = Browser()

    for fd in (await browser._launch_fds()).values():
        os.close(fd)
    browser.devtools.close()

    assert json.loads(preferences.read_text()) == {"homepage": "about:blank", "profile": {
        "default_content_setting_values": {"cookies": 1, **prompts}}}
    assert not (browser_home / ".cache/chromium").exists()


def test_its_settings_give_its_language_and_time_zone_its_start_tabs_open_over_its_pipe_and_none_is_a_flag(
        monkeypatch):
    _settings(monkeypatch, start_urls=["https://slack.com", "https://linear.app"], locale="fr-FR",
              time_zone="Europe/Paris")
    launch = Browser().launch
    _settings(monkeypatch, start_urls=["--remote-allow-origins=*"])

    assert launch.command[-1] == "about:blank" and "--lang=fr-FR" in launch.command
    assert not any("slack.com" in arg or "linear.app" in arg for arg in launch.command)
    assert launch.env == {"TZ": "Europe/Paris"}
    with pytest.raises(ValueError, match="start_urls are URLs, not flags"):
        Browser()


@pytest.mark.asyncio
async def test_alone_its_server_launches_it_at_once_in_its_environment_with_no_display_wait_or_launcher(
        monkeypatch, tmp_path):
    _alone(monkeypatch)
    monkeypatch.setenv(SITE_PROXY_ENV, PROXY)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.chdir(tmp_path)
    launched, waited, launchers = [], [], []

    async def launch(argv, cwd, env, fds=None):
        launched.append((env, sorted(fds or {})))
        return 4242

    async def up():
        waited.append(True)

    monkeypatch.setattr(session, "launch", launch)
    monkeypatch.setattr(session, "wait_until_up", up)
    monkeypatch.setattr(session, "add_launcher", lambda name, folder: launchers.append(name))
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")

    await Browser()._computer_app.start()

    assert launched == [({"TZ": "UTC", "ENVIRONMENT_NAME": "chromium"}, [3, 4])] and waited == [] and launchers == []


@pytest.mark.asyncio
async def test_in_a_computer_its_server_waits_for_the_display_and_adds_its_launcher(monkeypatch, tmp_path):
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.chdir(tmp_path)
    waited, launchers = [], []

    async def launch(argv, cwd, env, fds=None):
        return 4242

    async def up():
        waited.append(True)

    monkeypatch.setattr(session, "launch", launch)
    monkeypatch.setattr(session, "wait_until_up", up)
    monkeypatch.setattr(session, "add_launcher", lambda name, folder: launchers.append(name))
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")

    await Browser()._computer_app.start()

    assert waited == [True] and launchers == ["chromium"]


@linux(launch=Launch(command=[sys.executable, "-c", FAKE_BROWSER], ready=Ready(command=["true"])))
class FakeBrowser(WebsiteBrowserEnvironment):
    pass


@pytest.mark.asyncio
async def test_each_launch_gets_a_new_pipe_its_server_talks_cdp_over_and_its_ready_check_asks(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")
    browser = FakeBrowser()
    runtime = browser._computer_app
    unlaunched = await browser.devtools_answers()

    await runtime.launch_app()
    version = await browser.devtools.call("Browser.getVersion")
    with pytest.raises(DevToolsError, match="unknown method"):
        await browser.devtools.call("Page.printToPDF")
    launched = await _answering(browser)
    await runtime.stop_app()
    stopped = await browser.devtools_answers()
    await runtime.launch_app()
    relaunched = await _answering(browser)
    await runtime.stop_app()
    browser.devtools.close()

    assert unlaunched is False and version == {"product": "Fake/1"} and launched is True
    assert stopped is False and relaunched is True
    assert {"devtools_answers", "start_tabs_loaded"} <= set(runtime.ready_checks)


@pytest.mark.asyncio
async def test_a_launch_while_it_runs_opens_a_window_over_its_pipe_and_starts_no_second_copy(monkeypatch, tmp_path,
                                                                                             browser_home):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")
    browser = FakeBrowser()
    runtime = browser._computer_app

    first = await runtime.launch_app()
    (browser_home / ".cache/chromium").mkdir(parents=True, exist_ok=True)
    (browser_home / ".cache/chromium/kept").touch()
    second = await runtime.launch_app()
    answers = await _answering(browser)
    await runtime.stop_app()
    browser.devtools.close()

    assert second == first and answers is True and (browser_home / ".cache/chromium/kept").exists()
    assert (tmp_path / "tabs.txt").read_text().splitlines() == ["window about:blank"]


@pytest.mark.asyncio
async def test_headless_its_start_tabs_open_blank_held_to_its_window_then_load_in_order_and_count_it_ready(
        monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")
    _alone(monkeypatch)
    monkeypatch.setenv(SITE_PROXY_ENV, PROXY)
    _settings(monkeypatch, start_urls=["https://slack.com", "https://linear.app", "https://github.com"])
    browser = FakeBrowser()

    await browser._computer_app.launch_app()
    for _ in range(50):
        if ready := await browser.start_tabs_loaded():
            break
        await asyncio.sleep(0.1)
    await browser._computer_app.stop_app()
    browser.devtools.close()

    assert browser.launch.command[-2:] == ["--headless=new", "about:blank"]
    assert ready is True and (tmp_path / "tabs.txt").read_text().splitlines() == [
        "about:blank", "about:blank", "navigate start https://slack.com", "navigate t1 https://linear.app",
        "navigate t2 https://github.com"]
    tabs = browser.start_tab_launches[-1]["tabs"]
    assert [(tab["tab"], tab["url"], tab["outcome"], tab["site"]) for tab in tabs] == [
        ("start", "https://slack.com", "loaded", None), ("t1", "https://linear.app", "loaded", None),
        ("t2", "https://github.com", "loaded", None)]


@pytest.mark.asyncio
async def test_a_restarted_server_launches_a_browser_again_and_leaves_any_other_app_be(monkeypatch, tmp_path):
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")
    launched, stopped = [], []

    async def launch(argv, cwd, env, fds=None):
        launched.append((argv[0], sorted(fds or {})))
        return 4242

    async def stop(pid, timeout_s):
        stopped.append(pid)

    async def up():
        pass

    monkeypatch.setattr(session, "launch", launch)
    monkeypatch.setattr(session, "stop", stop)
    monkeypatch.setattr(session, "alive", lambda pid: True)
    monkeypatch.setattr(session, "wait_until_up", up)
    monkeypatch.setattr(session, "add_launcher", lambda name, folder: None)
    (tmp_path / "launched").mkdir()
    (tmp_path / "launched" / tmp_path.name).touch()
    (tmp_path / ".agentenv").mkdir()
    (tmp_path / ".agentenv" / "app.pid").write_text("4241")

    await Mail()._computer_app.start()
    left_alone = list(launched)
    browser, on_demand = Browser(), OnDemandBrowser()
    await browser._computer_app.start()
    await on_demand._computer_app.start()
    browser.devtools.close()
    on_demand.devtools.close()

    assert left_alone == [] and stopped == [4241, 4242]
    assert launched == [("chromium/chrome", [3, 4]), ("chromium/chrome", [3, 4])]


@pytest.mark.asyncio
async def test_a_browser_on_demand_that_was_never_launched_is_left_unlaunched_by_its_server(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")
    launched = []

    async def launch(argv, cwd, env, fds=None):
        launched.append(argv)
        return 4242

    monkeypatch.setattr(session, "launch", launch)

    await OnDemandBrowser()._computer_app.start()

    assert launched == []


def test_its_policies_are_agentenvs_with_its_own_over_them_where_its_build_reads_them(monkeypatch, tmp_path):
    monkeypatch.setattr(Browser, "policy_dir", tmp_path / "policies")

    Browser.__new__(Browser).write_policies()

    written = json.loads((tmp_path / "policies" / "agentenv.json").read_text())
    assert written == {**POLICIES, "DefaultGeolocationSetting": 2}
    assert written["DownloadDirectory"] == "${user_home}/Downloads"
    assert [step.name for step in Browser.declaration().install_steps] == ["write_policies"]


def test_its_card_declares_browser_v1_with_its_build_and_clock_and_alone_its_window(monkeypatch):
    in_computer = Browser().extension_params(BROWSER)
    _alone(monkeypatch)
    monkeypatch.setenv(SITE_PROXY_ENV, PROXY)
    alone = Browser().extension_params(BROWSER)

    assert in_computer == {"chromium": "156.0.8078.4", "clock": "real"}
    assert alone == {"chromium": "156.0.8078.4", "clock": "real", "window": {"width": 1280, "height": 800}}


@pytest.mark.asyncio
async def test_its_skill_names_sites_by_their_names_and_its_upload_and_download_folders_which_it_makes(browser_home):
    browser = Browser()

    skill = await browser.browser_skill("website-browser")

    assert Browser.declaration().skill.method == "browser_skill"
    assert skill["description"] == "How to use this website browser." and skill["skill_md"].endswith(SKILL)
    assert (browser_home / "Uploads").is_dir() and (browser_home / "Downloads").is_dir()


def test_an_app_that_isnt_a_browser_launches_and_declares_as_before():
    mail = Mail()
    stored = Mail.declaration().model_dump(mode="json", exclude_none=True)

    assert mail.launch is Mail._installs["linux"].launch and mail._computer_app.desktop is True
    assert mail._computer_app.launch_running is None
    assert stored["os"]["linux"]["launch"] == {"at": "boot", "command": ["mail"], "window": {"class": "Mail"},
                                               "ready": {"window": True, "timeout_s": 60.0}}


#: How long a test waits on the fake browser, and how often it looks.
WAIT_S, POLL_S = 5.0, 0.05


async def _answering(browser) -> bool:
    """Whether ``browser``'s ready check says it answers, within ``WAIT_S``."""
    for _ in range(int(WAIT_S / POLL_S)):
        if await browser.devtools_answers():
            return True
        await asyncio.sleep(POLL_S)
    return False


class StubDevTools:
    """A browser connection that answers each call from ``answers`` by method, records it, and hands the control the
    events a test sends."""

    def __init__(self, answers=None):
        self.calls, self.answers, self.listener = [], answers or {}, None

    def listen(self, listener):
        self.listener = listener

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        self.calls.append((method, params or {}, session_id))
        answer = self.answers.get(method, {})
        return answer(params, session_id) if callable(answer) else answer

    async def request(self, method, params=None, *, session_id=None):
        try:
            return {"result": await self.call(method, params, session_id=session_id)}
        except DevToolsError as e:
            return {"error": {"message": str(e)}}

    def send(self, method, params, session=None):
        self.listener({"method": method, "params": params, **({"sessionId": session} if session else {})})


def _page(target, url="about:blank", title=""):
    return {"targetInfo": {"targetId": target, "type": "page", "url": url, "title": title}}


@pytest.mark.asyncio
async def test_its_control_records_tabs_in_opened_order_and_who_closed_each_and_leaves_out_its_hidden_pages(tmp_path):
    devtools = StubDevTools({"Target.createTarget": {"targetId": "hidden"},
                             "Target.attachToTarget": {"sessionId": "s-hidden"}})
    control = BrowserControl(devtools, "agent")
    await control.attach()

    for target in ("a", "b", "c"):
        devtools.send("Target.targetCreated", _page(target))
    devtools.send("Target.targetInfoChanged", _page("a", "https://slack.com/", "Slack"))
    devtools.send("Target.targetDestroyed", {"targetId": "b"})
    await control.close_tab("c")
    devtools.send("Target.targetDestroyed", {"targetId": "c"})
    async with control.hidden_page() as session:
        devtools.send("Target.targetCreated", _page("hidden"))
    record = await control.record()
    await control.attach()

    assert [tab["id"] for tab in record["tabs"]] == ["a"] and record["tabs"][0]["title"] == "Slack"
    assert [(tab["id"], tab["closed_by"]) for tab in record["closed_tabs"]] == [("b", "unknown"), ("c", "agentenv")]
    assert record["since"] == control.since
    assert session == "s-hidden" and ("Target.closeTarget", {"targetId": "hidden"}, None) in devtools.calls
    assert [(tab["id"], tab["closed_by"]) for tab in control.closed_tabs][-1] == ("a", "stop")
    assert [call[0] for call in devtools.calls[:3]] == ["Target.setDiscoverTargets", "Target.setAutoAttach",
                                                        "Browser.setDownloadBehavior"]
    assert devtools.calls[2][1] == {"behavior": "default", "eventsEnabled": True}


@pytest.mark.asyncio
async def test_each_page_it_attaches_to_is_told_to_run_so_a_popup_it_attached_to_as_it_opened_navigates(tmp_path):
    devtools = StubDevTools()
    control = BrowserControl(devtools, "agent")
    await control.attach()
    devtools.send("Target.attachedToTarget", {"sessionId": "s-popup", "waitingForDebugger": True, **_page("popup")})
    await asyncio.sleep(0)

    assert ("Page.enable", {}, "s-popup") in devtools.calls
    assert ("Runtime.runIfWaitingForDebugger", {}, "s-popup") in devtools.calls
    assert not any(method == "Emulation.setDeviceMetricsOverride" for method, _, _ in devtools.calls)
    assert next(params for method, params, _ in devtools.calls if method == "Target.setAutoAttach")[
        "waitForDebuggerOnStart"] is False


@pytest.mark.asyncio
async def test_alone_each_page_it_attaches_to_is_held_to_its_window_before_it_runs(tmp_path):
    devtools = StubDevTools()
    control = BrowserControl(devtools, "agent", (1024, 768))
    await control.attach()
    devtools.send("Target.attachedToTarget", {"sessionId": "s-popup", "waitingForDebugger": True, **_page("popup")})
    await asyncio.sleep(0)

    sent = [method for method, _, session in devtools.calls if session == "s-popup"]
    assert next(params for method, params, _ in devtools.calls if method == "Target.setAutoAttach")[
        "waitForDebuggerOnStart"] is True
    assert sent == ["Page.enable", "Emulation.setDeviceMetricsOverride", "Page.setInterceptFileChooserDialog",
                    "Runtime.runIfWaitingForDebugger"]
    assert ("Emulation.setDeviceMetricsOverride", {"width": 1024, "height": 768, "deviceScaleFactor": 1,
                                                   "mobile": False}, "s-popup") in devtools.calls


@pytest.mark.asyncio
async def test_a_page_it_couldnt_prepare_is_still_told_to_run_so_it_never_stays_held():
    def refused(params, session):
        raise DevToolsError("Target closed")

    devtools = StubDevTools({"Page.enable": refused})
    control = BrowserControl(devtools, "agent", (1024, 768))
    await control.attach()
    devtools.send("Target.attachedToTarget", {"sessionId": "s-popup", "waitingForDebugger": True, **_page("popup")})
    await asyncio.sleep(0)

    assert [method for method, _, session in devtools.calls if session == "s-popup"] == [
        "Page.enable", "Runtime.runIfWaitingForDebugger"]


class Navigating(StubDevTools):
    """A browser whose first tab is ``start`` and whose new tabs are t1, t2, …, each attached as it opens, answering a
    navigation as ``pages`` says by URL: a status and the URL it ends on, then committed and, unless ``no-load``
    follows, loaded, or with ``("redirect", status, url)`` following, sending itself there by script before it loads;
    ``refused``, a network error; ``silent``, never answered, as a page that never answers isn't; or ``stale``, a load
    event with no commit."""

    def __init__(self, pages):
        super().__init__({"Target.getTargets": {"targetInfos": []}})
        self.pages, self.opened, self.loaders = pages, 0, 0

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        if method == "Target.createTarget":
            self.calls.append((method, params, session_id))
            self.opened += 1
            target = f"t{self.opened}"
            self.send("Target.targetCreated", _page(target))
            self.send("Target.attachedToTarget", {"sessionId": f"s-{target}", "waitingForDebugger": True,
                                                  **_page(target)})
            return {"targetId": target}
        if method != "Page.navigate":
            return await super().call(method, params, session_id=session_id, timeout_s=timeout_s)
        self.calls.append((method, params, session_id))
        page, target = self.pages[params["url"]], session_id[2:]
        self.loaders += 1
        loader, request = f"L{self.loaders}", f"r{self.loaders}"
        if page == "refused":
            return {"frameId": target, "loaderId": loader, "errorText": "net::ERR_CONNECTION_REFUSED"}
        if page == "silent":
            await asyncio.Event().wait()
        if page == "stale":
            self.send("Page.loadEventFired", {}, session_id)
            return {"frameId": target, "loaderId": loader}
        status, final_url, *rest = page
        self._document(session_id, target, loader, request, status, final_url)
        if rest and isinstance(rest[0], tuple):
            _, status, final_url = rest[0]
            self._document(session_id, target, f"{loader}-redirect", f"{request}-redirect", status, final_url)
        if "no-load" not in rest:
            self.send("Page.loadEventFired", {}, session_id)
        return {"frameId": target, "loaderId": loader}

    def _document(self, session, target, loader, request, status, url):
        document = {"requestId": request, "loaderId": loader, "type": "Document", "frameId": target}
        self.send("Network.requestWillBeSent", document, session)
        self.send("Network.responseReceived", {**document, "response": {"status": status, "url": url}}, session)
        self.send("Page.frameNavigated", {"frame": {"id": target, "loaderId": loader, "url": url}}, session)


async def _started(devtools, viewport=(1280, 800)) -> BrowserControl:
    """A control following a browser just launched, on its first tab, ``start``."""
    control = BrowserControl(devtools, "agent", viewport)
    await control.attach()
    devtools.send("Target.targetCreated", _page("start"))
    devtools.send("Target.attachedToTarget", {"sessionId": "s-start", "waitingForDebugger": True, **_page("start")})
    return control


def _shop(url):
    return "shop" if "shop.test" in url else None


def _outcomes(tabs):
    return [(tab["tab"], tab["url"], tab["final_url"], tab["status"], tab["error"], tab["site"], tab["outcome"])
            for tab in tabs]


@pytest.mark.asyncio
async def test_its_start_tabs_load_in_order_in_its_first_tab_then_blank_ones_held_first_and_the_first_is_active():
    devtools = Navigating({"https://shop.test/": (200, "https://shop.test/login"), "https://slack.com/": "silent",
                           "https://github.com/": (404, "https://github.com/")})
    control = await _started(devtools)
    loop = asyncio.get_running_loop()
    started = loop.time()

    tabs = await control.load_start_tabs(list(devtools.pages), _shop, own_timeout_s=5, live_timeout_s=0.3)

    assert _outcomes(tabs) == [
        ("start", "https://shop.test/", "https://shop.test/login", 200, None, "shop", "loaded"),
        ("t1", "https://slack.com/", None, None, None, None, "timed_out"),
        ("t2", "https://github.com/", "https://github.com/", 404, None, None, "failed")]
    assert loop.time() - started < 2
    assert list(control.tabs) == ["start", "t1", "t2"] and all(isinstance(tab["load_ms"], int) for tab in tabs)
    assert _sent(devtools, "Target.createTarget") == [({"url": "about:blank", "background": True}, None)] * 2
    for tab_session in ("s-start", "s-t1", "s-t2"):
        sent = [method for method, _, on in devtools.calls if on == tab_session]
        assert sent.index("Emulation.setDeviceMetricsOverride") < sent.index("Network.enable") < \
            sent.index("Page.navigate") < sent.index("Network.disable"), sent
    assert devtools.calls[-2][:2] == ("Target.activateTarget", {"targetId": "start"})
    assert not control._loading


@pytest.mark.asyncio
@pytest.mark.parametrize("url, page, outcome, error, status, waited", [
    ("https://shop.test/", (503, "https://shop.test/", "no-load"), "failed", None, 503, False),
    ("https://shop.test/", "refused", "failed", "net::ERR_CONNECTION_REFUSED", None, False),
    ("https://shop.test/", "silent", "timed_out", None, None, True),
    ("https://shop.test/", "stale", "timed_out", None, None, True),
    ("https://slack.com/", "silent", "timed_out", None, None, True),
    ("https://slack.com/", (503, "https://slack.com/", "no-load"), "timed_out", None, 503, True),
    ("https://slack.com/", "refused", "failed", "net::ERR_CONNECTION_REFUSED", None, False),
], ids=["own-error-status-at-once", "own-refused", "own-never-loads", "own-stale-load", "live-never-loads",
        "live-error-status-waited", "live-refused"])
async def test_a_start_tab_settles_strictly_on_an_env_site_and_at_most_its_time_on_a_live_one(
        url, page, outcome, error, status, waited):
    control = await _started(Navigating({url: page}))
    loop = asyncio.get_running_loop()
    started = loop.time()

    (tab,) = await control.load_start_tabs([url], _shop, own_timeout_s=0.3 if url.startswith("https://shop") else 5,
                                           live_timeout_s=0.3 if url.startswith("https://slack") else 5)

    assert (tab["outcome"], tab["error"], tab["status"]) == (outcome, error, status)
    assert (loop.time() - started >= 0.3) is waited


@pytest.mark.asyncio
@pytest.mark.parametrize("redirect, final, status, outcome", [
    (("redirect", 200, "https://shop.test/notes"), "https://shop.test/notes", 200, "loaded"),
    (("redirect", 404, "https://shop.test/gone"), "https://shop.test/gone", 404, "failed"),
], ids=["to-a-page", "to-an-error"])
async def test_a_start_page_that_sends_itself_elsewhere_before_it_loads_is_judged_by_the_page_it_ends_on(
        redirect, final, status, outcome):
    control = await _started(Navigating({"https://shop.test/": (200, "https://shop.test/", redirect)}))

    (tab,) = await control.load_start_tabs(["https://shop.test/"], _shop, own_timeout_s=5)

    assert (tab["final_url"], tab["status"], tab["outcome"]) == (final, status, outcome)


@pytest.mark.asyncio
async def test_a_start_tab_still_loading_when_the_browser_ends_fails_naming_that_and_is_kept_with_its_pipe_gone():
    def gone(params, session):
        raise ConnectionError("the browser closed its debugging pipe")

    devtools = Navigating({"https://shop.test/": "silent"})
    devtools.answers["Target.activateTarget"] = gone
    control = await _started(devtools)
    loading = asyncio.ensure_future(control.load_start_tabs(["https://shop.test/"], _shop, own_timeout_s=5))
    for _ in range(100):
        if control._loading:
            break
        await asyncio.sleep(0.01)
    control.ended(STOP)

    (tab,) = await loading

    assert (tab["outcome"], tab["error"]) == ("failed", "the browser ended")


class Painting(StubDevTools):
    """A browser whose every tab has focus and paints, but those whose sessions are in ``blank``, which never answer a
    capture; it records each capture's timeout."""

    def __init__(self, blank=()):
        super().__init__({"Runtime.evaluate": lambda params, session: {"result": {"value": ["visible", True]}}})
        self.blank, self.timeouts = set(blank), []

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        if method != "Page.captureScreenshot":
            return await super().call(method, params, session_id=session_id, timeout_s=timeout_s)
        self.timeouts.append(timeout_s)
        if session_id in self.blank:
            raise asyncio.TimeoutError
        return {"data": f"png of {session_id}"}


async def _one_tab(devtools, url) -> BrowserControl:
    control = BrowserControl(devtools, "agent")
    await control.attach()
    devtools.send("Target.targetCreated", _page("a", url))
    devtools.send("Target.attachedToTarget", {"sessionId": "s-a", **_page("a", url)})
    await asyncio.sleep(0)
    return control


@pytest.mark.asyncio
@pytest.mark.parametrize("url, blank, png, timeout", [
    ("https://mail.test/", (), "png of s-a", 3.0),
    ("https://mail.test/", ("s-a",), None, 3.0),
    ("", ("s-a",), None, 1.0),
])
async def test_its_capture_is_the_active_tabs_page_or_none_for_one_that_paints_nothing_sooner_with_no_url(
        url, blank, png, timeout):
    devtools = Painting(blank)
    control = await _one_tab(devtools, url)

    assert await control.capture() == png and devtools.timeouts == [timeout]


@pytest.mark.asyncio
async def test_its_page_input_goes_to_the_active_tab_in_order_and_takes_only_input_calls():
    devtools = Painting()
    control = await _one_tab(devtools, "https://mail.test/")
    events = [{"method": "Input.dispatchMouseEvent", "params": {"type": "mousePressed", "x": 1, "y": 2}},
              {"method": "Input.insertText", "params": {"text": "hi"}}]

    await control.dispatch(events)
    with pytest.raises(InvalidArgument, match="Page.navigate"):
        await control.dispatch([*events, {"method": "Page.navigate", "params": {"url": "https://elsewhere.test/"}}])

    sent = [(method, params, session) for method, params, session in devtools.calls if method.startswith("Input.")]
    assert sent == [("Input.dispatchMouseEvent", {"type": "mousePressed", "x": 1, "y": 2}, "s-a"),
                    ("Input.insertText", {"text": "hi"}, "s-a")]


class Tabbed(Painting):
    """A browser whose tabs paint and none has focus, with a history of two pages, the second current, and each call's
    answer by method in ``answers``."""

    def __init__(self, answers=None, blank=()):
        super().__init__(blank)
        self.answers.update({"Runtime.evaluate": lambda params, session: {"result": {"value": ["visible", False]}},
                             "Page.getNavigationHistory": {"currentIndex": 1, "entries": [{"id": 7}, {"id": 8}]},
                             "Target.createTarget": {"targetId": "b"}, "Page.navigate": {}, **(answers or {})})


async def _tabs(devtools, *targets) -> BrowserControl:
    control = BrowserControl(devtools, "agent", (1280, 800))
    await control.attach()
    for target in targets:
        devtools.send("Target.targetCreated", _page(target, f"https://mail.test/{target}", f"Mail {target}"))
        devtools.send("Target.attachedToTarget", {"sessionId": f"s-{target}", **_page(target)})
    await asyncio.sleep(0)
    return control


def _sent(devtools, method):
    return [(params, session) for name, params, session in devtools.calls if name == method]


@pytest.mark.asyncio
async def test_it_navigates_the_active_tab_and_moves_in_its_history_or_says_why_it_cant():
    devtools = Tabbed({"Page.navigate": lambda params, session: {"errorText": "net::ERR_NAME_NOT_RESOLVED"}
                       if "nowhere" in params["url"] else {}})
    control = await _tabs(devtools, "a")

    await control.navigate("https://mail.test/inbox")
    await control.history("back")
    await control.history("reload")
    with pytest.raises(InvalidArgument, match="net::ERR_NAME_NOT_RESOLVED"):
        await control.navigate("https://nowhere.test/")
    with pytest.raises(InvalidArgument, match="no page to go forward to"):
        await control.history("forward")

    assert _sent(devtools, "Page.navigate")[0] == ({"url": "https://mail.test/inbox"}, "s-a")
    assert _sent(devtools, "Page.navigateToHistoryEntry") == [({"entryId": 7}, "s-a")]
    assert _sent(devtools, "Page.reload") == [({}, "s-a")]


@pytest.mark.asyncio
async def test_a_tab_it_opens_loads_once_held_to_the_window_comes_to_the_front_and_one_it_closes_is_its_own():
    devtools = Tabbed()
    control = await _tabs(devtools, "a")

    opening = asyncio.ensure_future(control.open_tab("https://mail.test/b"))
    await asyncio.sleep(0)
    devtools.send("Target.targetCreated", _page("b", "https://mail.test/b", "Mail b"))
    devtools.send("Target.attachedToTarget", {"sessionId": "s-b", **_page("b")})
    opened = await opening
    after_open = await control.tab_list()
    await control.activate("a")
    after_switch = await control.tab_list()
    with pytest.raises(InvalidArgument, match="no tab c is open"):
        await control.activate("c")
    await control.close("b")
    devtools.send("Target.targetDestroyed", {"targetId": "b"})

    on_b = [name for name, _, session in devtools.calls if session == "s-b"]
    assert opened == "b" and _sent(devtools, "Target.createTarget") == [({"url": "about:blank", "background": False},
                                                                         None)]
    assert ({"url": "https://mail.test/b"}, "s-b") in _sent(devtools, "Page.navigate")
    assert on_b.index("Emulation.setDeviceMetricsOverride") < on_b.index("Page.navigate")
    assert [(tab["id"], tab["active"]) for tab in after_open] == [("a", False), ("b", True)]
    assert [(tab["id"], tab["active"]) for tab in after_switch] == [("a", True), ("b", False)]
    assert ({}, "s-a") in _sent(devtools, "Page.bringToFront")
    assert [(tab["id"], tab["closed_by"]) for tab in control.closed_tabs] == [("b", "agentenv")]


@pytest.mark.asyncio
@pytest.mark.parametrize("focus_moves", [False, True])
async def test_a_tab_opened_in_the_background_opens_behind_the_active_one_which_stays_in_front(focus_moves):
    focus = {"s-a": True}
    devtools = Tabbed({"Runtime.evaluate": lambda params, session: {
        "result": {"value": ["visible", focus.get(session, False)]}}})
    control = await _tabs(devtools, "a")

    opening = asyncio.ensure_future(control.open_tab("https://mail.test/b", background=True))
    await asyncio.sleep(0)
    devtools.send("Target.targetCreated", _page("b", "https://mail.test/b", "Mail b"))
    devtools.send("Target.attachedToTarget", {"sessionId": "s-b", **_page("b")})
    if focus_moves:
        focus.update({"s-a": False, "s-b": True})
    opened = await opening

    assert opened == "b" and _sent(devtools, "Target.createTarget") == [({"url": "about:blank", "background": True},
                                                                         None)]
    assert ({"url": "https://mail.test/b"}, "s-b") in _sent(devtools, "Page.navigate")
    assert ({}, "s-b") not in _sent(devtools, "Page.bringToFront")
    assert _sent(devtools, "Target.activateTarget") == ([({"targetId": "a"}, None)] if focus_moves else [])
    if not focus_moves:
        assert [(tab["id"], tab["active"]) for tab in await control.tab_list()] == [("a", True), ("b", False)]


@pytest.mark.asyncio
async def test_a_background_open_leaves_a_tab_another_call_switched_to_meanwhile_in_front():
    focus = {"s-a": True}
    devtools = Tabbed({"Runtime.evaluate": lambda params, session: {
        "result": {"value": ["visible", focus.get(session, False)]}}})
    control = await _tabs(devtools, "a", "c")

    opening = asyncio.ensure_future(control.open_tab("https://mail.test/b", background=True))
    await asyncio.sleep(0)
    focus.update({"s-a": False, "s-c": True})
    devtools.send("Target.targetCreated", _page("b", "https://mail.test/b", "Mail b"))
    devtools.send("Target.attachedToTarget", {"sessionId": "s-b", **_page("b")})
    await opening

    assert _sent(devtools, "Target.activateTarget") == []
    assert [(tab["id"], tab["active"]) for tab in await control.tab_list()] == [("a", False), ("c", True), ("b", False)]


@pytest.mark.asyncio
async def test_a_hidden_tab_whose_page_still_says_it_has_focus_isnt_the_active_one():
    states = {"s-a": ["hidden", True], "s-b": ["visible", True]}
    devtools = Tabbed({"Runtime.evaluate": lambda params, session: {"result": {"value": states[session]}}})
    control = await _tabs(devtools, "a", "b")

    assert await control.active_tab() == "b"


@pytest.mark.asyncio
async def test_a_dialog_makes_its_tab_the_active_one_its_state_names_it_and_the_agent_answers_it():
    devtools = Tabbed()
    control = await _tabs(devtools, "a", "b")
    with pytest.raises(InvalidArgument, match="no dialog is open"):
        await control.answer_dialog(True)
    devtools.send("Page.javascriptDialogOpening", {"type": "prompt", "message": "Name?", "defaultPrompt": "Ada",
                                                   "url": "https://mail.test/b"}, session="s-b")

    state = await control.state()
    await control.answer_dialog(True, "Grace")
    devtools.send("Page.javascriptDialogClosed", {"result": True, "userInput": "Grace"}, session="s-b")

    assert state == {"tab": "b", "url": "https://mail.test/b", "title": "Mail b",
                     "dialog": {"type": "prompt", "message": "Name?", "default_text": "Ada"}}
    assert _sent(devtools, "Page.handleJavaScriptDialog") == [({"accept": True, "promptText": "Grace"}, "s-b")]
    (dialog,) = control.dialog_log
    assert (dialog["answer"], dialog["by"], dialog["input"]) == ("accepted", "agent", "Grace")


@pytest.mark.asyncio
async def test_input_that_opens_a_dialog_returns_once_it_opens_and_the_rest_waits_for_nothing():
    class Held(Tabbed):
        """A page whose handler of a mouse release opens a dialog, so the release isn't answered until it's closed."""

        def __init__(self):
            super().__init__()
            self.released = asyncio.get_running_loop().create_future()

        async def call(self, method, params=None, *, session_id=None, timeout_s=None):
            if method == "Input.dispatchMouseEvent" and params["type"] == "mouseReleased":
                self.calls.append((method, params, session_id))
                self.send("Page.javascriptDialogOpening", {"type": "confirm", "message": "Delete?"},
                          session=session_id)
                return await self.released
            return await super().call(method, params, session_id=session_id, timeout_s=timeout_s)

    devtools = Held()
    control = await _tabs(devtools, "a")
    events = [{"method": "Input.dispatchMouseEvent", "params": {"type": kind, "x": 1, "y": 1}}
              for kind in ("mousePressed", "mouseReleased", "mouseMoved")]

    await asyncio.wait_for(control.dispatch(events), 5)
    sent = [params["type"] for method, params, _ in devtools.calls if method == "Input.dispatchMouseEvent"]
    devtools.released.set_result({})
    await asyncio.sleep(0)

    assert sent == ["mousePressed", "mouseReleased"] and (await control.state())["dialog"]["message"] == "Delete?"


@pytest.mark.asyncio
async def test_input_that_opens_a_dialog_answered_for_the_agent_goes_on_once_it_is_answered():
    class Answered(Tabbed):
        """A page whose handler of a mouse release opens a dialog, answered at once by the control's setting."""

        def __init__(self):
            super().__init__({"Page.handleJavaScriptDialog": self._close})
            self.released = asyncio.get_running_loop().create_future()

        def _close(self, params, session):
            self.released.set_result({})
            return {}

        async def call(self, method, params=None, *, session_id=None, timeout_s=None):
            if method == "Input.dispatchMouseEvent" and params["type"] == "mouseReleased":
                self.calls.append((method, params, session_id))
                self.send("Page.javascriptDialogOpening", {"type": "alert", "message": "Saved"}, session=session_id)
                return await self.released
            return await super().call(method, params, session_id=session_id, timeout_s=timeout_s)

    devtools = Answered()
    control = BrowserControl(devtools, "accept", (1280, 800))
    await control.attach()
    devtools.send("Target.targetCreated", _page("a", "https://mail.test/a", "Mail a"))
    devtools.send("Target.attachedToTarget", {"sessionId": "s-a", **_page("a")})
    await asyncio.sleep(0)
    events = [{"method": "Input.dispatchMouseEvent", "params": {"type": kind, "x": 1, "y": 1}}
              for kind in ("mousePressed", "mouseReleased", "mouseMoved")]

    await asyncio.wait_for(control.dispatch(events), 5)

    assert [params["type"] for method, params, _ in devtools.calls if method == "Input.dispatchMouseEvent"] == [
        "mousePressed", "mouseReleased", "mouseMoved"]


@pytest.mark.asyncio
async def test_its_page_capture_names_the_tab_it_captured_though_another_becomes_active_meanwhile():
    class Switching(Tabbed):
        """Tabs of which only ``focused``'s page has focus, which a capture of the first moves to the second."""

        def __init__(self):
            super().__init__({"Runtime.evaluate": lambda params, session: {
                "result": {"value": ["visible", session == self.focused]}}})
            self.focused = "s-a"

        async def call(self, method, params=None, *, session_id=None, timeout_s=None):
            if method == "Page.captureScreenshot":
                self.focused = "s-b"
            return await super().call(method, params, session_id=session_id, timeout_s=timeout_s)

    devtools = Switching()
    control = await _tabs(devtools, "a", "b")

    shot = await control.capture_page()

    assert shot == {"png": "png of s-a", "page": {"tab": "a", "url": "https://mail.test/a", "title": "Mail a"}}
    assert (await control.state())["tab"] == "b"


@pytest.mark.asyncio
async def test_while_a_dialog_keeps_its_page_from_painting_its_capture_is_the_last_it_took_without_waiting_on_it():
    devtools = Tabbed()
    control = await _tabs(devtools, "a")
    before = await control.capture()
    devtools.blank.add("s-a")
    devtools.send("Page.javascriptDialogOpening", {"type": "alert", "message": "Saved"}, session="s-a")

    assert before == "png of s-a" and await control.capture() == "png of s-a"
    assert len(devtools.timeouts) == 1


@pytest.mark.asyncio
async def test_the_file_chooser_a_page_opens_takes_files_in_uploads_only_and_as_many_as_it_allows(tmp_path,
                                                                                                    monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "Uploads" / "docs").mkdir(parents=True)
    for name in ("cv.pdf", "docs/letter.pdf"):
        (tmp_path / "Uploads" / name).write_text("x")
    (tmp_path / "secret.txt").write_text("x")
    devtools = Tabbed()
    control = await _tabs(devtools, "a")
    with pytest.raises(InvalidArgument, match="no file chooser is open"):
        await control.choose_files(["cv.pdf"])
    devtools.send("Page.fileChooserOpened", {"frameId": "f", "mode": "selectSingle", "backendNodeId": 42},
                  session="s-a")

    state = await control.state()
    for paths, refused in ((["../secret.txt"], "isn't in ~/Uploads"), (["gone.pdf"], "no file gone.pdf"),
                           (["cv.pdf", "docs/letter.pdf"], "takes one file")):
        with pytest.raises(InvalidArgument, match=refused):
            await control.choose_files(paths)
    await control.choose_files(["docs/letter.pdf"])

    assert state["file_chooser"] == {"multiple": False}
    assert _sent(devtools, "DOM.setFileInputFiles") == [
        ({"files": [str((tmp_path / "Uploads" / "docs" / "letter.pdf").resolve())], "backendNodeId": 42}, "s-a")]
    assert "file_chooser" not in await control.state()


@pytest.mark.asyncio
async def test_with_no_tab_open_its_capture_and_input_are_unavailable():
    control = BrowserControl(Painting(), "agent")
    await control.attach()

    with pytest.raises(DeviceUnavailable, match="no tab open"):
        await control.capture()
    with pytest.raises(DeviceUnavailable, match="no tab open"):
        await control.dispatch([])


def test_its_declaration_says_it_is_a_website_browser_and_another_apps_says_nothing():
    assert Browser.declaration().browser is True and Mail.declaration().browser is None


@pytest.mark.asyncio
async def test_its_record_reads_each_tabs_title_when_read_and_when_its_page_loads_since_no_event_says(tmp_path):
    now = {"a": "Inbox", "b": "Draft"}
    devtools = StubDevTools({"Target.getTargets": lambda params, session: {"targetInfos": [
        _page(target, f"https://mail.test/{target}", title)["targetInfo"] for target, title in now.items()]}})
    control = BrowserControl(devtools, "agent")
    await control.attach()
    for target in ("a", "b"):
        devtools.send("Target.targetCreated", _page(target, f"https://mail.test/{target}", f"mail.test/{target}"))
        devtools.send("Target.attachedToTarget", {"sessionId": f"s-{target}", **_page(target)})
    devtools.send("Page.loadEventFired", {"timestamp": 1}, session="s-b")
    await asyncio.sleep(0)
    del now["b"]
    devtools.send("Target.targetDestroyed", {"targetId": "b"})
    now["a"] = "Inbox (1)"
    record = await control.record()

    assert [(tab["id"], tab["title"]) for tab in record["tabs"]] == [("a", "Inbox (1)")]
    assert [(tab["id"], tab["title"]) for tab in record["closed_tabs"]] == [("b", "Draft")]
    assert ("Target.getTargets", {"filter": [{"type": "page"}]}, None) in devtools.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("dialogs, kind, sent, answer, by", [
    ("accept", "prompt", {"accept": True, "promptText": "yes"}, "accepted", "agentenv"),
    ("dismiss", "prompt", {"accept": False, "promptText": "yes"}, "dismissed", "agentenv"),
    ("dismiss", "beforeunload", {"accept": True, "promptText": "yes"}, "accepted", "agentenv"),
    ("agent", "prompt", None, "dismissed", "agent"),
], ids=["accept", "dismiss", "dismiss-leaves", "agent"])
async def test_its_control_answers_a_dialog_on_its_pages_session_as_its_setting_says_and_records_who_did(
        tmp_path, dialogs, kind, sent, answer, by):
    devtools = StubDevTools()
    control = BrowserControl(devtools, dialogs)
    devtools.send("Target.targetCreated", _page("a"))
    devtools.send("Target.attachedToTarget", {"sessionId": "s-a", **_page("a")})

    devtools.send("Page.javascriptDialogOpening", {"type": kind, "message": "Sure?", "url": "https://a.test/",
                                                   "defaultPrompt": "yes"}, "s-a")
    await asyncio.sleep(0)
    devtools.send("Page.javascriptDialogClosed", {"result": answer == "accepted"}, "s-a")

    assert ("Page.enable", {}, "s-a") in devtools.calls
    handled = [(params, session) for method, params, session in devtools.calls
               if method == "Page.handleJavaScriptDialog"]
    assert handled == ([(sent, "s-a")] if sent else [])
    (dialog,) = control.dialog_log
    assert (dialog["tab"], dialog["type"], dialog["message"], dialog["answer"], dialog["by"]) == (
        "a", kind, "Sure?", answer, by)


@pytest.mark.asyncio
async def test_its_control_records_a_completed_download_with_its_size_and_sha256(tmp_path):
    devtools = StubDevTools()
    control = BrowserControl(devtools, "agent")
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.7 report")

    devtools.send("Browser.downloadWillBegin", {"guid": "g", "url": "https://a.test/report.pdf",
                                                "suggestedFilename": "report.pdf"})
    devtools.send("Browser.downloadProgress", {"guid": "g", "state": "inProgress", "receivedBytes": 3})
    in_progress = dict(control.downloads["g"])
    devtools.send("Browser.downloadProgress", {"guid": "g", "state": "completed",
                                               "filePath": str(tmp_path / "report.pdf")})
    for _ in range(int(WAIT_S / POLL_S)):
        if "sha256" in control.downloads["g"]:
            break
        await asyncio.sleep(POLL_S)

    done = control.downloads["g"]
    assert in_progress["state"] == "inProgress" and "sha256" not in in_progress
    assert (done["state"], done["size"], done["path"]) == ("completed", 15, str(tmp_path / "report.pdf"))
    assert done["sha256"] == hashlib.sha256(b"%PDF-1.7 report").hexdigest()


@pytest.mark.asyncio
async def test_its_control_records_ended_at_when_download_hashing_fails(tmp_path):
    devtools = StubDevTools()
    control = BrowserControl(devtools, "agent")
    missing_file = tmp_path / "nonexistent.pdf"

    devtools.send("Browser.downloadWillBegin", {"guid": "g2", "url": "https://a.test/missing.pdf",
                                                "suggestedFilename": "missing.pdf"})
    devtools.send("Browser.downloadProgress", {"guid": "g2", "state": "completed",
                                               "filePath": str(missing_file)})
    for _ in range(int(WAIT_S / POLL_S)):
        if "ended_at" in control.downloads.get("g2", {}):
            break
        await asyncio.sleep(POLL_S)

    done = control.downloads["g2"]
    assert done["state"] == "completed"
    assert "ended_at" in done
    assert "sha256" not in done


@pytest.mark.asyncio
async def test_its_control_says_the_tab_whose_page_has_focus_is_active_else_the_first_visible_one(tmp_path):
    states = {"s-a": ["hidden", False], "s-b": ["visible", False], "s-c": ["visible", True]}
    devtools = StubDevTools({"Runtime.evaluate": lambda params, session: {"result": {"value": states[session]}}})
    control = BrowserControl(devtools, "agent")
    for target in ("a", "b", "c"):
        devtools.send("Target.targetCreated", _page(target))
        devtools.send("Target.attachedToTarget", {"sessionId": f"s-{target}", **_page(target)})

    focused = await control.active_tab()
    states["s-c"] = ["hidden", False]
    visible = await control.active_tab()

    assert (focused, visible) == ("c", "b")


@pytest.mark.asyncio
async def test_its_connection_hears_events_its_server_logs_each_launch_and_an_exit_it_didnt_ask_for(monkeypatch,
                                                                                                    tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")
    browser = FakeBrowser()
    runtime = browser._computer_app

    first = await runtime.launch_app("boot")
    await _answering(browser)
    record = await browser.browser_record()
    await runtime.stop_app()
    second = await runtime.launch_app("data/reset")
    await _answering(browser)
    os.killpg(second, signal.SIGKILL)
    for _ in range(int(WAIT_S / POLL_S)):
        if any(entry["event"] == "exit" for entry in runtime.launches()):
            break
        await asyncio.sleep(POLL_S)
    after_exit = await browser.browser_record()
    browser.devtools.close()

    assert [tab["id"] for tab in record["tabs"]] == ["start"]
    assert after_exit["tabs"] == [] and [(tab["id"], tab["closed_by"]) for tab in after_exit["closed_tabs"]] == [
        ("start", "stop"), ("start", "exit")]
    assert [(entry["event"], entry.get("cause"), entry["pid"]) for entry in runtime.launches()] == [
        ("launch", "boot", first), ("stop", None, first), ("launch", "data/reset", second), ("exit", None, second)]
    assert runtime.launches()[-1]["signal"] == "SIGKILL"
