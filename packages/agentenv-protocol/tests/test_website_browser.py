"""WebsiteBrowserEnvironment: agentenv's flags after a subclass's command, in a computer and alone, and the proxy rule
for each; a subclass refused for carrying one of them; the start tabs, the time zone and the window from its settings;
its launch run with no desktop when alone; a CDP connection over a new debugging pipe at each launch, which its ready
check asks, and a browser launched again when its server restarts; its shared memory in files only on a small
/dev/shm; an empty cache and the deploy's prompt answers in its profile at each launch; its policies, card params and
skill; and an app that isn't a browser launched and declared as before."""
from __future__ import annotations

import asyncio
import json
import os
import sys

import pytest

from agentenv_protocol.computer import APP_SETTINGS_ENV, ComputerAppEnvironment, Launch, Ready, app, linux, session
from agentenv_protocol import website_browser
from agentenv_protocol.computer import app_runtime
from agentenv_protocol.computer.platform import Platform
from agentenv_protocol.website_browser import (ALONE_BYPASS, ALONE_ENV, BROWSER, COMPUTER_BYPASS, OPEN_WEB_ENV,
                                               POLICIES, SITE_PROXY_ENV, SKILL, DevToolsError,
                                               WebsiteBrowserEnvironment)

PROXY = "http://127.0.0.1:3128"
OWNED = ["--user-data-dir", "--disk-cache-dir", "--remote-debugging-pipe", "--no-first-run",
         "--no-default-browser-check", "--lang=en-US"]
# A browser that answers CDP on its debugging pipe, fds 3 and 4, as a Chromium does: Browser.getVersion with its
# product, Target.createTarget by writing the tab's URL to tabs.txt beside it, a new window's after "window ", anything
# else with an error.
FAKE_BROWSER = """import json, os
buffer = b""
while chunk := os.read(3, 65536):
    buffer += chunk
    while b"\\0" in buffer:
        raw, buffer = buffer.split(b"\\0", 1)
        message = json.loads(raw)
        if message["method"] == "Browser.getVersion":
            answer = {"result": {"product": "Fake/1"}}
        elif message["method"] == "Target.createTarget":
            with open("tabs.txt", "a") as tabs:
                tabs.write(("window " if message["params"].get("newWindow") else "") + message["params"]["url"] + "\\n")
            answer = {"result": {"targetId": message["params"]["url"]}}
        else:
            answer = {"error": {"code": -32601, "message": "unknown method"}}
        os.write(4, json.dumps({"id": message["id"], **answer}).encode() + b"\\0")
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
    monkeypatch.setattr(website_browser, "_shm_bytes", lambda path="/dev/shm": 1 << 30)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("ENVIRONMENT_NAME", "chromium")
    for name in (ALONE_ENV, OPEN_WEB_ENV, SITE_PROXY_ENV, APP_SETTINGS_ENV):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "home"


def _settings(monkeypatch, **values) -> None:
    monkeypatch.setenv(APP_SETTINGS_ENV, json.dumps({"chromium": values}))


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
    monkeypatch.setenv(ALONE_ENV, "1")
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

    monkeypatch.setenv(ALONE_ENV, "1")
    monkeypatch.setenv(SITE_PROXY_ENV, PROXY)

    waiting = Waiting()

    assert waiting.launch.ready == Ready(window=False, port=9000, command=["true"], timeout_s=45)
    assert Launch.model_validate(waiting.launch.model_dump()) == waiting.launch
    assert waiting._computer_app.timeout_s == 45


def test_alone_on_the_open_web_it_runs_headful(monkeypatch):
    monkeypatch.setenv(ALONE_ENV, "1")
    monkeypatch.setenv(OPEN_WEB_ENV, "1")
    monkeypatch.setenv(SITE_PROXY_ENV, PROXY)

    command = Browser().launch.command

    assert "--window-size=1280,800" in command and not any(flag.startswith("--headless") for flag in command)


def test_alone_it_refuses_to_launch_without_a_site_proxy(monkeypatch):
    monkeypatch.setenv(ALONE_ENV, "1")

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
    monkeypatch.setattr(website_browser, "_shm_bytes", lambda path="/dev/shm": mib * 2**20)

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


def test_its_settings_give_its_start_tabs_language_and_time_zone_and_no_start_url_is_a_flag(monkeypatch):
    _settings(monkeypatch, start_urls=["https://slack.com", "https://linear.app"], locale="fr-FR",
              time_zone="Europe/Paris")
    launch = Browser().launch
    _settings(monkeypatch, start_urls=["--remote-allow-origins=*"])

    assert launch.command[-2:] == ["https://slack.com", "https://linear.app"] and "--lang=fr-FR" in launch.command
    assert launch.env == {"TZ": "Europe/Paris"}
    with pytest.raises(ValueError, match="start_urls are URLs, not flags"):
        Browser()


@pytest.mark.asyncio
async def test_alone_its_server_launches_it_at_once_in_its_environment_with_no_display_wait_or_launcher(
        monkeypatch, tmp_path):
    monkeypatch.setenv(ALONE_ENV, "1")
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
        await browser.devtools.call("Page.navigate", {"url": "about:blank"})
    launched = await browser.devtools_answers()
    await runtime.stop_app()
    stopped = await browser.devtools_answers()
    await runtime.launch_app()
    relaunched = await browser.devtools_answers()
    await runtime.stop_app()
    browser.devtools.close()

    assert unlaunched is False and version == {"product": "Fake/1"} and launched is True
    assert stopped is False and relaunched is True
    assert "devtools_answers" in runtime.ready_checks


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
    answers = await browser.devtools_answers()
    await runtime.stop_app()
    browser.devtools.close()

    assert second == first and answers is True and (browser_home / ".cache/chromium/kept").exists()
    assert (tmp_path / "tabs.txt").read_text().splitlines() == ["window about:blank"]


@pytest.mark.asyncio
async def test_headless_it_opens_one_start_tab_from_its_command_line_and_the_rest_over_its_pipe(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")
    monkeypatch.setenv(ALONE_ENV, "1")
    monkeypatch.setenv(SITE_PROXY_ENV, PROXY)
    _settings(monkeypatch, start_urls=["https://slack.com", "https://linear.app", "https://github.com"])
    browser = FakeBrowser()

    await browser._computer_app.launch_app()
    for _ in range(50):
        if ready := await browser.devtools_answers():
            break
        await asyncio.sleep(0.1)
    await browser._computer_app.stop_app()
    browser.devtools.close()

    assert browser.launch.command[-2:] == ["--headless=new", "https://slack.com"]
    assert ready is True and (tmp_path / "tabs.txt").read_text() == "https://linear.app\nhttps://github.com\n"


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
    in_computer = next(e for e in Browser().declared_extensions() if e.uri == BROWSER)
    monkeypatch.setenv(ALONE_ENV, "1")
    monkeypatch.setenv(SITE_PROXY_ENV, PROXY)
    alone = next(e for e in Browser().declared_extensions() if e.uri == BROWSER)

    assert in_computer.params == {"chromium": "156.0.8078.4", "clock": "real"}
    assert alone.params == {"chromium": "156.0.8078.4", "clock": "real", "window": {"width": 1280, "height": 800}}


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

