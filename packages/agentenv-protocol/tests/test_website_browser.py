"""WebsiteBrowserEnvironment: agentenv's flags after a subclass's command, in a computer and alone, and the proxy rule
for each; a subclass refused for carrying one of them; the start tabs, the time zone and the window from its settings;
its launch run with no desktop when alone; the debugging address read again each time and checked for readiness; its
policies, card params and skill; and an app that isn't a browser launched and declared as before."""
from __future__ import annotations

import asyncio
import json

import pytest

from agentenv_protocol.computer import APP_SETTINGS_ENV, ComputerAppEnvironment, Launch, Ready, app, linux, session
from agentenv_protocol.computer import app_runtime
from agentenv_protocol.computer.platform import Platform
from agentenv_protocol.website_browser import (ALONE_BYPASS, ALONE_ENV, BROWSER, COMPUTER_BYPASS, OPEN_WEB_ENV,
                                               POLICIES, SITE_PROXY_ENV, SKILL, WebsiteBrowserEnvironment, cdp)

PROXY = "http://127.0.0.1:3128"
OWNED = ["--user-data-dir", "--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check", "--lang=en-US"]


@linux(launch=Launch(command=["chromium/chrome", "--no-sandbox"], window={"class": "Chromium"},
                     ready=Ready(window=True, timeout_s=30)))
class Browser(WebsiteBrowserEnvironment):
    version = "156.0.8078.4"
    policies = {"DefaultGeolocationSetting": 2}


@linux(launch=Launch(command=["mail"], window={"class": "Mail"}, ready=Ready(window=True)))
class Mail(ComputerAppEnvironment):
    pass


@pytest.fixture(autouse=True)
def browser_home(monkeypatch, tmp_path):
    """A browser on Linux, its session user's home under ``tmp_path``, inside a computer with no site proxy."""
    monkeypatch.setattr(app, "detect", lambda: Platform(os="linux"))
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
    assert launch.command == ["chromium/chrome", "--no-sandbox", profile, *OWNED[1:], "about:blank"]
    assert launch.env == {"TZ": "UTC"}
    assert launch.window == {"class": "Chromium"} and launch.ready == Ready(window=True, timeout_s=30)


def test_in_a_computer_with_a_site_proxy_only_loopback_bypasses_it(monkeypatch):
    monkeypatch.setenv(SITE_PROXY_ENV, PROXY)

    command = Browser().launch.command

    assert [f"--proxy-server={PROXY}", f"--proxy-bypass-list={COMPUTER_BYPASS}",
            "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"] == command[7:10]
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
                                  "--proxy-server=http://elsewhere:3128", "--user-data-dir=/tmp/profile"])
def test_a_subclass_whose_command_carries_a_flag_agentenv_owns_is_refused_when_declared(flag):
    @linux(launch=Launch(command=["chromium/chrome", flag]))
    class Widened(WebsiteBrowserEnvironment):
        pass

    with pytest.raises(ValueError, match="agentenv sets for every website browser"):
        Widened.declaration()


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

    async def launch(argv, cwd, env):
        launched.append(env)
        return 4242

    async def up():
        waited.append(True)

    monkeypatch.setattr(session, "launch", launch)
    monkeypatch.setattr(session, "wait_until_up", up)
    monkeypatch.setattr(session, "add_launcher", lambda name, folder: launchers.append(name))
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")

    await Browser()._computer_app.start()

    assert launched == [{"TZ": "UTC", "ENVIRONMENT_NAME": "chromium"}] and waited == [] and launchers == []


@pytest.mark.asyncio
async def test_in_a_computer_its_server_waits_for_the_display_and_adds_its_launcher(monkeypatch, tmp_path):
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.chdir(tmp_path)
    waited, launchers = [], []

    async def launch(argv, cwd, env):
        return 4242

    async def up():
        waited.append(True)

    monkeypatch.setattr(session, "launch", launch)
    monkeypatch.setattr(session, "wait_until_up", up)
    monkeypatch.setattr(session, "add_launcher", lambda name, folder: launchers.append(name))
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")

    await Browser()._computer_app.start()

    assert waited == [True] and launchers == ["chromium"]


@pytest.mark.asyncio
async def test_its_debugging_address_is_read_again_each_time_and_answers_its_ready_check(browser_home):
    browser = Browser()
    profile = browser_home / ".config/chromium"
    profile.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        await browser.devtools_address()
    unwritten = await browser.debugging_address_answers()

    server = await asyncio.start_server(_json_version, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    (profile / "DevToolsActivePort").write_text("1\n/devtools/browser/first\n")
    first = await browser.devtools_address()
    (profile / "DevToolsActivePort").write_text(f"{port}\n/devtools/browser/second\n")
    async with server:
        relaunched = await browser.devtools_address()
        answers = await browser.debugging_address_answers()

    assert unwritten is False
    assert first == "ws://127.0.0.1:1/devtools/browser/first"
    assert relaunched == f"ws://127.0.0.1:{port}/devtools/browser/second" and answers is True
    assert "debugging_address_answers" in browser._computer_app.ready_checks


@pytest.mark.asyncio
async def test_a_subclass_that_marks_its_own_method_cdp_gives_that_address():
    class Elsewhere(Browser):
        @cdp
        async def own(self) -> str:
            return "ws://127.0.0.1:9333/devtools/browser/own"

    assert await Elsewhere().devtools_address() == "ws://127.0.0.1:9333/devtools/browser/own"


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
    assert stored["os"]["linux"]["launch"] == {"at": "boot", "command": ["mail"], "window": {"class": "Mail"},
                                               "ready": {"window": True, "timeout_s": 60.0}}


async def _json_version(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    await reader.readuntil(b"\r\n\r\n")
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}")
    await writer.drain()
    writer.close()
