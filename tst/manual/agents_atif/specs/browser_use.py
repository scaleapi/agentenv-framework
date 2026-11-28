"""Browser Use's spec for ``record.py``: Claude Haiku through the endpoint's Anthropic Messages route, the agent's
histories and workspaces (screenshots, compaction archives) under ``~/.browser-use-a2a``, and a headless Chromium on the
recording network that every scenario drives over CDP, as cdp-config/v1 gives it, never Playwright MCP. Its own
scenarios: two turns of one conversation on example.com, its screenshots returned to the model, more than two so
browser-use prunes the oldest; signed thinking at a high effort; a two-step budget, which ends in browser-use's
delivery-only turn; a run budget of one second, which cuts its request off; and the endpoint failing calls the SDK
retries. Chrome answers DevTools only to a Host that is an IP or localhost, so the endpoint is the browser's address on
the network. The fault proxy appends the path to ``LITELLM_BASE_URL``'s, so record with it unsuffixed by ``/v1``, as the
Anthropic route calls ``/v1/messages``."""
import json

from tst.manual.atif_recording import run

#: record.py's network for this image, which its services join.
NETWORK = "atif-recording-agentenv-browser-use"
CHROMIUM = f"{NETWORK}-chromium"
#: chromedp's headless Chrome, serving DevTools on 9222 to the network.
HEADLESS = ("chromedp/headless-shell:158.0.8127.37"
            "@sha256:8c574e39f0db5c1d9efe7c472c6e0fcfc1d91ac0be70eb1dead0f2d52cf104cc")
PYTHON = "public.ecr.aws/docker/library/python:3.12-slim"
EXAMPLE = ("Open https://example.com in the browser, take a screenshot so the image comes back to you, then take two "
           "more, one after scrolling down and one after scrolling back up, and reply with the main heading you see.")


def services() -> dict[str, list[str]]:
    return {CHROMIUM: [HEADLESS]}


def posts(scenario: dict) -> list[tuple[str, dict]]:
    """The browser for every scenario: its address on the network, and the WebSocket its DevTools gives, read from
    a container on the network, since this machine may not reach the network's addresses."""
    address = run(["docker", "inspect", "-f", f'{{{{(index .NetworkSettings.Networks "{NETWORK}").IPAddress}}}}',
                   CHROMIUM]).stdout.strip()
    version = run(["docker", "run", "--rm", "--network", NETWORK, PYTHON, "python3", "-c",
                   f"import urllib.request; print(urllib.request.urlopen('http://{address}:9222/json/version').read()"
                   ".decode())"])
    return [("/ext/cdp-config", {"name": "chromium", "http_url": f"http://{address}:9222/",
                                 "ws_url": json.loads(version.stdout)["webSocketDebuggerUrl"]})]


SPEC = {
    "image": "agentenv-browser-use",
    "model": "anthropic/claude-haiku-4-5",
    "files": "/home/agent/.browser-use-a2a",
    "version": ["/opt/browser-use/bin/python", "-c", "import importlib.metadata as m; print(m.version('browser-use'))"],
    "scenarios": ["refused"],
    "services": services,
    "posts": posts,
    "own": {
        "browse": {"prompts": [EXAMPLE, "Now follow the page's only link, and reply with the new page's main heading."]},
        "thinking": {"config": {"effort": "high"}, "prompts": [
            "Open https://example.com, think about what the page is for, and reply with one sentence saying so."]},
        "delivery": {"config": {"max_steps": 2}, "prompts": [EXAMPLE]},
        "cut": {"config": {"timeout_seconds": 101}, "prompts": ["Reply with ok."]},
        "retried": {"faults": "retry", "prompts": [EXAMPLE]},
    },
}
