"""One turn of browser-use 1.0, run in its own environment: reads the turn as JSON on stdin and prints its result as
one JSON line on stdout.

The model is the turn's, on agent-env's endpoint. A Claude model goes through the endpoint's Anthropic Messages route,
as browser-use runs Claude (its reasoning as adaptive thinking at the turn's effort, its prompt cached), with the
profile of its bare name, since a provider prefix hides the model's thinking and effort from it. Any other model goes
through the OpenAI-compatible route, with the turn's effort as ``reasoning_effort``. The browser is the turn's CDP
address: when it has one tab open, as a browser computer does, the agent drives that tab, which is the computer's
screen, else a tab of its own. Each env's MCP tools are the agent's tools beside its REPL, named ``<server>__<tool>``.
A context's turns continue one conversation through the turn's history file. A turn that builds an atif/1 copy also
reports what the conversation doesn't hold: each model request and its answer, each compaction's summary, and each
screenshot browser-use pruned from the conversation."""

import asyncio
import contextlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx2
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient as AnthropicHttpxClient
from browser_use import Agent, Browser, Tools
from browser_use.agent import loop as browser_loop
from browser_use.browser.browser import _daemon_request
from browser_use.tools import TOOL_RETRIES
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import AsyncOpenAI, DefaultAsyncHttpxClient as OpenAIHttpxClient
from pydantic import TypeAdapter
from pydantic_ai import ModelRetry, Tool
from pydantic_ai.messages import BinaryContent, ModelMessagesTypeAdapter, ModelRequest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

TOOL_NAME = re.compile(r"[^a-zA-Z0-9_-]")
_CHECKPOINT = TypeAdapter(ModelRequest | None)
# The model requests the turn sent, and those the endpoint refused (an error answer bills nothing): a request cut off
# before its response, by the run's timeout or a retry, has no usage in the run's own count.
CALLS = {"sent": 0, "refused": 0}
# For an atif/1 copy: each model request, when it was sent, and when and how the endpoint answered it, if it did; each
# compaction's summarizer response; and each screenshot browser-use pruned, by the call that returned it.
ATTEMPTS: list[dict] = []
SUMMARIES: list[dict] = []
PRUNED: dict[str, list[dict]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def _sent(request) -> None:
    CALLS["sent"] += 1
    ATTEMPTS.append({"started": _now()})


async def _answered(response) -> None:
    if response.status_code >= 400:
        CALLS["refused"] += 1
    # The run makes one model request at a time, so a response answers the last one sent.
    ATTEMPTS[-1].update(status=response.status_code, timestamp=_now())


HOOKS = {"request": [_sent], "response": [_answered]}


def _api_root(url: str) -> str:
    """``LITELLM_BASE_URL`` without a trailing ``/`` and one trailing ``/v1``: it may be configured either way, and
    Anthropic's client appends its own ``/v1/messages``. Core's ``api_root``, copied since this image can't import
    core; a unit test keeps the copies equal."""
    parts = urlsplit(url.strip())
    return urlunsplit(parts._replace(path=parts.path.rstrip("/").removesuffix("/v1").rstrip("/")))


def model(turn: dict):
    """The turn's model on agent-env's endpoint, with the endpoint's headers."""
    name, base_url, key = turn["model"], os.environ["LITELLM_BASE_URL"], os.environ["LITELLM_API_KEY"]
    headers = json.loads(os.environ.get("LITELLM_EXTRA_HEADERS") or "{}")
    if "claude" in name.lower():
        client = AsyncAnthropic(auth_token=key, base_url=_api_root(base_url), default_headers=headers,
                                http_client=AnthropicHttpxClient(event_hooks=HOOKS))
        return AnthropicModel(name, provider=AnthropicProvider(anthropic_client=client),
                              profile=anthropic_model_profile(name.rsplit("/", 1)[-1]))
    client = AsyncOpenAI(api_key=key, base_url=base_url, default_headers=headers,
                         http_client=OpenAIHttpxClient(event_hooks=HOOKS))
    settings = {"openai_reasoning_effort": turn["effort"]} if turn["effort"] else None
    return OpenAIChatModel(name, provider=OpenAIProvider(openai_client=client), settings=settings)


async def mcp_tools(stack: contextlib.AsyncExitStack, servers: dict) -> Tools:
    """Each server's MCP tools as the agent's, on sessions open for the turn."""
    tools, names = Tools(), set()
    for server, info in servers.items():
        http = await stack.enter_async_context(httpx2.AsyncClient(headers=info.get("headers") or {}, timeout=None))
        streams = await stack.enter_async_context(streamable_http_client(info["url"], http_client=http))
        session = await stack.enter_async_context(ClientSession(*streams))
        await session.initialize()
        cursor = None
        while True:
            page = await session.list_tools(cursor=cursor) if cursor else await session.list_tools()
            for tool in page.tools:
                base = TOOL_NAME.sub("_", f"{server}__{tool.name}")[:60]
                name, n = base, 2
                while name in names:
                    name, n = f"{base}_{n}", n + 1
                names.add(name)
                function = Tool.from_schema(_caller(session, tool.name), name, tool.description or "", tool.inputSchema)
                # As browser-use's own actions: an error goes back to the model, and max_steps bounds the run.
                function.max_retries = TOOL_RETRIES
                tools.functions.append(function)
            if not (cursor := page.nextCursor):
                break
    return tools


def _caller(session: ClientSession, tool: str):
    async def call(**arguments):
        result = await session.call_tool(tool, arguments)
        text = "\n".join(getattr(part, "text", None) or json.dumps(part.model_dump(mode="json"))
                         for part in result.content)
        if result.isError:
            raise ModelRetry(text)
        return text
    return call


async def drive_its_tab(browser: Browser) -> None:
    """Attach browser-use to the browser's one open tab, as it attaches a kept-alive run's tab, closing the blank one
    its daemon opened for itself. The daemon closes that tab of its own when it stops, so a run in it would leave the
    screen blank, and its last page out of reach, when the turn ends."""
    await browser.start()

    def attach() -> None:
        own = _daemon_request(browser.name, {"meta": "current_tab"}, timeout=5)["targetId"]
        pages = [t for t in browser._cdp()("Target.getTargets")["targetInfos"]
                 if t["type"] == "page" and t["targetId"] != own]
        if len(pages) == 1:
            browser._agent_tab = pages[0]["targetId"]
            browser._attach_agent_tab()

    await asyncio.to_thread(attach)


def keep_for_atif() -> None:
    """Have browser-use keep what its conversation doesn't hold: a compaction's summary, which its own call writes
    outside it, and each screenshot, which it prunes from it before each request, keeping the last two results'."""
    compact, prune = browser_loop._compact, browser_loop._prune_images

    async def compact_keeping(session, messages, request_context):
        summaries, covered = len(session.summaries), session.covered
        try:
            await compact(session, messages, request_context)
        finally:
            SUMMARIES.extend({"compacted": session.covered != covered,
                              "response": ModelMessagesTypeAdapter.dump_python([summary], mode="json")[0]}
                             for summary in session.summaries[summaries:])

    def prune_keeping(messages):
        pruned = prune(messages)
        for before, after in zip(messages, pruned or ()):
            for old, new in zip(before.parts, after.parts):
                if old is not new:
                    PRUNED[old.tool_call_id] = [{"index": index, "media_type": item.media_type, "data": item.base64}
                                                for index, item in enumerate(old.content)
                                                if isinstance(item, BinaryContent) and item.is_image]
        return pruned

    browser_loop._compact, browser_loop._prune_images = compact_keeping, prune_keeping


def restore(agent: Agent, history: Path) -> None:
    """Continue the conversation a context's earlier turns saved, as ``add_new_task`` continues one in process."""
    saved = json.loads(history.read_text())
    agent._history = ModelMessagesTypeAdapter.validate_python(saved["messages"])
    agent._compacted = {**saved["compacted"], "checkpoint": _CHECKPOINT.validate_python(saved["compacted"]["checkpoint"])}
    agent._continue = True


def save(agent: Agent, history: Path) -> None:
    compacted = {**agent._compacted, "checkpoint": _CHECKPOINT.dump_python(agent._compacted.get("checkpoint"),
                                                                          mode="json")}
    history.write_text(json.dumps({"messages": ModelMessagesTypeAdapter.dump_python(agent._history or [], mode="json"),
                                   "compacted": compacted}))


async def main() -> None:
    turn = json.loads(sys.stdin.read())
    if turn["atif"]:
        keep_for_atif()
    history = Path(turn["historyFile"])
    async with contextlib.AsyncExitStack() as stack:
        browser = Browser(cdp_url=turn["cdpUrl"])
        with contextlib.suppress(Exception):  # a browser it can't reach: run() says why, as its result
            await drive_its_tab(browser)
        agent = Agent(
            turn["prompt"], model(turn), browser=browser,
            tools=await mcp_tools(stack, turn["mcpServers"]), extend_system_message=turn["systemPrompt"],
            reasoning=turn["effort"] if "claude" in turn["model"].lower() else None, mode=turn["mode"],
            workspace=turn["workspace"])
        stack.push_async_callback(agent.close)
        if history.exists():
            restore(agent, history)
        # The run's messages are the conversation's; the turn's own start here, as browser-use slices its usage.
        prior = len(agent._history or [])
        result = await agent.run(**{key: turn[key] for key in ("max_steps", "timeout") if turn.get(key)})
        if agent._history:
            save(agent, history)
    print(json.dumps({"type": "result", "result": {
        "status": result.status, "text": result.text, "success": result.success, "error": result.error,
        "steps": result.steps, "usage": result.usage, "cost_usd": result.cost_usd, "model": turn["model"],
        "messages": ModelMessagesTypeAdapter.dump_python(result.messages, mode="json"), "prior": prior,
        "calls": CALLS,
        **({"atif": {"attempts": ATTEMPTS, "summaries": SUMMARIES, "images": PRUNED}} if turn["atif"] else {}),
    }}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
