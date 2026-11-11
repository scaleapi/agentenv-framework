"""A browser's debugging pipe hands a request's ``answered`` its answer as it reads it, before the events the browser
wrote after it, even when both come in one read and the caller hasn't resumed yet."""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from agentenv_protocol.website_browser.devtools import DevToolsPipe


@pytest.mark.asyncio
async def test_an_answer_is_heard_before_the_events_written_after_it_in_the_same_read():
    from_browser, to_us = os.pipe()
    from_us, to_browser = os.pipe()
    pipe, heard = DevToolsPipe(), []
    pipe.listen(lambda event: heard.append(event["method"]))
    await pipe.connect(to_browser, from_browser)
    try:
        requested = asyncio.ensure_future(pipe.request("Page.getFrameTree", answered=lambda answer: heard.append("answer")))
        await asyncio.sleep(0)
        command = json.loads(os.read(from_us, 4096).rstrip(b"\0"))
        os.write(to_us, json.dumps({"id": command["id"], "result": {}}).encode() + b"\0"
                 + json.dumps({"method": "Runtime.executionContextCreated", "params": {}}).encode() + b"\0")
        answer = await asyncio.wait_for(requested, 5)
    finally:
        pipe.close()
        for fd in (to_us, from_us):
            os.close(fd)

    assert heard == ["answer", "Runtime.executionContextCreated"] and answer == {"id": command["id"], "result": {}}
