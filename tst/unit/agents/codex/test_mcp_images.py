"""The Codex CLI sends the model a tool result's structuredContent alone when it has one, so an MCP server's messages
reach it through mcp_images.py, which drops structuredContent from results that carry an image."""

import json
import subprocess
import sys

from tst.unit.agents.codex.agent import codex_config

SCREENSHOT = {"jsonrpc": "2.0", "id": 3, "result": {
    "content": [{"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"},
                {"type": "text", "text": "{\"tree_markdown\": \"- [0] frame \\\"Mail\\\"\"}"}],
    "structuredContent": {"tree_markdown": "- [0] frame \"Mail\""}}}
TEXT_ONLY = {"jsonrpc": "2.0", "id": 4, "result": {
    "content": [{"type": "text", "text": "{\"windows\": []}"}], "structuredContent": {"windows": []}}}
TOOLS = {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "zoom", "description": "an image of a region"}]}}


def test_a_result_with_an_image_reaches_codex_as_its_content_and_every_other_message_as_sent():
    lines = "".join(json.dumps(message) + "\n" for message in (TOOLS, SCREENSHOT, TEXT_ONLY))
    server = [sys.executable, "-c", f"import sys; sys.stdout.write({lines!r})"]

    done = subprocess.run([sys.executable, str(codex_config.MCP_IMAGES), *server], capture_output=True, text=True,
                          check=True, timeout=30)

    tools, screenshot, text_only = map(json.loads, done.stdout.splitlines())
    assert screenshot == {**SCREENSHOT, "result": {"content": SCREENSHOT["result"]["content"]}}
    assert (tools, text_only) == (TOOLS, TEXT_ONLY)


def test_what_codex_writes_reaches_the_server_and_its_exit_status_is_the_servers():
    server = [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read().upper()); sys.exit(3)"]

    done = subprocess.run([sys.executable, str(codex_config.MCP_IMAGES), *server], input="ping\n",
                          capture_output=True, text=True, timeout=30)

    assert (done.stdout, done.returncode) == ("PING\n", 3)
