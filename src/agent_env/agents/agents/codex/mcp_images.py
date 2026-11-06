"""mcp_images.py COMMAND [ARG...]: run a stdio MCP server for the Codex CLI and pass on what it says, each tool result
that carries an image without its structuredContent. The CLI sends the model a result's structuredContent alone when it
has one, so the image beside it would never reach the model; the result's content goes instead, as other clients send it."""

import json
import signal
import subprocess
import sys


def main() -> int:
    server = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE)
    signal.signal(signal.SIGTERM, lambda *_: server.terminate())
    out = sys.stdout.buffer
    for line in server.stdout:
        out.write(without_structured_content(line))
        out.flush()
    return server.wait()


def without_structured_content(line: bytes) -> bytes:
    if b'"structuredContent"' not in line or b'"image"' not in line:
        return line
    try:
        message = json.loads(line)
    except ValueError:
        return line
    result = message.get("result") if isinstance(message, dict) else None
    if not isinstance(result, dict) or not any(
            isinstance(part, dict) and part.get("type") == "image" for part in result.get("content") or ()):
        return line
    del result["structuredContent"]
    return json.dumps(message, ensure_ascii=False).encode() + b"\n"


if __name__ == "__main__":
    sys.exit(main())
