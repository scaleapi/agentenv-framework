"""What the atif/1 recorders share (``claude_code_atif/record.py``, ``codex_gemini_atif/record.py``): an agent's image
built from this checkout as a release builds it, its A2A turns sent and read back with their live atif/1 pages, and the
model key kept out of everything they write."""
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SECRET = os.environ.get("LITELLM_API_KEY", "")
TERMINAL = {"completed", "failed", "canceled", "rejected", "input-required", "auth-required"}


def mask(text: str) -> str:
    return text.replace(SECRET, "***") if SECRET else text


def run(command: list[str], **kw) -> subprocess.CompletedProcess:
    done = subprocess.run(command, capture_output=True, text=True, **kw)
    done.stdout, done.stderr = mask(done.stdout), mask(done.stderr)
    return done


def http(method: str, url: str, body=None, timeout: float = 60):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw or b"null")
        except ValueError:
            return exc.code, mask(raw.decode(errors="replace"))[:300]
    except (urllib.error.URLError, OSError) as exc:
        return 0, str(exc)


def save(path: Path, value) -> None:
    path.write_text(mask(json.dumps(value, indent=1, ensure_ascii=False)))


def build(image_name: str) -> str:
    """The agent's image, built from this checkout for this machine's architecture."""
    planned = run([sys.executable, ".github/scripts/infra_images.py", "plan", "--only", "agents"], cwd=REPO)
    entry = next(e for e in json.loads(planned.stdout) if e["name"] == image_name)
    arch = "arm64" if platform.machine() in ("arm64", "aarch64") else "amd64"
    env = {**os.environ, "GITHUB_REPOSITORY": "scaleapi/agentenv-framework", "GITHUB_SHA": "recording"}
    built = run([sys.executable, ".github/scripts/infra_build.py", "build", json.dumps(entry), arch], cwd=REPO, env=env)
    if built.returncode != 0:
        raise SystemExit(f"the image didn't build:\n{built.stdout[-3000:]}\n{built.stderr[-3000:]}")
    return f"{entry['repository']}:{entry['tag']}-{arch}"


def start(command: list[str]) -> None:
    started = run(command)
    if started.returncode != 0:
        raise RuntimeError(started.stderr[-500:])


def turn(base: str, context_id: str, prompt: str) -> tuple[str, dict, list]:
    message = {"messageId": uuid.uuid4().hex, "role": "user", "contextId": context_id,
               "parts": [{"kind": "text", "text": prompt}]}
    status, sent = http("POST", f"{base}/a2a", {"jsonrpc": "2.0", "id": "1", "method": "message/send",
                                                "params": {"message": message, "configuration": {"blocking": False}}})
    if status != 200 or "error" in (sent or {}):
        raise RuntimeError(f"message/send {status} {sent}")
    task_id, pages, deadline = sent["result"]["id"], [], time.time() + 900
    while time.time() < deadline:
        status, page = http("POST", f"{base}/ext/trajectory", {"task_id": task_id, "format": "atif/1", "after": 0})
        if status == 200:
            pages.append(page)
        status, polled = http("POST", f"{base}/a2a", {"jsonrpc": "2.0", "id": "p", "method": "tasks/get",
                                                      "params": {"id": task_id}})
        if status == 200 and "result" in polled and polled["result"]["status"]["state"] in TERMINAL:
            return task_id, polled["result"], pages
        time.sleep(1.0)
    raise RuntimeError("the turn didn't end in 15 minutes")
