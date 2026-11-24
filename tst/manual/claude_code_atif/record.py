"""Record the Claude Code ATIF fixtures (``tst/data/claude_code_atif``) again, as after a bump of the pinned CLI, which
``test_claude_code_atif.py``'s ``tested_with`` check asks for. Run by hand from a checkout with agent-env installed:

    LITELLM_BASE_URL=... LITELLM_API_KEY=... python tst/manual/claude_code_atif/record.py OUT [--only coding,...]
    python tst/manual/claude_code_atif/make_fixtures.py OUT
    python tst/unit/agents/claude_code/test_claude_code_atif_recorded.py

It builds the agent's image from the checkout as a release builds it (``infra_build.py``), and runs each scenario in a
container of it with atif/1 prescribed, through a model endpoint serving Anthropic's Messages API. For each turn it
writes the native final read, the atif/1 final read (document, aux log and turn context), the atif/1 pages read live
every second, and the CLI's transcripts after. The scenarios: two coding turns of one conversation, an Agent call, a
browser's screenshots through Playwright MCP beside the agent, a thinking model, a conversation compacted between its
turns (``/compact``), one compacted partway through a turn as its history fills (a small auto-compact window), the
endpoint refusing every call, and a stalled call that the bridge resumes, the last two through
``fault_proxy.py`` in a container of their own. They cost about a dollar. Needs Docker;
``mcr.microsoft.com/playwright/mcp`` is pulled for the browser scenario.

The key is never written: every text is checked for it, and a transcript holding it is dropped. Then
``make_fixtures.py`` writes the fixtures, keeping each request shape's tools once under ``tools/``, and running the
recorded test as a script writes the expected documents. Note the CLI version in ``claude_code_atif.TESTED_WITH``."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SECRET = os.environ.get("LITELLM_API_KEY", "")
NETWORK = "claude-atif-recording"
BROWSER = "claude-atif-playwright"
FAULTS = "claude-atif-faults"
PLAYWRIGHT_MCP = ("mcr.microsoft.com/playwright/mcp:v0.0.86"
                  "@sha256:2dfc17e08e8db54678ec965ce9ffb4100be0aa68135774d761bac9035ceb6668")
FAULT_PORTS = {"refuse": 18401, "stall": 18402}
# Anthropic's documented string that has a thinking model answer with a redacted thinking block. Through the CLI,
# Sonnet 5.5 and 4.5 return omitted thinking for it instead, so the scenario records a thinking model's turn.
REDACTED_THINKING = ("ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_"
                     "46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB")
TERMINAL = {"completed", "failed", "canceled", "rejected", "input-required", "auth-required"}


def scenarios(thinking_model: str) -> dict[str, dict]:
    return {
        "coding": {"prompts": [
            "Create hello.py that prints the sum of the integers 1 to 100, run it with python3, and reply with only "
            "the number it prints.",
            "Now change it to sum 1 to 1000, run it again, and reply with only the number."]},
        "subagent": {"prompts": [
            "Use the Agent tool to have a sub-agent run `wc -l /etc/passwd` and report the count. Then reply with "
            "only that number."]},
        "browser": {"browser": True, "prompts": [
            "Use the browser tools to open https://example.com, take a screenshot of the page, and reply with the "
            "page's main heading."]},
        "thinking": {"model": thinking_model, "config": {"max_thinking_tokens": 2048},
                     "prompts": [REDACTED_THINKING + "\n\nThink about it, then reply with the word ok."]},
        "api-error": {"faults": "refuse", "env": {"CLAUDE_API_ERROR_RETRIES": "0"}, "prompts": ["Reply with ok."]},
        "retried": {"faults": "stall", "env": {"CLAUDE_API_ERROR_RETRIES": "1", "API_TIMEOUT_MS": "15000"},
                    "prompts": ["Run `echo first` with your Bash tool, then run `echo second` with it, then reply "
                                "with both outputs on one line."]},
        "autocompacted": {"env": {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "50000"}, "prompts": [
            "Create big.txt holding 3000 numbered lines of filler text with one Bash command, then read it whole with "
            "your Read tool, 500 lines at a time, and reply with only its last line."]},
        "compacted": {"prompts": [
            "Create notes.txt holding three lines, alpha, beta and gamma, with your Bash tool, then reply with only ok.",
            "/compact",
            "Read notes.txt with your tools and reply with only its second line."]},
    }


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


def build() -> str:
    """The agent's image, built from this checkout for this machine's architecture."""
    planned = run([sys.executable, ".github/scripts/infra_images.py", "plan", "--only", "agents"], cwd=REPO)
    entry = next(e for e in json.loads(planned.stdout) if e["name"] == "agentenv-claude-code")
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


def start_agent(image: str, name: str, scenario: dict) -> tuple[str, dict]:
    endpoint = (f"http://{FAULTS}:{FAULT_PORTS[scenario['faults']]}" if "faults" in scenario
                else os.environ["LITELLM_BASE_URL"])
    env = ["-e", "A2A_PORT=8000", "-e", "LITELLM_API_KEY", "-e", f"LITELLM_BASE_URL={endpoint}"]
    for key, value in scenario.get("env", {}).items():
        env += ["-e", f"{key}={value}"]
    run(["docker", "rm", "-f", name])
    start(["docker", "run", "-d", "--name", name, "--network", NETWORK, "-p", "127.0.0.1::8000", *env, image])
    base = f"http://{run(['docker', 'port', name, '8000']).stdout.splitlines()[0]}"
    for _ in range(90):
        status, card = http("GET", f"{base}/.well-known/agent.json", timeout=5)
        if status == 200:
            return base, card
        time.sleep(1)
    raise RuntimeError(f"no card: {run(['docker', 'logs', '--tail', '40', name]).stdout[-2000:]}")


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


def record(image: str, name: str, scenario: dict, model: str, out: Path) -> None:
    folder = out / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    container = f"claude-atif-{name}"
    try:
        base, _ = start_agent(image, container, scenario)
        config = {"model": scenario.get("model", model), "timeout_seconds": 600, **scenario.get("config", {})}
        for path, body in (("/ext/agent-config", config),
                           *((("/ext/mcp-config", {"url": f"http://{BROWSER}:8931/mcp", "name": "browser"}),)
                             if scenario.get("browser") else ()),
                           ("/ext/trajectory/prescribe", {"formats": ["atif/1"]})):
            status, answer = http("POST", base + path, body)
            if status >= 300:
                raise RuntimeError(f"{path}: {status} {answer}")
        for index, prompt in enumerate(scenario["prompts"]):
            started = time.time()
            task_id, task, pages = turn(base, f"ctx-{name}", prompt)
            _, native = http("POST", f"{base}/ext/trajectory", {"task_id": task_id})
            _, atif = http("POST", f"{base}/ext/trajectory", {"task_id": task_id, "format": "atif/1"})
            save(folder / f"turn-{index}.json", {"prompt": prompt, "task": task, "native": native, "atif": atif,
                                                 "live_pages": pages})
            print(f"{name} turn {index}: {task['status']['state']} in {int(time.time() - started)} s, "
                  f"{len(pages)} live pages", flush=True)
    finally:
        projects = folder / "projects"
        run(["docker", "cp", f"{container}:/home/agent/.claude/projects", str(projects)])
        for path in projects.rglob("*") if projects.exists() else ():
            if path.is_file() and SECRET and SECRET.encode() in path.read_bytes():
                print(name, "dropped a transcript file holding the key:", path.name, flush=True)
                path.unlink()
        logs = run(["docker", "logs", "--tail", "300", container])
        (folder / "agent.log").write_text(logs.stdout + logs.stderr)
        run(["docker", "rm", "-f", container])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--only", help="comma-separated scenarios; all by default")
    parser.add_argument("--model", default="anthropic/claude-sonnet-5-5")
    parser.add_argument("--thinking-model", default="anthropic/claude-sonnet-4-5-20250929")
    args = parser.parse_args()
    if not SECRET or not os.environ.get("LITELLM_BASE_URL"):
        raise SystemExit("set LITELLM_BASE_URL and LITELLM_API_KEY")
    all_scenarios = scenarios(args.thinking_model)
    only = args.only.split(",") if args.only else list(all_scenarios)
    args.out.mkdir(parents=True, exist_ok=True)
    image = build()
    print("built", image, run(["docker", "run", "--rm", "--entrypoint", "claude", image, "--version"]).stdout.strip())
    run(["docker", "network", "create", NETWORK])
    try:
        start(["docker", "run", "-d", "--rm", "--name", FAULTS, "--network", NETWORK, "--entrypoint", "python3",
               "-e", f"UPSTREAM={os.environ['LITELLM_BASE_URL']}", "-v", f"{HERE / 'fault_proxy.py'}:/fault_proxy.py:ro",
               image, "/fault_proxy.py", "0.0.0.0", *map(str, FAULT_PORTS.values())])
        if any(all_scenarios[name].get("browser") for name in only):
            start(["docker", "run", "-d", "--rm", "--name", BROWSER, "--network", NETWORK, "--init", PLAYWRIGHT_MCP,
                   "--headless", "--host", "0.0.0.0", "--port", "8931", "--allowed-hosts", "*", "--isolated",
                   "--no-sandbox"])
        for name in only:
            try:
                record(image, name, all_scenarios[name], args.model, args.out)
            except Exception as exc:
                print(name, "failed:", mask(f"{type(exc).__name__}: {exc}")[:800], flush=True)
    finally:
        run(["docker", "rm", "-f", FAULTS, BROWSER])
        run(["docker", "network", "rm", NETWORK])


if __name__ == "__main__":
    main()
