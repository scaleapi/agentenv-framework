"""Record the Codex and Gemini ATIF fixtures (``tst/data/codex_atif``, ``tst/data/gemini_atif``) again, as after a bump
of an agent's pinned CLI, which each adapter test's ``tested_with`` check asks for. Run by hand from a checkout with
agent-env installed:

    LITELLM_BASE_URL=... LITELLM_API_KEY=... python tst/manual/codex_gemini_atif/record.py OUT codex|gemini [--only a,b]
    python tst/manual/codex_gemini_atif/make_fixtures.py OUT codex|gemini
    python tst/unit/agents/codex/test_codex_atif_recorded.py      # or gemini/test_gemini_atif_recorded.py

It builds the agent's image from the checkout as a release builds it (``infra_build.py``), and runs each scenario in a
container of it with atif/1 prescribed. For each turn it writes the native final read, the atif/1 final read (document,
aux log and turn context), the atif/1 pages read live every second, and the CLI's own files after (Codex's rollouts,
Gemini's ``~/.gemini``). The scenarios: two coding turns of one conversation; a browser's screenshot returned to the
model, through Playwright MCP beside the agent; and the endpoint refusing every call and failing calls that are then
retried, through ``fault_proxy.py`` in a container of its own; for Codex also a high reasoning effort, a sub-agent and
a sub-agent that spawns its own. They cost about a dollar. Needs Docker; ``mcr.microsoft.com/playwright/mcp`` is pulled
for the browser scenario.

The key is never written: every text is masked, and a copied file holding it is dropped. Then ``make_fixtures.py``
writes the fixtures, and running the recorded test as a script writes the expected documents. Note the CLI version in
the adapter's ``TESTED_WITH``."""
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
NETWORK = "atif-recording"
BROWSER = "atif-recording-playwright"
FAULTS = "atif-recording-faults"
PLAYWRIGHT_MCP = ("mcr.microsoft.com/playwright/mcp:v0.0.86"
                  "@sha256:2dfc17e08e8db54678ec965ce9ffb4100be0aa68135774d761bac9035ceb6668")
FAULT_PORTS = {"refuse": 18601, "resume": 18602, "retry": 18603}
TERMINAL = {"completed", "failed", "canceled", "rejected", "input-required", "auth-required"}
AGENTS = {
    "codex": {"image": "agentenv-codex", "model": "gpt-5.5", "files": "/home/agent/.codex/sessions",
              "version": ["codex", "--version"]},
    "gemini": {"image": "agentenv-gemini", "model": "gemini-2.5-flash", "files": "/home/agent/.gemini",
               "version": ["gemini", "--version"]},
}
CODING = ["Create hello.py that prints the sum of the integers 1 to 100, run it with python3, and reply with only the "
          "number it prints.",
          "Now change it to sum 1 to 1000, run it again, and reply with only the number."]
SCENARIOS = {
    "coding": {"prompts": CODING},
    "browser": {"browser": True, "prompts": [
        "Use the browser tools to open https://example.com, then call the screenshot tool with no filename so the image "
        "comes back to you, and reply with the main heading you see in it."]},
    "refused": {"faults": "refuse", "prompts": ["Reply with ok."]},
    "retried": {"faults": {"codex": "resume", "gemini": "retry"}, "prompts": [
        "Run `echo first` with your shell tool, then run `echo second` with it, then reply with both outputs on one "
        "line."]},
    "reasoning": {"agents": ["codex"], "config": {"effort": "high"}, "prompts": [
        "Think it through, then reply with only the answer: how many positive integers below 1000 are divisible by 3 "
        "or 5 but not by 15?"]},
    "subagent": {"agents": ["codex"], "prompts": [
        "Spawn a sub-agent and have it run `wc -l /etc/passwd` and report the count back to you. Then reply with only "
        "that number."]},
    "nested": {"agents": ["codex"], "prompts": [
        "Spawn a sub-agent, and tell it to spawn a sub-agent of its own that runs `uname -s` and reports back. Have the "
        "answer come back to you through both, then reply with only that answer."]},
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


def fault(agent: str, scenario: dict) -> str | None:
    faults = scenario.get("faults")
    return faults.get(agent) if isinstance(faults, dict) else faults


def start_agent(agent: str, image: str, name: str, scenario: dict) -> str:
    mode = fault(agent, scenario)
    endpoint = f"http://{FAULTS}:{FAULT_PORTS[mode]}" if mode else os.environ["LITELLM_BASE_URL"]
    run(["docker", "rm", "-f", name])
    start(["docker", "run", "-d", "--name", name, "--network", NETWORK, "-p", "127.0.0.1::8000", "-e", "A2A_PORT=8000",
           "-e", "LITELLM_API_KEY", "-e", f"LITELLM_BASE_URL={endpoint}", image])
    base = f"http://{run(['docker', 'port', name, '8000']).stdout.splitlines()[0]}"
    for _ in range(90):
        if http("GET", f"{base}/.well-known/agent.json", timeout=5)[0] == 200:
            return base
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


def record(agent: str, image: str, name: str, scenario: dict, out: Path) -> None:
    spec = AGENTS[agent]
    folder = out / agent / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    container = f"atif-{agent}-{name}"
    try:
        base = start_agent(agent, image, container, scenario)
        config = {"model": spec["model"], "timeout_seconds": 600, **scenario.get("config", {})}
        posts = [("/ext/agent-config", config), ("/ext/trajectory/prescribe", {"formats": ["atif/1"]})]
        if scenario.get("browser"):
            posts.append(("/ext/mcp-config", {"url": f"http://{BROWSER}:8931/mcp", "name": "browser"}))
        for path, body in posts:
            status, answer = http("POST", base + path, body)
            if status >= 300:
                raise RuntimeError(f"{path}: {status} {answer}")
        for index, prompt in enumerate(scenario["prompts"]):
            started = time.time()
            task_id, task, pages = turn(base, f"ctx-{agent}-{name}", prompt)
            _, native = http("POST", f"{base}/ext/trajectory", {"task_id": task_id})
            _, atif = http("POST", f"{base}/ext/trajectory", {"task_id": task_id, "format": "atif/1"})
            save(folder / f"turn-{index}.json", {"prompt": prompt, "task": task, "native": native, "atif": atif,
                                                 "live_pages": pages})
            print(f"{agent} {name} turn {index}: {task['status']['state']} in {int(time.time() - started)} s, "
                  f"{len(pages)} live pages", flush=True)
    finally:
        files = folder / "files"
        run(["docker", "cp", f"{container}:{spec['files']}", str(files)])
        for path in files.rglob("*") if files.exists() else ():
            if path.is_file() and SECRET and SECRET.encode() in path.read_bytes():
                print(agent, name, "dropped a file holding the key:", path.name, flush=True)
                path.unlink()
        logs = run(["docker", "logs", "--tail", "400", container])
        (folder / "agent.log").write_text(logs.stdout + logs.stderr)
        run(["docker", "rm", "-f", container])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("agent", choices=sorted(AGENTS))
    parser.add_argument("--only", help="comma-separated scenarios; all the agent's by default")
    args = parser.parse_args()
    if not SECRET or not os.environ.get("LITELLM_BASE_URL"):
        raise SystemExit("set LITELLM_BASE_URL and LITELLM_API_KEY")
    names = [name for name in (args.only.split(",") if args.only else SCENARIOS)
             if args.agent in SCENARIOS[name].get("agents", AGENTS)]
    image = build(AGENTS[args.agent]["image"])
    version = run(["docker", "run", "--rm", "--entrypoint", AGENTS[args.agent]["version"][0], image,
                   *AGENTS[args.agent]["version"][1:]]).stdout.strip()
    print("built", image, version, flush=True)
    (args.out / args.agent).mkdir(parents=True, exist_ok=True)
    (args.out / args.agent / "cli_version.txt").write_text(version)
    run(["docker", "network", "create", NETWORK])
    try:
        start(["docker", "run", "-d", "--rm", "--name", FAULTS, "--network", NETWORK, "--entrypoint", "python3",
               "-e", f"UPSTREAM={os.environ['LITELLM_BASE_URL']}", "-v", f"{HERE / 'fault_proxy.py'}:/p.py:ro",
               image, "/p.py", "0.0.0.0", *map(str, FAULT_PORTS.values())])
        if any(SCENARIOS[name].get("browser") for name in names):
            start(["docker", "run", "-d", "--rm", "--name", BROWSER, "--network", NETWORK, "--init", PLAYWRIGHT_MCP,
                   "--headless", "--host", "0.0.0.0", "--port", "8931", "--allowed-hosts", "*", "--isolated",
                   "--no-sandbox", "--image-responses", "allow"])
        for name in names:
            try:
                record(args.agent, image, name, SCENARIOS[name], args.out)
            except Exception as exc:
                print(args.agent, name, "failed:", mask(f"{type(exc).__name__}: {exc}")[:800], flush=True)
    finally:
        run(["docker", "rm", "-f", FAULTS, BROWSER])
        run(["docker", "network", "rm", NETWORK])


if __name__ == "__main__":
    main()
