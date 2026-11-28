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
retried, through ``fault_proxy.py`` in a container of its own; for Codex also a high reasoning effort, a sub-agent, and
a second turn's sub-agent forked with the conversation so far, on gpt-6.2-astra. They cost about a dollar. Needs Docker; ``mcr.microsoft.com/playwright/mcp`` is pulled
for the browser scenario.

The key is never written: every text is masked, and a copied file holding it is dropped. Then ``make_fixtures.py``
writes the fixtures, and running the recorded test as a script writes the expected documents. Note the CLI version in
the adapter's ``TESTED_WITH``."""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tst.manual.atif_recording import SECRET, build, http, mask, run, save, start, turn  # noqa: E402

HERE = Path(__file__).resolve().parent
NETWORK = "atif-recording"
BROWSER = "atif-recording-playwright"
FAULTS = "atif-recording-faults"
PLAYWRIGHT_MCP = ("mcr.microsoft.com/playwright/mcp:v0.0.86"
                  "@sha256:2dfc17e08e8db54678ec965ce9ffb4100be0aa68135774d761bac9035ceb6668")
FAULT_PORTS = {"refuse": 18601, "resume": 18602, "retry": 18603}
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
    # On a model outside the CLI's code mode, such as gpt-5.5, Codex 0.172.0 forks the sub-agent with the parent's
    # spawn_agent call in its history but without the collaboration tools, and the endpoint refuses its every call.
    "forked": {"agents": ["codex"], "config": {"model": "gpt-6.2-astra"}, "prompts": [
        "Run `echo one` with your shell tool and reply with only its output.",
        "Spawn a sub-agent forked with this conversation's full history (all turns), have it run `echo two` with its "
        "shell tool and report the output back, then reply with only that output."]},
}


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
