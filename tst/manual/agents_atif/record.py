"""Record an agent's ATIF fixtures (``tst/data/<agent>_atif``) again, as after a bump of its pinned harness, which each
adapter test's ``tested_with`` check asks for. One recorder for every agent whose spec is in ``specs/<agent>.py``. Run
by hand from a checkout with agent-env installed:

    LITELLM_BASE_URL=... LITELLM_API_KEY=... python tst/manual/agents_atif/record.py OUT AGENT [--only a,b]
    python tst/manual/agents_atif/make_fixtures.py OUT AGENT

It builds the agent's image from the checkout as a release builds it (``infra_build.py``), and runs each scenario in a
container of it with atif/1 prescribed. For each turn it writes the native final read, the atif/1 final read (document,
aux log and turn context), the atif/1 pages read live every second, and the agent's own files after (its spec's
``files``). The common scenarios: two coding turns of one conversation; a browser's screenshot returned to the model,
through Playwright MCP beside the agent; and the endpoint refusing every call, and failing calls the agent then
retries, through ``codex_gemini_atif/fault_proxy.py`` in a container of its own. A spec adds its agent's own (a
sub-agent, a high reasoning effort) and may run services beside the agent (a browser over CDP). Needs Docker.

The key is never written: every text is masked, and a copied file holding it is dropped."""
import argparse
import importlib
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tst.manual.atif_recording import SECRET, build, http, mask, run, save, start, turn  # noqa: E402

HERE = Path(__file__).resolve().parent
FAULT_PROXY = REPO / "tst" / "manual" / "codex_gemini_atif" / "fault_proxy.py"
PYTHON = "public.ecr.aws/docker/library/python:3.12-slim"
PLAYWRIGHT_MCP = ("mcr.microsoft.com/playwright/mcp:v0.0.86"
                  "@sha256:2dfc17e08e8db54678ec965ce9ffb4100be0aa68135774d761bac9035ceb6668")
FAULT_PORTS = {"refuse": 18601, "resume": 18602, "retry": 18603}
CODING = ["Create hello.py that prints the sum of the integers 1 to 100, run it with python3, and reply with only the "
          "number it prints.",
          "Now change it to sum 1 to 1000, run it again, and reply with only the number."]
COMMON = {
    "coding": {"prompts": CODING},
    "browser": {"browser": True, "prompts": [
        "Use the browser tools to open https://example.com, then call the screenshot tool with no filename so the image "
        "comes back to you, and reply with the main heading you see in it."]},
    "refused": {"faults": "refuse", "prompts": ["Reply with ok."]},
    "retried": {"faults": "retry", "prompts": [
        "Run `echo first` with your shell tool, then run `echo second` with it, then reply with both outputs on one "
        "line."]},
}


def spec_of(agent: str) -> dict:
    """The agent's spec: ``image``, ``model``, ``files`` (its own state, copied after each scenario), ``version`` (a
    command printing its harness's version), and optionally ``config`` (agent-config every scenario posts),
    ``scenarios`` (which common ones it records, all by default), ``own`` (its own scenarios, as COMMON's), ``env``
    (container environment), ``posts(scenario)`` (further config posts) and ``services()`` (containers to run on the
    recording network, as ``docker run`` argument lists by name)."""
    return importlib.import_module(f"tst.manual.agents_atif.specs.{agent.replace('-', '_')}").SPEC


def network_of(spec: dict) -> str:
    """Each agent's recording has a network, fault proxy and browser of its own, so several record at once."""
    return f"atif-recording-{spec['image']}"


def faults_of(spec: dict) -> str:
    return f"{network_of(spec)}-faults"


def browser_of(spec: dict) -> str:
    return f"{network_of(spec)}-playwright"


def scenarios_of(spec: dict) -> dict:
    chosen = {name: COMMON[name] for name in spec.get("scenarios", COMMON)}
    return {**chosen, **spec.get("own", {})}


def start_agent(spec: dict, image: str, name: str, scenario: dict) -> str:
    mode = scenario.get("faults")
    endpoint = f"http://{faults_of(spec)}:{FAULT_PORTS[mode]}" if mode else os.environ["LITELLM_BASE_URL"]
    env = [arg for key, value in spec.get("env", {}).items() for arg in ("-e", f"{key}={value}")]
    run(["docker", "rm", "-f", name])
    start(["docker", "run", "-d", "--name", name, "--network", network_of(spec), "-p", "127.0.0.1::8000", "-e", "A2A_PORT=8000",
           "-e", "LITELLM_API_KEY", "-e", f"LITELLM_BASE_URL={endpoint}", *env, image])
    base = f"http://{run(['docker', 'port', name, '8000']).stdout.splitlines()[0]}"
    for _ in range(120):
        if http("GET", f"{base}/.well-known/agent.json", timeout=5)[0] == 200:
            return base
        time.sleep(1)
    raise RuntimeError(f"no card: {mask(run(['docker', 'logs', '--tail', '40', name]).stdout[-2000:])}")


def record(agent: str, spec: dict, image: str, name: str, scenario: dict, out: Path) -> None:
    folder = out / agent / name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    container = f"atif-{agent}-{name}"
    try:
        base = start_agent(spec, image, container, scenario)
        config = {"model": spec["model"], "timeout_seconds": 600, **spec.get("config", {}), **scenario.get("config", {})}
        posts = [("/ext/agent-config", config), ("/ext/trajectory/prescribe", {"formats": ["atif/1"]})]
        if scenario.get("browser") and not spec.get("posts"):
            posts.append(("/ext/mcp-config", {"url": f"http://{browser_of(spec)}:8931/mcp", "name": "browser"}))
        posts += spec["posts"](scenario) if spec.get("posts") else []
        for path, body in posts:
            status, answer = http("POST", base + path, body)
            if status >= 300:
                raise RuntimeError(f"{path}: {status} {mask(str(answer))[:400]}")
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
        if spec.get("files"):
            run(["docker", "cp", f"{container}:{spec['files']}", str(files)])
        for path in files.rglob("*") if files.exists() else ():
            if path.is_file() and SECRET and SECRET.encode() in path.read_bytes():
                print(agent, name, "dropped a file holding the key:", path.name, flush=True)
                path.unlink()
        logs = run(["docker", "logs", "--tail", "400", container])
        (folder / "agent.log").write_text(mask(logs.stdout + logs.stderr))
        run(["docker", "rm", "-f", container])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("agent")
    parser.add_argument("--only", help="comma-separated scenarios; all the agent's by default")
    args = parser.parse_args()
    if not SECRET or not os.environ.get("LITELLM_BASE_URL"):
        raise SystemExit("set LITELLM_BASE_URL and LITELLM_API_KEY")
    spec = spec_of(args.agent)
    every = scenarios_of(spec)
    names = args.only.split(",") if args.only else list(every)
    image = build(spec["image"])
    version = run(["docker", "run", "--rm", "--entrypoint", spec["version"][0], image, *spec["version"][1:]])
    print("built", image, version.stdout.strip(), flush=True)
    (args.out / args.agent).mkdir(parents=True, exist_ok=True)
    (args.out / args.agent / "cli_version.txt").write_text(version.stdout.strip())
    services = spec["services"]() if spec.get("services") else {}
    network, faults, browser = network_of(spec), faults_of(spec), browser_of(spec)
    run(["docker", "network", "create", network])
    try:
        start(["docker", "run", "-d", "--rm", "--name", faults, "--network", network, "--entrypoint", "python3",
               "-e", f"UPSTREAM={os.environ['LITELLM_BASE_URL']}", "-v", f"{FAULT_PROXY}:/p.py:ro",
               PYTHON, "/p.py", "0.0.0.0", *map(str, FAULT_PORTS.values())])
        if any(every[name].get("browser") for name in names) and not spec.get("posts"):
            start(["docker", "run", "-d", "--rm", "--name", browser, "--network", network, "--init", PLAYWRIGHT_MCP,
                   "--headless", "--host", "0.0.0.0", "--port", "8931", "--allowed-hosts", "*", "--isolated",
                   "--no-sandbox", "--image-responses", "allow"])
        for service, command in services.items():
            start(["docker", "run", "-d", "--rm", "--name", service, "--network", network, *command])
        for name in names:
            try:
                record(args.agent, spec, image, name, every[name], args.out)
            except Exception as exc:
                print(args.agent, name, "failed:", mask(f"{type(exc).__name__}: {exc}")[:800], flush=True)
    finally:
        run(["docker", "rm", "-f", faults, browser, *services])
        run(["docker", "network", "rm", network])


if __name__ == "__main__":
    main()
