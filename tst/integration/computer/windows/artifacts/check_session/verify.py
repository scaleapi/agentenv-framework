"""The protocol's Windows session on the deployed computer, run inside it as the session user on the notes app's venv:
a launch's exit code is its app's; a pause holds every process of the app's job, one the app started too, and they run
again after; a stop ends an app with no window at once, one whose window closes when asked within the timeout, and one
whose window won't at the timeout, nothing of it left running after each; the helper killed ends its app; and a
program that isn't there is refused, saying why."""
import json

from agentenv_protocol.computer.conformance.client import Driver

PYTHON = "C:/agentenv/apps/notes/venv/Scripts/python.exe"
SESSION = r"""
import asyncio, json, os, signal, sys, tempfile, time
from pathlib import Path
from agentenv_protocol.computer import _reaping, _session_windows as session

work = Path(tempfile.mkdtemp())
pythonw = str(Path(sys.executable).with_name("pythonw.exe"))
(work / "writer.py").write_text(
    "import subprocess, sys, time\nfrom pathlib import Path\nout = Path(sys.argv[1])\n"
    "if len(sys.argv) > 2:\n    subprocess.Popen([sys.executable, __file__, sys.argv[1] + '.child'])\n"
    "while True:\n    with out.open('a') as f:\n        f.write('x')\n    time.sleep(0.01)\n")
(work / "window.py").write_text(
    "import sys, tkinter as tk\nroot = tk.Tk()\nroot.title(sys.argv[1])\n"
    "if sys.argv[2] == 'stubborn':\n    root.protocol('WM_DELETE_WINDOW', lambda: None)\nroot.mainloop()\n")
facts, written = {}, {}


def sizes(*paths):
    '''Each file's size as an open handle reads it, which a stat by name can read behind.'''
    read = []
    for path in paths:
        try:
            with path.open("rb") as file:
                read.append(file.seek(0, 2))
        except FileNotFoundError:
            read.append(0)
    return read


async def listed(title):
    return any(window["title"] == title for window in await session.windows())


async def main():
    pid = await session.launch([sys.executable, "-c", "import sys; sys.exit(7)"], work, {})
    facts["exit"] = await _reaping.returncode(pid, 30)

    out = work / "writes"
    files = (out, Path(f"{out}.child"))
    pid = await session.launch([pythonw, str(work / "writer.py"), str(out), "spawn"], work, {})
    await asyncio.sleep(3)
    with session.paused(pid):
        await asyncio.sleep(0.5)
        held = sizes(*files)
        await asyncio.sleep(2)
        still = sizes(*files)
    await asyncio.sleep(2)
    facts["pause"] = {"held": held == still and min(held) > 0,
                      "ran after": all(after > before for after, before in zip(sizes(*files), still)),
                      "alive": session.alive(pid)}
    await session.stop(pid, 10)
    ended = sizes(*files)
    await asyncio.sleep(1)
    written["windowless stop"] = [ended, sizes(*files)]
    facts["windowless stop"] = {"alive": session.alive(pid), "wrote after": written["windowless stop"][1] != ended}

    for title, kind, timeout_s in (("Obliging", "obliging", 10), ("Stubborn", "stubborn", 4)):
        pid = await session.launch([pythonw, str(work / "window.py"), title, kind], work, {})
        deadline = time.monotonic() + 30
        while not await listed(title) and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        shown, started = await listed(title), time.monotonic()
        await session.stop(pid, timeout_s)
        facts[f"{kind} stop"] = {"shown": shown, "waited out": time.monotonic() - started >= timeout_s,
                                 "alive": session.alive(pid), "listed": await listed(title)}

    out = work / "orphan"
    pid = await session.launch([pythonw, str(work / "writer.py"), str(out)], work, {})
    await asyncio.sleep(2)
    os.kill(pid, signal.SIGTERM)
    await asyncio.sleep(2)
    ended = sizes(out)
    await asyncio.sleep(1)
    written["helper killed"] = [ended, sizes(out)]
    facts["helper killed"] = {"alive": session.alive(pid), "wrote after": written["helper killed"][1] != ended}

    try:
        await session.launch(["no-such-program.exe"], work, {})
        facts["missing program"] = "started"
    except OSError as e:
        facts["missing program"] = str(e)


asyncio.run(main())
print(json.dumps({"facts": facts, "written": written}))
"""
EXPECTED = {
    "exit": 7,
    "pause": {"held": True, "ran after": True, "alive": True},
    "windowless stop": {"alive": False, "wrote after": False},
    "obliging stop": {"shown": True, "waited out": False, "alive": False, "listed": False},
    "stubborn stop": {"shown": True, "waited out": True, "alive": False, "listed": False},
    "helper killed": {"alive": False, "wrote after": False},
}
MISSING = "couldn't start no-such-program.exe: FileNotFoundError"


async def verify(mcp_url, env_url):
    driver = await Driver.connect(env_url, 120)
    out = await driver.run(PYTHON, "-c", SESSION, timeout_s=240)
    try:
        read = json.loads(out.get("stdout") or "")
    except ValueError:
        return [{"id": "session", "result": False, "description": json.dumps(out)[-2000:]}]
    facts, written = read["facts"], read["written"]
    missing = facts.get("missing program", "")
    return [{"id": name, "result": facts.get(name) == expected,
             "description": json.dumps({**facts.get(name, {}), "sizes": written[name]} if name in written
                                       else facts.get(name))}
            for name, expected in EXPECTED.items()] + [
        {"id": "missing program", "result": missing.startswith(MISSING), "description": missing}]
