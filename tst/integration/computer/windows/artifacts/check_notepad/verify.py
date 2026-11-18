"""Whether the line the agent was asked to type in Notepad is in hello.txt in the session user's Documents, read from
the file through computer-exec."""
from agentenv_protocol.computer.conformance.client import Driver

LINE = "agent-env on Windows"
READ = ("python3", "-c", "import sys;sys.stdout.buffer.write(open(sys.argv[1],'rb').read())", "Documents/hello.txt")


async def verify(mcp_url, env_url):
    driver = await Driver.connect(env_url, 60)
    out = await driver.run(*READ)
    found = out.get("exit_code") == 0 and LINE in out.get("stdout", "")
    return [{"id": "notepad", "result": found, "description": "Documents\\hello.txt holds the line" if found else
             f"Documents\\hello.txt doesn't hold the line: {(out.get('stdout') or out.get('stderr') or '')[-300:]!r}"}]
