"""``python -m agent_env.cli`` entry point.

Lets the CLI run as a module (used by ``agent-env up``'s bootstrap, which shells out with
``sys.executable -m agent_env.cli ...`` so it works regardless of how ``agent-env`` is on PATH).
"""

from agent_env.cli import cli

if __name__ == "__main__":
    cli()
