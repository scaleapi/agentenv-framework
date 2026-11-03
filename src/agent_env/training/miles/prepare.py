"""The server's tasks as Miles' prompt data: a JSON line per task of each split, its chat ``messages`` and the
``metadata`` the agent function plays it from."""

from agent_env.training._chat import write_rows

__all__ = ["write_rows"]
