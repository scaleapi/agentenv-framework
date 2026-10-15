"""The echo env: an MCP server with one tool, which returns the text it's sent."""
from typing import Annotated

from pydantic import Field

from agentenv_protocol import AgentEnvEnvironment, environment_card, tool


@environment_card(name="echo")
class EchoEnv(AgentEnvEnvironment):
    @tool(name="echo")
    def echo(self, text: Annotated[str, Field(description="The text to send back.")]) -> str:
        """Return the text it's sent."""
        return text


if __name__ == "__main__":
    EchoEnv().serve()
