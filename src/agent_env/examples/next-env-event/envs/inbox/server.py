"""The inbox env: mail the task's triggers deliver over the week, for the agent to read and answer."""
from typing import Annotated

from pydantic import Field

from agentenv_protocol import AgentEnvEnvironment, environment_card, tool


@environment_card(name="inbox")
class InboxEnv(AgentEnvEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.inbox: list[dict] = []
        self.sent: list[dict] = []

    @tool(name="deliver_email", hidden_by_default=True)
    def deliver_email(self, sender: Annotated[str, Field(description="Who it is from.")],
                      subject: Annotated[str, Field(description="Its subject.")],
                      body: Annotated[str, Field(description="Its text.")]) -> str:
        """Deliver an email to the inbox; the task's triggers call it, and the agent never sees it."""
        self.inbox.append({"id": len(self.inbox) + 1, "from": sender, "subject": subject, "body": body})
        return "delivered"

    @tool(name="list_inbox")
    def list_inbox(self) -> list[dict]:
        """Every email in the inbox, oldest first."""
        return self.inbox

    @tool(name="send_email")
    def send_email(self, to: Annotated[str, Field(description="Who to send it to.")],
                   subject: Annotated[str, Field(description="Its subject.")],
                   body: Annotated[str, Field(description="Its text.")]) -> str:
        """Send an email."""
        self.sent.append({"to": to, "subject": subject, "body": body})
        return "sent"


if __name__ == "__main__":
    InboxEnv().serve()
