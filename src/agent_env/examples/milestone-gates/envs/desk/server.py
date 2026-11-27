"""The desk env: one supplier email in the inbox, and mail the agent can draft or send, for the milestone-gates
example's gates to read."""
from typing import Annotated, Optional

from pydantic import Field

from agentenv_protocol import AgentEnvEnvironment, environment_card, tool


@environment_card(name="desk")
class DeskEnv(AgentEnvEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.inbox = [{"id": 1, "from": "billing@supplier.example", "subject": "Invoice 4417 overdue",
                       "body": "Invoice 4417 for 1,240.00 was due on the 1st and is still open. Can you confirm when "
                               "it will be paid?"}]
        self.drafts: list[dict] = []
        self.sent: list[dict] = []

    @tool(name="list_inbox")
    def list_inbox(self) -> list[dict]:
        """Every email in the inbox, oldest first."""
        return self.inbox

    @tool(name="save_draft")
    def save_draft(self, to: Annotated[str, Field(description="Who it will go to.")],
                   subject: Annotated[str, Field(description="Its subject.")],
                   body: Annotated[str, Field(description="Its text.")]) -> str:
        """Save an email as a draft, without sending it."""
        self.drafts.append({"id": len(self.drafts) + 1, "to": to, "subject": subject, "body": body})
        return "saved"

    @tool(name="list_drafts")
    def list_drafts(self) -> list[dict]:
        """Every saved draft, oldest first."""
        return self.drafts

    @tool(name="send_email")
    def send_email(self, to: Annotated[str, Field(description="Who to send it to.")],
                   subject: Annotated[str, Field(description="Its subject.")],
                   body: Annotated[str, Field(description="Its text.")],
                   cc: Annotated[Optional[str], Field(description="Who to copy, comma-separated.")] = None) -> str:
        """Send an email."""
        self.sent.append({"id": len(self.sent) + 1, "to": to, "cc": cc or "", "subject": subject, "body": body})
        return "sent"

    @tool(name="list_sent")
    def list_sent(self) -> list[dict]:
        """Every sent email, oldest first."""
        return self.sent


if __name__ == "__main__":
    DeskEnv().serve()
