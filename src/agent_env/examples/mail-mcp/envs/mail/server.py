"""The mail env: an MCP server with one tool, which searches a small, fixed inbox."""
import json
from typing import Annotated

from pydantic import Field

from agentenv_protocol import AgentEnvEnvironment, environment_card, tool

INBOX = (
    {"id": "m-1", "from": "billing@acme-supplies.example", "subject": "Invoice INV-1042 for October",
     "body": "Invoice INV-1042 for $2,740.00 is due on November 15."},
    {"id": "m-2", "from": "dana@example.com", "subject": "Team lunch on Friday",
     "body": "Pizza at noon in the kitchen."},
    {"id": "m-3", "from": "it@example.com", "subject": "Your password expires soon",
     "body": "Choose a new password within 7 days."},
)


@environment_card(name="mail")
class MailEnv(AgentEnvEnvironment):
    @tool(name="search_emails")
    def search_emails(
        self,
        query: Annotated[str, Field(description="Words each email returned holds, in its sender, subject or body.")],
    ) -> str:
        """Search the inbox: the emails holding every word of the query, ignoring case, as a JSON list."""
        words = query.lower().split()
        return json.dumps([email for email in INBOX if all(word in " ".join(email.values()).lower() for word in words)])


if __name__ == "__main__":
    MailEnv().serve()
