"""A live site for integration tests: github.com, whose pages its server serves none of. Its data plane reads a public
repository through api.github.com and keeps what a task loads; its session is a test cookie, which GitHub ignores."""
import json
import urllib.request
from typing import Optional

from pydantic import BaseModel

from agentenv_protocol import DataPart, add_data, get_data, reset_data
from agentenv_protocol.website import WebsiteEnvironment, ready_check, session_state, website


@website(domains=["github.com"], live=True,
         allow_hosts=["*.github.com", "*.githubassets.com", "*.githubusercontent.com"])
class GitHub(WebsiteEnvironment):
    class Settings(BaseModel):
        repository: str = "octocat/Hello-World"

    def __init__(self) -> None:
        super().__init__()
        self.loaded: list = []
        self._repository: Optional[dict] = None

    def repository(self) -> dict:
        """The repository as the public API answers, read once: an unauthenticated caller has 60 calls an hour."""
        if self._repository is None:
            url = f"https://api.github.com/repos/{self.settings.repository}"
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"}),
                                        timeout=10) as response:
                self._repository = json.load(response)
        return self._repository

    @ready_check
    def api_answers(self) -> bool:
        return self.repository()["full_name"] == self.settings.repository

    @session_state
    def account(self, user: Optional[str]) -> dict:
        return {"cookies": [{"name": "agentenv_test_session", "value": user or "", "path": "/", "max_age": 600}],
                "local_storage": []}

    @reset_data
    async def _reset(self) -> None:
        self.loaded.clear()

    @add_data
    async def _add(self, parts: list) -> None:
        self.loaded.extend(part.data for part in parts if part.kind == "data")

    @get_data
    async def _state(self) -> list:
        return [DataPart(data={"repository": self.repository()["full_name"], "loaded": self.loaded})]


if __name__ == "__main__":
    GitHub().serve()
