"""Widget Docs, replayed: the pages of a captured docs site, served as they were captured for any host, with the
newsletter sign-up the capture can't answer written as ``@api`` routes on top. The sign-ups are the site's state, kept
behind one lock since plain ``def`` handlers run in worker threads: the data plane's ``data/get`` exports them and
``data/reset`` empties them."""
from __future__ import annotations

import copy
import threading

from pydantic import BaseModel

from agentenv_protocol import DataPart, environment_card, get_data
from agentenv_protocol.website import BadRequest, Conflict, api, website
from agentenv_protocol.website.replay import ReplayWebsite

SUBSCRIPTIONS = "subscriptions"


class Subscription(BaseModel):
    email: str


@environment_card(name="widget-docs")
@website(domains=["docs.widgets.test"])
class WidgetDocs(ReplayWebsite):
    capture = "capture.warc.gz"

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.subscriptions: list[dict] = []

    def add_site_data(self, parts: list) -> None:
        """Each data part's ``subscriptions``, added to those there are."""
        with self._lock:
            for part in parts:
                if not isinstance(part, DataPart):
                    raise ValueError(f"Widget Docs loads its sign-ups from a data part, not a {type(part).__name__}")
                self.subscriptions += copy.deepcopy(part.data.get(SUBSCRIPTIONS, []))

    def reset_site_data(self) -> None:
        with self._lock:
            self.subscriptions = []

    @get_data
    def export(self) -> list:
        with self._lock:
            return [DataPart(data={SUBSCRIPTIONS: copy.deepcopy(self.subscriptions)})]

    @api.post("/api/subscribe", status=201)
    def subscribe(self, body: Subscription) -> dict:
        """Sign an email address up for the newsletter."""
        if "@" not in body.email:
            raise BadRequest("that is not an email address")
        with self._lock:
            if any(s["email"] == body.email for s in self.subscriptions):
                raise Conflict(f"{body.email} is subscribed already")
            subscribed = {"id": len(self.subscriptions) + 1, "email": body.email, "at": self.now().isoformat()}
            self.subscriptions.append(subscribed)
            return copy.deepcopy(subscribed)

    @api.get("/api/subscriptions")
    def list_subscriptions(self) -> list[dict]:
        """Everyone signed up."""
        with self._lock:
            return copy.deepcopy(self.subscriptions)


if __name__ == "__main__":
    WidgetDocs().serve()
