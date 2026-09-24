"""card_name_from_github authenticates the Contents API call only when a token is passed in."""

import httpx

from agent_env.utils import card_naming


class _Recorder:
    def __init__(self):
        self.headers = None

    def client(self, **kwargs):
        self.headers = dict(kwargs.get("headers") or {})
        return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[])), **kwargs)


def test_token_becomes_the_bearer_header(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(card_naming.httpx, "Client", recorder.client)
    card_naming.card_name_from_github("https://github.com/o/r/blob/main/srv/Dockerfile", github_token="ghs_x")
    assert recorder.headers["Authorization"] == "Bearer ghs_x"


def test_without_a_token_the_call_is_anonymous(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(card_naming.httpx, "Client", recorder.client)
    assert card_naming.card_name_from_github("https://github.com/o/r/blob/main/srv/Dockerfile") is None
    assert "Authorization" not in recorder.headers
