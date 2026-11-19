"""Each test keeps its agents' turn counts in its own folder: the SDK keeps them under the user's home, where one test's
contexts would go on counting from another's."""
import pytest

from agentenv_protocol.a2a_agent import _turn_index


@pytest.fixture(autouse=True)
def _own_turn_counts(tmp_path_factory, monkeypatch):
    monkeypatch.setattr(_turn_index, "TURNS_DIR", tmp_path_factory.mktemp("turns"))
