"""Each test keeps its agents' turn counts in its own state folder: the SDK keeps them in the user's, where one test's
contexts would go on counting from another's."""
import pytest


@pytest.fixture(autouse=True)
def _own_turn_counts(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("state")))
