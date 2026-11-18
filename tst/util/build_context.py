"""The file an image's build context sends under a name, its own or one its ``shared_sources.toml`` takes from
agent-env's sources: what a test of a shipped image reads, as its build reads it."""

from pathlib import Path

from agent_env.utils.docker.context import BuildContext


def sent(folder: Path, name: str) -> Path:
    """The file ``folder``'s build context sends as ``name``."""
    paths = {entry.name: entry.path for entry in BuildContext.of(folder, folder / "Dockerfile").entries}
    if paths.get(name) is None:
        raise AssertionError(f"{folder.name}'s build context sends no file {name}")
    return paths[name]
