"""What a bundle import wrote and how it treated each reference, and that told as text: the report, the folder's README
and its provenance."""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

from agent_env.config.model import ENV_API_KEY, ENV_BASE_URL
from agent_env.entity.refs import EntityKind
from agent_env.store.image_store.oci_registry_credentials import registry_host_from_ref
from agent_env.utils.docker.build import DOCKER_PLUGINS
from agent_env.utils.docker.lock import IN_IMAGE, Locked
from agent_env.utils.docker.prune import Pruned

# The file an import records what it wrote in, and from where, for publishing the folder back.
PROVENANCE = "provenance.json"


@dataclass(frozen=True)
class Pinned:
    """A reference left to the store, pinned at the version the store held: the folder can't hold what it names."""

    where: str
    kind: EntityKind
    id: str
    version: int
    why: str


@dataclass(frozen=True)
class Floating:
    """A store reference written without a version, so a run reads whatever version is current then."""

    where: str
    kind: EntityKind
    id: str
    why: str


@dataclass(frozen=True)
class Dropped:
    """A value left out of the folder, or masked in it: a secret, a scoped field, or one its toml has no key for."""

    where: str
    why: str


@dataclass(frozen=True)
class Need:
    """Something a task uses that core doesn't ship, and what registers it here."""

    where: str
    what: str  # step type, env type or artifact type
    name: str
    by: str  # the installed distribution that registers it, or "configuration" for a config file's impls


@dataclass(frozen=True)
class Written:
    """An entity written into the folder."""

    folder: str  # in the bundle: envs/<name>, agents/<name>, artifacts/<name> or skills/<name>
    kind: EntityKind
    id: str
    version: int
    type: str


@dataclass(frozen=True)
class WrittenImage:
    """An image written into its entity's folder, from its stored build context."""

    folder: str
    key: str
    artifact: str
    version: int
    found: bool  # the image records no Dockerfile, so the context's one was taken
    pruned: Pruned  # what its folder holds of the context
    locked: Locked | None  # what was pinned to the stored image's versions
    unlocked: str | None  # why nothing was


@dataclass(frozen=True)
class Pulled:
    """An image a folder names by its registry ref, which the machine running it pulls."""

    folder: str
    key: str
    image: str
    why: str | None = None  # why it's pulled though it has a build context: the context can't be written


@dataclass(frozen=True)
class ImportedTask:
    path: str  # in the bundle: tasks/<name>.json
    id: str
    version: int
    steps: int


@dataclass(frozen=True)
class ImportedEval:
    path: str  # in the bundle: evals/<name>.toml
    id: str
    version: int


@dataclass(frozen=True)
class BundleImport:
    """What an import wrote, and how it treated each reference and secret."""

    root: Path
    files: tuple[str, ...]  # what it wrote beside the entities' folders
    tasks: tuple[ImportedTask, ...]
    eval: ImportedEval | None
    written: tuple[Written, ...]
    images: tuple[WrittenImage, ...]
    pulled: tuple[Pulled, ...]
    pinned: tuple[Pinned, ...]
    floating: tuple[Floating, ...]
    dropped: tuple[Dropped, ...]
    needs: tuple[Need, ...]
    undeclared: tuple[str, ...]  # the steps whose type declares no references, so any they make float
    duplicates: tuple[str, ...]  # the eval's entries that named a task it already named
    downloaded: int  # bytes, of what the store said its objects' sizes are

    def report(self) -> list[tuple[str, list[str]]]:
        """What still needs the store, what the task needs beyond core, what was written, and how each reference,
        secret and scoped field was treated, as titled lists of lines; an empty section is left out."""
        needed: dict[tuple[str, str, str], list[str]] = {}
        for need in self.needs:
            needed.setdefault((need.by, need.name, need.what), []).append(need.where)
        return _sections([
            ("Still needs the store", [f"{pin.kind.value} {pin.id} v{pin.version}: {pin.where}: {pin.why}"
                                       for pin in self.pinned]),
            ("Needs beyond core", [f"{by}: {name} ({what}), at {'; '.join(wheres)}"
                                   for (by, name, what), wheres in sorted(needed.items())]),
            ("Written", [f"{entity.folder}: {entity.type} {entity.id} v{entity.version}" for entity in self.written]),
            ("Images", [_image_line(image) for image in self.images]
             + [f"{pull.folder}: {pull.key} is pulled from its registry, as {pull.image}"
                + (f", since {pull.why}" if pull.why else "") for pull in self.pulled]),
            ("Floating", [f"{ref.kind.value} {ref.id}: {ref.where}: {ref.why}" for ref in self.floating]
             + [f"{step}: its type declares no references, so any it makes float" for step in self.undeclared]),
            ("Left out", [f"{drop.where}: {drop.why}" for drop in self.dropped]
             + [f"{entry}, which the eval names more than once" for entry in self.duplicates]),
        ])


def size_text(size: int) -> str:
    return f"{size} B" if size < 1_000 else f"{size / 1_000:.1f} KB" if size < 1_000_000 else f"{size / 1_000_000:.1f} MB"


def provenance(source: dict, imported: BundleImport) -> str:
    """What the folder was imported from and what it holds, by store id and version: no store's locations."""
    return json.dumps({
        "agent_env": package_version("agentenv-framework"),
        "imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from": source,
        "written": [{"folder": entity.folder, "kind": entity.kind.value, "type": entity.type, "id": entity.id,
                     "version": entity.version} for entity in imported.written],
        "images": [{"folder": image.folder, "key": image.key, "id": image.artifact, "version": image.version,
                    "dockerfile": image.pruned.dockerfile, "dockerfile_found": image.found}
                   for image in imported.images],
        "store": [{"where": pin.where, "kind": pin.kind.value, "id": pin.id, "version": pin.version, "why": pin.why}
                  for pin in imported.pinned],
    }, indent=2, ensure_ascii=False) + "\n"


def readme(description: str, imported: BundleImport) -> str:
    """The bundle's README: its description, where it came from, how to run and share it, what it holds, then the
    import's report."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    held = ("Every env, agent and artifact it uses is written here, so `agent-env run` runs it with no store"
            + (", except what *Still needs the store* lists, which it reads from the store this was imported from."
               if imported.pinned else "."))
    lines = [description, "", f"Imported with agent-env {package_version('agentenv-framework')} on {stamp}. {held}", "",
             *_RUN_IT, *_needs_lines(imported), "", *_SHARE_IT, ""]
    contents = _sections([("Tasks", [f"{task.path}: task {task.id} v{task.version}" for task in imported.tasks]),
                          ("Eval", [f"{imported.eval.path}: eval {imported.eval.id} v{imported.eval.version}"]
                           if imported.eval else [])])
    for title, items in [*contents, *imported.report()]:
        lines += [f"## {title}", "", *(f"- {item}" for item in items), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def _image_line(image: WrittenImage) -> str:
    """How an image was written: its Dockerfile, what its folder holds, and why it holds all of its context."""
    pruned = image.pruned
    found = " (found: the image records no Dockerfile)" if image.found else ""
    largest = ", ".join(f"{name} ({size_text(size)})" for name, size in pruned.largest)
    whole = f"; all of its build context, since {pruned.whole}" if pruned.whole else ""
    return (f"{image.folder}: {image.key} built from {pruned.dockerfile}{found}, {pruned.files} file(s), "
            f"{size_text(pruned.size)}{whole}; largest: {largest}; {_locked_text(image)}")


def _locked_text(image: WrittenImage) -> str:
    """What of an image's installs is pinned to the stored image's versions, and what isn't."""
    if image.locked is None:
        return f"versions not pinned: {image.unlocked}"
    locked = image.locked
    pinned = ([f"{locked.python} Python package(s) by {locked.constraints}"] if locked.constraints else []) + (
        [f"Python packages by {locked.lock_file}, as the build installs them"] if locked.lock_file else []) + list(
        locked.npm)
    left = f"; left as the build resolves them: {'; '.join(locked.not_covered)}" if locked.not_covered else ""
    return f"pinned to the stored image: {', '.join(pinned) or 'nothing it installs'}{left}"


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


_PLUGINS = list(DOCKER_PLUGINS.values())
_RUN_IT = [
    "## Run it",
    "",
    "    agent-env run .",
    "",
    "A machine running it needs:",
    "",
    "- agent-env (`pip install agentenv-framework`).",
    f"- Docker with its {_listed([plugin.name for plugin in _PLUGINS])} plugins. On Linux, Docker Engine and the "
    f"{_listed([f'`{plugin.package}`' for plugin in _PLUGINS])} packages (or a distribution's "
    f"{_listed(['`docker.io`', *(f'`{plugin.distribution_package}`' for plugin in _PLUGINS)])}). On macOS, a Docker "
    f"VM such as Docker Desktop or colima, and the {_listed(['`docker`', *(f'`{plugin.formula}`' for plugin in _PLUGINS)])} "
    "CLIs; Homebrew's plugins are found once `~/.docker/config.json` lists their folder in `cliPluginsExtraDirs`.",
    f"- A model endpoint for its agents and judges: `{ENV_BASE_URL}` and `{ENV_API_KEY}`, or `[model] base_url` and "
    "`api_key` in `.agentenv/config.toml`.",
    "- An agent where a step names none: the configured default (`[agents] default_a2a_agent_id`).",
    "",
    "Its images are built from the build contexts in their folders on the machine that runs it, pinned to the "
    "Python and npm versions of the images the store held, as *Images* says. To build one unpinned, take out the "
    f"COPY and two ARG lines naming {posixpath.basename(IN_IMAGE)} after each FROM in its Dockerfile.",
]
_SHARE_IT = [
    "## Share it",
    "",
    "The folder is source: commit it to git, or archive it without macOS's extended attributes, which tar on Linux warns "
    "about for every file (`COPYFILE_DISABLE=1 tar -czf` on macOS, or GNU tar's `--no-xattrs`).",
]


def _needs_lines(imported: BundleImport) -> list[str]:
    plugins = sorted({need.by for need in imported.needs if need.by != "configuration"})
    configured = any(need.by == "configuration" for need in imported.needs)
    lines = [f"- The plugins it uses: `{'`, `'.join(plugins)}`."] if plugins else []
    if configured:
        lines.append("- The configuration that registers what *Needs beyond core* says configuration registers.")
    registries = sorted({host for pulled in imported.pulled if (host := registry_host_from_ref(pulled.image))})
    if registries:
        lines.append(f"- Access to the registries its pulled images are in: `{'`, `'.join(registries)}`. agent-env reads "
                     "a private one only through its configured image store, such as an ECR store for an ECR registry.")
    return lines


def _sections(sections: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    return [(title, lines) for title, lines in sections if lines]
