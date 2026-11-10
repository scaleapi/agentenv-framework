"""Derive an env name from ``@environment_card(name=...)`` declarations, and what a site's ``@website(...)`` declares of
it (``WebsiteSource``), from source that's read, never run (import-light: the hub imports this)."""

import ast
import posixpath
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx


@dataclass(frozen=True)
class WebsiteSource:
    """What the one ``@website(...)`` class of a site's source declares of it: ``live``, ``domains`` and
    ``allow_hosts`` where they're given as literals (None where not), and the fields its ``Settings`` declares as
    ``SecretStr``, which an env stores as its secret settings."""

    live: Optional[bool] = None
    domains: Optional[list[str]] = None
    allow_hosts: Optional[list[str]] = None
    secret_settings: list[str] = field(default_factory=list)

def card_name_from_source(dockerfile: str, context: str | None) -> str | None:
    """Derive the env name by statically reading `@environment_card(name="...")` from the
    server's source — the Dockerfile's own directory (where the server's source lives),
    falling back to the build context. Nothing is built or run.

    Returns the single declared card name, or None when the source declares none, uses a
    non-literal name, or is ambiguous (several distinct names) — the caller then requires
    an explicit --environment-name.
    """
    names = _scan_environment_card_names(Path(dockerfile).resolve().parent)
    if not names and context:
        names = _scan_environment_card_names(Path(context).resolve())
    return names[0] if len(names) == 1 else None


def card_name_from_github(
    dockerfile_github_url: str, context_github_url: str | None = None, *, github_token: str | None = None
) -> str | None:
    """The GitHub analogue of `card_name_from_source`: fetch the `.py` files in the Dockerfile's
    GitHub directory (Contents API, authenticated with ``github_token`` when one is given) and read the single
    `@environment_card(name="...")`. Any failure (bad URL, API/auth error, none, or ambiguous)
    returns None so the caller falls back to requiring --environment-name.
    """
    try:
        from agent_env.artifact.artifacts.docker_image import DockerImageArtifact

        parts = DockerImageArtifact._parse_github_url(dockerfile_github_url)
        if not parts.path:
            return None
        headers = {"Accept": "application/vnd.github+json"}
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"
        with httpx.Client(timeout=20, headers=headers, follow_redirects=True) as client:
            names = _github_card_names(client, parts.owner, parts.repo, parts.ref, posixpath.dirname(parts.path))
            if not names and context_github_url:
                cparts = DockerImageArtifact._parse_github_url(context_github_url)
                if cparts.path is not None:
                    names = _github_card_names(client, cparts.owner, cparts.repo, cparts.ref, cparts.path)
        return names[0] if len(names) == 1 else None
    except Exception:
        return None


def _github_card_names(client: httpx.Client, owner: str, repo: str, ref: str | None, directory: str) -> list[str]:
    return sorted({name for text in _github_sources(client, owner, repo, ref, directory)
                   for name in _card_names_in_source(text)})


def _github_sources(client: httpx.Client, owner: str, repo: str, ref: str | None, directory: str) -> list[str]:
    """The text of each ``.py`` file under ``directory`` on GitHub, tests aside."""
    ref_q = f"?ref={ref}" if ref else ""
    resp = client.get(f"https://api.github.com/repos/{owner}/{repo}/contents/{directory}{ref_q}")
    resp.raise_for_status()
    entries = resp.json()
    if not isinstance(entries, list):
        return []
    texts: list[str] = []
    for entry in entries:
        name = entry.get("name") or ""
        if entry.get("type") == "dir":
            texts += _github_sources(client, owner, repo, ref, entry["path"])
        elif entry.get("type") == "file" and name.endswith(".py") and "test" not in name:
            download_url = entry.get("download_url")
            if download_url:
                texts.append(client.get(download_url).text)
    return texts


def website_source_from_source(dockerfile: str, context: str | None) -> Optional[WebsiteSource]:
    """What the one ``@website(...)`` class in a server's source declares, read as ``card_name_from_source`` reads the
    name: the Dockerfile's folder, else the build context. None when there's none, or several."""
    found = _website_sources(_folder_texts(Path(dockerfile).resolve().parent))
    if not found and context:
        found = _website_sources(_folder_texts(Path(context).resolve()))
    return _one(found)


def website_source_from_github(dockerfile_github_url: str, context_github_url: str | None = None, *,
                               github_token: str | None = None) -> Optional[WebsiteSource]:
    """``website_source_from_source`` for a server whose source is on GitHub; None on any failure."""
    try:
        from agent_env.artifact.artifacts.docker_image import DockerImageArtifact

        parts = DockerImageArtifact._parse_github_url(dockerfile_github_url)
        if not parts.path:
            return None
        headers = {"Accept": "application/vnd.github+json"}
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"
        with httpx.Client(timeout=20, headers=headers, follow_redirects=True) as client:
            found = _website_sources(_github_sources(client, parts.owner, parts.repo, parts.ref,
                                                     posixpath.dirname(parts.path)))
            if not found and context_github_url:
                cparts = DockerImageArtifact._parse_github_url(context_github_url)
                if cparts.path is not None:
                    found = _website_sources(_github_sources(client, cparts.owner, cparts.repo, cparts.ref,
                                                             cparts.path))
        return _one(found)
    except Exception:
        return None


def website_source_in_files(files: Mapping[str, Path], dockerfile: str) -> Optional[WebsiteSource]:
    """``website_source_from_source`` among ``files``, keyed by POSIX path, as ``card_names_in_files`` reads them."""
    folder = posixpath.dirname(dockerfile)
    sources = {key: path for key, path in files.items() if key.endswith(".py")}
    near = {key: path for key, path in sources.items() if not folder or key.startswith(f"{folder}/")}
    for chosen in (near, sources):
        if found := _website_sources(_read(path) for path in chosen.values()):
            return _one(found)
    return None


def _folder_texts(root: Path) -> list[str]:
    return [_read(py) for py in root.rglob("*.py")]


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _one(found: list[WebsiteSource]) -> Optional[WebsiteSource]:
    distinct = [source for i, source in enumerate(found) if source not in found[:i]]
    return distinct[0] if len(distinct) == 1 else None


def _website_sources(texts: Iterable[str]) -> list[WebsiteSource]:
    found = []
    for text in texts:
        if "website" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                call = next((d for d in node.decorator_list if _decorator_name(d) == "website"), None)
                if isinstance(call, ast.Call):
                    found.append(_website_source(call, node))
    return found


def _website_source(call: ast.Call, cls: ast.ClassDef) -> WebsiteSource:
    """What ``call``, a class's ``@website(...)``, declares: ``live`` and ``allow_hosts`` as written, their defaults
    when left out, and unknown (None) when written as other than a literal or when the call unpacks ``**`` keywords,
    which may hold them."""
    given = {kw.arg: kw.value for kw in call.keywords if kw.arg}
    unpacked = any(kw.arg is None for kw in call.keywords)
    live = given.get("live")
    settings = next((n for n in cls.body if isinstance(n, ast.ClassDef) and n.name == "Settings"), None)
    secrets = [n.target.id for n in (settings.body if settings is not None else [])
               if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and _secret(n.annotation)]
    return WebsiteSource(
        live=(live.value if isinstance(live, ast.Constant) and isinstance(live.value, bool) else None) if live else (
            None if unpacked else False),
        domains=_names(given.get("domains")),
        allow_hosts=_names(given["allow_hosts"]) if "allow_hosts" in given else None if unpacked else [],
        secret_settings=secrets)


def _names(node: Optional[ast.expr]) -> Optional[list[str]]:
    if isinstance(node, (ast.List, ast.Tuple)) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
        return [e.value for e in node.elts]
    return None


def _secret(annotation: ast.expr) -> bool:
    """Whether ``annotation`` is ``SecretStr``, optional or not, written as a name or as a string (``"SecretStr"``,
    ``Optional["SecretStr"]``)."""
    for n in ast.walk(annotation):
        if isinstance(n, ast.Name) and n.id == "SecretStr" or isinstance(n, ast.Attribute) and n.attr == "SecretStr":
            return True
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            try:
                if _secret(ast.parse(n.value, mode="eval").body):
                    return True
            except SyntaxError:
                continue
    return False


def _decorator_name(decorator: ast.expr) -> Optional[str]:
    func = decorator.func if isinstance(decorator, ast.Call) else decorator
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def card_names_in_files(files: Mapping[str, Path], dockerfile: str) -> list[str]:
    """The names `@environment_card(name="...")` gives in the `.py` files among ``files``, keyed by POSIX path: in
    those in the Dockerfile's folder, else in all of them, as `card_name_from_source` reads a folder."""
    folder = posixpath.dirname(dockerfile)
    sources = {key: path for key, path in files.items() if key.endswith(".py")}
    near = {key: path for key, path in sources.items() if not folder or key.startswith(f"{folder}/")}
    for chosen in (near, sources):
        found: set[str] = set()
        for path in chosen.values():
            try:
                found |= _card_names_in_source(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
        if found:
            return sorted(found)
    return []


def _scan_environment_card_names(root: Path) -> list[str]:
    found: set[str] = set()
    for py in root.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        found |= _card_names_in_source(text)
    return sorted(found)


def _card_names_in_source(text: str) -> set[str]:
    if "environment_card" not in text:
        return set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for decorator in node.decorator_list:
                if name := _environment_card_name(decorator):
                    found.add(name)
    return found


def _environment_card_name(decorator: ast.expr) -> str | None:
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    fname = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    if fname != "environment_card":
        return None
    for kw in decorator.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None
