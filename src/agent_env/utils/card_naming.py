"""Derive an env name from ``@environment_card(name=...)`` declarations (import-light: the hub imports this)."""

import ast
import posixpath
from collections.abc import Mapping
from pathlib import Path

import httpx

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
    ref_q = f"?ref={ref}" if ref else ""
    resp = client.get(f"https://api.github.com/repos/{owner}/{repo}/contents/{directory}{ref_q}")
    resp.raise_for_status()
    entries = resp.json()
    if not isinstance(entries, list):
        return []
    found: set[str] = set()
    for entry in entries:
        name = entry.get("name") or ""
        if entry.get("type") == "dir":
            found |= set(_github_card_names(client, owner, repo, ref, entry["path"]))
        elif entry.get("type") == "file" and name.endswith(".py") and "test" not in name:
            download_url = entry.get("download_url")
            if download_url:
                found |= _card_names_in_source(client.get(download_url).text)
    return sorted(found)


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
