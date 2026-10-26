import json
from dataclasses import asdict, dataclass, field

import click

from agent_env.a2a_agent.store import A2A_AGENTS_COLLECTION
from agent_env.bundle import BundleError, BundleKind
from agent_env.bundle.installed import checked, installed_bundles
from agent_env.config import get_config
from agent_env.plugins._bundles import InstalledAliases, InstalledBundle, installed_aliases
from agent_env.store import Filter, Sort

_LIMIT = 50


@dataclass
class _Agent:
    id: str
    source: str  # "installed" or "store"
    version: int | None = None  # the store's latest
    aliases: list[str] = field(default_factory=list)
    shadowed_aliases: list[str] = field(default_factory=list)  # those the store's agent of that id takes
    package: str | None = None
    package_version: str | None = None
    bundle: str | None = None

    def origin(self) -> str:
        if self.source == "store":
            return f"the store, v{self.version}"
        return f"{self.package} {self.package_version}, {self.bundle} bundle"


@click.command("list")
@click.option("--limit", default=_LIMIT, show_default=True, type=click.IntRange(min=0),
              help="The most agents from the store to list, newest first; 0 lists them all")
@click.option("--json", "as_json", is_flag=True, help="Print the agents as JSON")
def list_agents(limit: int, as_json: bool):
    """List the agents a task can name in a2a_agent_id: those the installed bundles ship, with the aliases that name
    them, then those in the store, newest first."""
    bundles = installed_bundles()
    aliases = installed_aliases(bundles)
    agents, unreadable = _installed(bundles, aliases)
    more, store_error = 0, None
    try:
        store = get_config().get_document_store()
        for agent in agents:
            agent.shadowed_aliases = [alias for alias in agent.aliases
                                      if store.latest_version(A2A_AGENTS_COLLECTION, alias) is not None]
        docs, total = store.latest_per_id_page(A2A_AGENTS_COLLECTION, Filter({}), sort=Sort.by("created_at_utc"),
                                               limit=limit or None)
        shipped = {agent.id for agent in agents}
        agents += [_Agent(doc["id"], "store", version=doc.get("version")) for doc in docs if doc["id"] not in shipped]
        more = total - len(docs)
    except Exception as e:  # each backend raises its own errors, and the installed agents are worth listing anyway
        store_error = f"{type(e).__name__}: {e}"
    if as_json:
        click.echo(json.dumps({"agents": [asdict(agent) for agent in agents], "refused_aliases": dict(aliases.refused),
                               "more_in_store": more, "store_error": store_error, "unreadable_bundles": unreadable},
                              indent=2))
        return
    _print(agents, aliases, more, store_error, unreadable)


def _installed(bundles: tuple[InstalledBundle, ...], aliases: InstalledAliases) -> tuple[list[_Agent], int]:
    """Each agent the installed bundles ship, and how many bundles can't be read."""
    agents, unreadable = [], 0
    for bundle in bundles:
        try:
            parsed = checked(bundle, load=False) if bundle.root is not None and not bundle.problem else None
        except BundleError:
            parsed = None
        if parsed is None:
            unreadable += 1
            continue
        agents += [_Agent(entry.id, "installed", aliases=list(aliases.naming(entry.id)), package=bundle.package,
                          package_version=bundle.version, bundle=bundle.name)
                   for entry in parsed.entries if entry.kind is BundleKind.AGENT]
    return agents, unreadable


def _print(agents: list[_Agent], aliases: InstalledAliases, more: int, store_error: str | None,
           unreadable: int) -> None:
    rows = [("ID", "ALIASES", "FROM")]
    rows += [(agent.id, ", ".join(f"{alias} (shadowed)" if alias in agent.shadowed_aliases else alias
                                  for alias in agent.aliases), agent.origin()) for agent in agents]
    widths = [max(len(row[i]) for row in rows) for i in range(2)]
    for row in rows:
        click.echo(f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  {row[2]}".rstrip())
    notes = ["A task names an agent in a2a_agent_id by its id, or by an alias."]
    if any(agent.shadowed_aliases for agent in agents):
        notes.append("A shadowed alias names the store's agent of that id instead.")
    if more:
        notes.append(f"The store has {more} more agent{'s' if more != 1 else ''}: --limit 0 lists them all.")
    if store_error:
        notes.append(f"The store can't be read, so only the installed agents are listed: {store_error}")
    if unreadable:
        notes.append(f"{unreadable} installed bundle{'s' if unreadable != 1 else ''} can't be read: agent-env run says "
                     "why.")
    click.echo("\n" + "\n".join(notes))
    if aliases.refused:
        click.echo("\nRefused aliases:\n" + "\n".join(f"  {why}" for why in aliases.refused.values()))
