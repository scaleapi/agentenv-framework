"""Multi environment combining multiple MCP servers and websites."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, ClassVar, Optional

logger = logging.getLogger(__name__)

# Off by default: a bake builds and pushes a multi-GB servicedb image, which is the right
# trade once per universe and the wrong one on every deploy. Callers that know they are
# seeding a reusable universe pass snapshot_after_load=True explicitly; this flag exists
# so an operator can turn it on for a whole worker without a code change.
_SNAPSHOT_AFTER_LOAD_ENV_VAR = "AGENT_ENV_SNAPSHOT_AFTER_LOAD"

# Escape hatch for the derived per-load concurrency cap: pin the old load-everything-at-once
# behaviour, or hold it fixed while bisecting a contention problem.
_LOAD_CONCURRENCY_ENV_VAR = "AGENT_ENV_LOAD_CONCURRENCY"

def _snapshot_after_load_default() -> bool:
    return os.environ.get(_SNAPSHOT_AFTER_LOAD_ENV_VAR, "").strip().lower() in ("1", "true", "yes")

if TYPE_CHECKING:
    from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact
    from agent_env.bundle.authoring import AuthoringContext
    from agent_env.providers.env_state import DatabaseStateProvider

from agent_env.entity_refs import EntityRef
from agent_env.env.env import Env, gateway_url_of
from agent_env.env.gateway import GatewayMode
from agent_env.env.envs._deployment import (
    as_builtin, builtin_provider_for, close_deployed, close_replaced, deploy_refusal, deploy_through_provider, host_staging_refusal,
    plugin_provider_like_a_builtin, provider_or_class,
)
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.store.ids import derive_id
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER
from agent_env.attribution import Attribution


class MultiEnv(Env):
    type: ClassVar[str] = "multi"
    description = "Multi environment combining multiple MCP servers and websites, deployed through the environment provider its env_provider_type names"
    toml_refs: ClassVar[tuple[EntityRef, ...]] = (EntityRef.env("mcp_server_envs[]", env_type=MCPServerEnv.type),
                                                  EntityRef.env("website_envs[]", env_type=WebsiteEnv.type))
    toml_keys: ClassVar[dict[str, type]] = {"mcp_server_envs": list, "website_envs": list, "name": str,
                                            "env_provider_type": str}

    def __init__(self, id: str, version: Optional[int], mcp_server_envs: list[MCPServerEnv], website_envs: list[WebsiteEnv] | None = None, metadata: Optional[dict[str, str]] = None, name: Optional[str] = None, env_provider_type: str = "gateway"):
        super().__init__(id, version, metadata=metadata)
        if name is not None and (not name or any(ch.isspace() for ch in name)):
            raise ValueError("name must be non-empty with no whitespace")
        if not env_provider_type:
            raise ValueError("env_provider_type cannot be empty")
        self.name = name
        self.env_provider_type = env_provider_type  # the children's own types are ignored: this one deploys them all
        self.mcp_server_envs = mcp_server_envs
        self.website_envs = website_envs or []
        self._sandbox = None
        self._gateway_url = None
        self._gateway_mode = GatewayMode.PERFORMANCE
        self._env_provider = None
        self._mcp_server_name: Optional[str] = None
        self._deployed: Optional[DeployedEnv] = None
        self._replaced: list[tuple] = []  # the provider and sandbox of each deployment a later deploy replaced, for close()

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["mcp_server_envs"] = [
            {"id": env.id, "type": env.type, "version": env.version}
            for env in self.mcp_server_envs
        ]
        base["website_envs"] = [
            {"id": env.id, "type": env.type, "version": env.version}
            for env in self.website_envs
        ]
        if self.name:
            base["name"] = self.name
        base["env_provider_type"] = self.env_provider_type
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "MultiEnv":
        mcp_server_envs = [
            Env.get(ref["id"], version=ref["version"])
            for ref in data.get("mcp_server_envs", [])
        ]
        website_envs = [
            Env.get(ref["id"], version=ref["version"])
            for ref in data.get("website_envs", [])
        ]
        return cls(id=data["id"], version=data.get("version"), mcp_server_envs=mcp_server_envs, website_envs=website_envs, metadata=data.get("metadata", {}),
                   name=data.get("name"), env_provider_type=data.get("env_provider_type", "gateway"))

    @classmethod
    def from_toml(cls, data: dict, ctx: AuthoringContext) -> MultiEnv:
        """Write the env authored as ``data`` (its env.toml, with ``mcp_server_envs`` and ``website_envs`` resolved to
        env ids) under ``ctx.id`` and return it, unless deploying it would be refused."""
        fields = cls.accept_toml(data, ctx)
        env = dict(mcp_server_envs=[ctx.env(ref, MCPServerEnv) for ref in fields.get("mcp_server_envs", [])],
                   website_envs=[ctx.env(ref, WebsiteEnv) for ref in fields.get("website_envs", [])],
                   name=fields.get("name"), env_provider_type=fields["env_provider_type"])
        if refusal := cls(id=ctx.id, version=None, **env).deploy_refusal():
            ctx.refuse([refusal])
        return cls.put(id=ctx.id, **env)

    @classmethod
    def accept_toml(cls, data: dict, ctx: AuthoringContext) -> dict:
        """The keys of ``data``, an env.toml, a multi env takes. Raises BundleError listing every problem."""
        fields, problems = ctx.accepted_env(data, cls)
        if not (fields.get("mcp_server_envs") or fields.get("website_envs")):
            problems.append(ctx.config_problem("a multi env needs at least one env in mcp_server_envs or website_envs"))
        name = fields.get("name")
        if name is not None and (not name or any(ch.isspace() for ch in name)):
            problems.append(ctx.config_problem(f"name must be non-empty with no whitespace, not {name!r}"))
        if problems:
            ctx.refuse(problems)
        return fields

    async def deploy(self, ttl_seconds: int = 10800, disk_size_gb: float = 10, gateway_mode: GatewayMode = GatewayMode.PERFORMANCE, cpu: float | None = None, memory_mb: int | None = None, sandbox_type: str | None = None, env_state_type: str | None = None, env_state_instance_id: str | None = None, *, attribution: Optional[Attribution] = None) -> DeployedEnv:
        deployed_env = await deploy_through_provider(
            self, environment_name=None, ttl_seconds=ttl_seconds, sandbox_type=sandbox_type,
            disk_size_gb=disk_size_gb, gateway_mode=gateway_mode, cpu=cpu, memory_mb=memory_mb,
            env_state_type=env_state_type, env_state_instance_id=env_state_instance_id, attribution=attribution,
        )
        self._gateway_mode = gateway_mode
        self._mcp_server_name = deployed_env.mcp_server_name
        _hand_provider_to_child_envs(self)
        _hand_record_to_child_envs(self, deployed_env)
        return deployed_env

    def deploy_refusal(self, **options) -> str | None:
        """Why deploy() would refuse these options before building anything, or None; raises, as deploy() does, for a type this process can't find."""
        return deploy_refusal(self, provider_or_class(self), options)

    @classmethod
    async def from_deployed_env(cls, deployed: DeployedEnv) -> MultiEnv:
        env = Env.get(deployed.env_id, deployed.env_version)
        if not isinstance(env, MultiEnv):
            raise TypeError(f"Expected MultiEnv, got {type(env).__name__}")
        provider = builtin_provider_for(env.env_provider_type) or plugin_provider_like_a_builtin(env.env_provider_type)
        if builtin := as_builtin(provider):
            env._sandbox = await builtin._reattach(env, deployed)
        # Any other plugin's deployment holds its own sandboxes: its children load through its card, and nothing is reattached.
        env._env_provider = provider
        env._gateway_url = gateway_url_of(deployed)
        env._deployed = deployed
        _hand_provider_to_child_envs(env)
        _hand_record_to_child_envs(env, deployed)
        env._instance_id = deployed.instance_id
        env._mcp_server_name = deployed.mcp_server_name
        return env

    async def load_environment_artifact(self, environment_artifact: EnvironmentArtifact) -> None:
        # A built-in's deploy, or a caller wiring one up by hand, leaves a gateway; a plugin's leaves only its record.
        if self._gateway_url is None and self._deployed is None:
            raise RuntimeError("Environment not deployed - call deploy() first")
        # MCP servers are checked first; if a website shares the same environment_name, the MCP server wins.
        for env in self.mcp_server_envs:
            if env.environment_name == environment_artifact.environment_name:
                await env.load_environment_artifact(environment_artifact)
                return
        for env in self.website_envs:
            if env.environment_name == environment_artifact.environment_name:
                await env.load_environment_artifact(environment_artifact)
                return
        logger.warning(f"No env with environment_name '{environment_artifact.environment_name}' — skipping")

    async def load_environment_universe_artifact(
        self,
        environment_universe_artifact: EnvironmentUniverseArtifact,
        snapshot_after_load: bool | None = None,
    ) -> LoadEnvironmentUniverseArtifactResult:
        """Seed this env from a universe artifact, preferring a pre-baked snapshot.

        A snapshot restore swaps a pre-ingested servicedb image; a miss re-ingests every
        service over HTTP under a fixed 600s per-service cap.

        Restore is NOT reliably faster. Measured on a 13-service / ~8.7GB universe:
        restore ~4min (an image pull plus recreating every service container) versus
        ~3.5min to re-ingest on 8 vCPU. What restore buys is a bound — no per-service
        timeout to squeeze under — and far less CPU: the same re-ingest FAILS on the
        1-vCPU sandbox default, where a 409MB service blew its cap while a 3.5GB one
        succeeded. Re-ingest cost tracks record count, not bytes, so the busier the
        universe the better restore looks.

        ``snapshot_after_load`` bakes a reusable snapshot after a successful re-ingest so
        the next load of the same (images, universe) pair can restore instead. It defaults
        to the ``AGENT_ENV_SNAPSHOT_AFTER_LOAD`` config flag (off), because a bake builds
        and pushes a multi-GB servicedb image (~5min, ~2.3GB stored) — worth it to make a
        universe cheap to re-serve, not worth it on a one-off deploy. The bake is
        best-effort and never fails a load that already succeeded.
        """
        import uuid
        from agent_env.env.env import LoadEnvironmentUniverseArtifactResult
        from agent_env.env.snapshot_store import compute_env_fingerprint, get_env_snapshot_store

        # Refused before any service is touched, rather than after they all load.
        if (refusal := host_staging_refusal(self, "Staging a universe's metadata files onto the env's host")) \
                and environment_universe_artifact.get_metadata():
            raise RuntimeError(refusal)

        # Check for a clean snapshot before doing slow REST-based loading.

        # Shortcutting data-load by using a snapshot is only supported when the env_state_type is local_postgres -
        # a remote-backed deploy has no local servicedb container to swap, so it falls through to
        # the normal load. So does a plugin's deployment, which has no state provider of ours.
        state_provider = _gateway_state_provider(self)
        can_restore = state_provider is not None and state_provider.supports_restore_from_snapshot
        # Keyed on the service images rather than self.id, so re-registering the same
        # servers under a new env id still hits (and a rebuilt server correctly misses).
        env_fingerprint = compute_env_fingerprint(self) if can_restore else None
        snapshot = get_env_snapshot_store().get_clean(
            self.id,
            environment_universe_artifact.id,
            environment_universe_artifact.version,
            env_fingerprint=env_fingerprint,
        ) if can_restore else None
        restored_from_snapshot = False
        snapshot_baked: bool | None = None
        snapshot_bake_error: str | None = None

        if snapshot:
            logger.info(f"Found clean snapshot for env={self.id} universe={environment_universe_artifact.id}: artifact={snapshot.db_image_artifact_id} v{snapshot.db_image_artifact_version} instance={snapshot.instance_id}")
            await self._load_from_snapshot(snapshot)
            restored_from_snapshot = True
            # A restore swaps the whole database at once, so per-service resume state from
            # any earlier partial re-ingest no longer describes anything real.
            if self._instance_id:
                try:
                    from agent_env.env.store import get_env_instance_store
                    get_env_instance_store().clear_loaded_environments(self._instance_id)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Could not clear resume state after a snapshot restore: %s", e)
        else:
            import asyncio
            import time
            environment_artifacts = environment_universe_artifact.get_environment_artifacts()
            total = len(environment_artifacts)
            # WARNING, not info: this is the path that can FAIL. Every service is
            # re-ingested over HTTP under a fixed per-service timeout, and one service
            # exceeding it fails the whole load -- which on an under-provisioned VM is the
            # normal outcome, not the edge case. The fingerprint belongs in the log
            # because "no snapshot was ever baked" and "one exists but the service images
            # changed" look identical from the outside and call for different fixes.
            logger.warning(
                "No clean snapshot for env=%s (fingerprint=%s) universe=%s v%s — "
                "re-ingesting all %d services over HTTP, each under its own timeout.%s",
                self.id, env_fingerprint, environment_universe_artifact.id,
                environment_universe_artifact.version, total,
                " Bake a snapshot to make subsequent loads an image swap instead." if can_restore else "",
            )

            # Bound how many load at once. Unbounded, every service races every other for
            # the same cores, and since each one is separately capped at
            # DATA_PLANE_LOAD_TIMEOUT_S, contention doesn't just slow the load down -- it
            # fails it. Measured on this universe: 13-way on 1 vCPU gives each service
            # ~1/13th of a core and a 409MB service exceeds its 600s cap, while a 3.5GB one
            # finishes; the cost tracks record count, not bytes. Throttling trades total
            # wall-clock (which nothing caps) for per-service headroom (which is capped).
            # Resume an interrupted load instead of re-running all of it. A step retry
            # re-enters this step from scratch, and because the load is destructive
            # (reset-then-add per service) that meant re-wiping and re-ingesting every
            # service -- three attempts x the full universe is how one slow service became
            # a ~44 minute failure.
            already = await self._resumable_environments(environment_universe_artifact, total)
            if already:
                environment_artifacts = [
                    a for a in environment_artifacts if a.environment_name not in already
                ]
                logger.warning(
                    "Resuming an interrupted load of %s v%s: skipping %d already-loaded "
                    "service(s) %s, loading the remaining %d",
                    environment_universe_artifact.id, environment_universe_artifact.version,
                    len(already), sorted(already), len(environment_artifacts),
                )
                total = len(environment_artifacts)
                if not environment_artifacts:
                    logger.info("Every service was already loaded; nothing to re-ingest")

            limit = await self._load_concurrency_limit(total) if total else 1
            semaphore = asyncio.Semaphore(limit)

            async def _load_one(idx: int, artifact):
                name = artifact.environment_name
                async with semaphore:
                    # Timed inside the semaphore: the per-service timeout only starts once
                    # the work does, so queue time must not be charged against it. A
                    # duration here is comparable to DATA_PLANE_LOAD_TIMEOUT_S.
                    t0 = time.monotonic()
                    logger.info(f"[{idx}/{total}] loading {name} ...")
                    print(f"[{idx}/{total}] loading {name} ...", flush=True)
                    try:
                        await self.load_environment_artifact(artifact)
                    except BaseException as e:  # noqa: BLE001 — includes CancelledError
                        dt = time.monotonic() - t0
                        logger.error(f"[{idx}/{total}] FAIL {name} after {dt:.0f}s: {type(e).__name__}: {e}")
                        print(f"[{idx}/{total}] ✗ {name} FAILED after {dt:.0f}s: {type(e).__name__}: {e}", flush=True)
                        raise
                    dt = time.monotonic() - t0
                    # Recorded per service, immediately, rather than once at the end: the
                    # whole point is that a load which dies partway leaves usable progress.
                    self._record_loaded_environment(environment_universe_artifact, name)
                    logger.info(f"[{idx}/{total}] OK {name} in {dt:.0f}s")
                    print(f"[{idx}/{total}] ✓ {name} loaded in {dt:.0f}s", flush=True)
                    return dt

            # Key failures by position, not environment_name: a duplicate service
            # must not let one success mask another's failure.
            outcomes = await asyncio.gather(
                *(_load_one(i, a) for i, a in enumerate(environment_artifacts, 1)),
                return_exceptions=True,
            )
            failed = [
                (i, environment_artifacts[i - 1].environment_name, outcome)
                for i, outcome in enumerate(outcomes, 1)
                if isinstance(outcome, BaseException)
            ]
            if failed:
                raise RuntimeError(
                    f"Failed to load {len(failed)}/{total} services: "
                    + ", ".join(
                        f"[{i}/{total}] {name} ({type(e).__name__}: {e})"
                        for i, name, e in failed
                    )
                )
        if self._instance_id:
            from agent_env.env.store import update_env_instance_environment_universe
            update_env_instance_environment_universe(self._instance_id, environment_universe_artifact.id, environment_universe_artifact.version)

        # Bake the snapshot that makes the NEXT load of this (images, universe) pair an
        # image swap. Deliberately placed here: after the universe stamp, which
        # EnvSnapshot.create reads back off the instance record, and before the metadata
        # download, so the captured PGDATA reflects the data load and nothing else.
        if not restored_from_snapshot and can_restore:
            if snapshot_after_load is None:
                snapshot_after_load = _snapshot_after_load_default()
            if snapshot_after_load:
                snapshot_baked, snapshot_bake_error = await self._bake_snapshot_after_load(
                    environment_universe_artifact
                )

        metadata_filepaths: dict[str, str] = {}
        metadata_artifacts = environment_universe_artifact.get_metadata()
        if metadata_artifacts:
            if self._sandbox is None:
                raise RuntimeError("Environment not deployed - call deploy() first")
            dest_dir = f"/tmp/metadata-{uuid.uuid4().hex[:8]}"
            if self._sandbox.mode != SANDBOX_MODE_CONTAINER:
                await self._sandbox.exec_script(f"mkdir -p {dest_dir}")
            for key, file_artifact in metadata_artifacts.items():
                dest_path = f"{dest_dir}/{file_artifact.filename}"
                logger.info(f"Downloading metadata '{key}' -> {dest_path}")
                if self._sandbox.mode == SANDBOX_MODE_CONTAINER:
                    await self._sandbox.write_file_from_object(file_artifact.object_url, dest_path)
                else:
                    await self._sandbox.load_object_file(file_artifact.object_url, dest_path)
                metadata_filepaths[key] = dest_path
                logger.info(f"Successfully downloaded metadata '{key}'")
        return LoadEnvironmentUniverseArtifactResult(
            metadata_filepaths=metadata_filepaths,
            restored_from_snapshot=restored_from_snapshot,
            snapshot_db_image_artifact_id=snapshot.db_image_artifact_id if snapshot else None,
            snapshot_baked=snapshot_baked,
            snapshot_bake_error=snapshot_bake_error,
        )

    async def _resumable_environments(self, environment_universe_artifact, total: int) -> set[str]:
        """Services safe to skip because a previous attempt already loaded them.

        Gated on the changelog being EMPTY, which is what keeps this from quietly breaking
        the documented contract that loading a universe is destructive. Two cases have to be
        told apart, and the changelog is what distinguishes them:

        * A retry of an interrupted load. Nothing has mutated the data (load triggers are
          installed only after each service's own load), so the changelog is empty and the
          recorded services are still exactly what the artifact says. Skipping is correct
          and saves re-wiping them.
        * A deliberate re-load to reset an env an agent has been working in. The changelog
          is non-empty, so NOTHING is skipped and every service is wiped and re-seeded --
          today's behaviour, which callers depend on.

        Coarse on purpose: the changelog is global, not per-service, so any mutation
        anywhere disables resume entirely. That errs toward doing redundant work rather than
        toward leaving stale data in place, which is the right way to be wrong here.
        """
        if not self._instance_id or self._sandbox is None:
            return set()
        if self._sandbox.mode == SANDBOX_MODE_CONTAINER:
            return set()
        try:
            from agent_env.env.snapshot_store import _check_changelog_empty
            from agent_env.env.store import get_env_instance_store

            recorded = set(get_env_instance_store().get_loaded_environments(
                self._instance_id, environment_universe_artifact.id, environment_universe_artifact.version,
            ))
            if not recorded:
                return set()
            if not await _check_changelog_empty(self._sandbox):
                logger.info(
                    "Not resuming: the changelog is non-empty, so this env's data has been "
                    "modified since it was loaded and every service must be re-seeded"
                )
                return set()
            if len(recorded) >= total:
                # A complete prior load with an untouched changelog. Re-seeding would be a
                # no-op, but skipping everything would silently turn a caller's explicit
                # load into nothing -- so do the work rather than guess at intent.
                logger.info(
                    "All %d services are recorded as loaded and the changelog is clean; "
                    "re-loading anyway rather than turning an explicit load into a no-op",
                    total,
                )
                return set()
            return recorded
        except Exception as e:  # noqa: BLE001 -- an optimisation, not a dependency
            logger.warning("Could not determine resumable services (%s); loading all of them", e)
            return set()

    def _record_loaded_environment(self, environment_universe_artifact, environment_name: str) -> None:
        """Persist one service's completion. Best-effort: losing a record costs redundant
        work on a retry, whereas raising here would fail a service that actually loaded."""
        if not self._instance_id:
            return
        try:
            from agent_env.env.store import get_env_instance_store

            get_env_instance_store().record_loaded_environment(
                self._instance_id, environment_universe_artifact.id,
                environment_universe_artifact.version, environment_name,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not record %s as loaded (%s); a retry will re-load it",
                environment_name, e,
            )

    async def _load_concurrency_limit(self, total: int) -> int:
        """How many services may load at once.

        Derived from the sandbox's actual core count rather than a constant, because the
        failure this prevents is contention-driven: the per-service data-plane timeout is
        fixed, so what matters is how much CPU each concurrent load can get. Two in flight
        per core keeps some I/O overlap while leaving each service a real share.

        At >= ceil(total/2) cores this is a no-op -- a well-provisioned VM keeps loading
        everything at once, which is what the measured 8 vCPU run did. It only bites on the
        small boxes where unbounded concurrency was turning a slow load into a failed one.

        ``AGENT_ENV_LOAD_CONCURRENCY`` overrides it outright, for pinning the old behaviour
        or bisecting a contention problem. Falls back to loading everything at once if the
        core count can't be read: that is today's behaviour, so a probe failure must not
        quietly change how a load runs.
        """
        override = os.environ.get(_LOAD_CONCURRENCY_ENV_VAR, "").strip()
        if override:
            try:
                value = int(override)
                if value < 1:
                    raise ValueError(f"must be >= 1, got {value}")
            except ValueError as e:
                # Loud, not silent: a typo'd override that fell back to the default would
                # make a contention experiment quietly measure the wrong thing.
                raise ValueError(
                    f"{_LOAD_CONCURRENCY_ENV_VAR}={override!r} is not a positive integer: {e}"
                ) from None
            limit = min(value, total)
            logger.info("Load concurrency %d/%d (from %s)", limit, total, _LOAD_CONCURRENCY_ENV_VAR)
            return limit

        cores = None
        if self._sandbox is not None:
            try:
                cores = int((await self._sandbox.exec_script("nproc")).strip())
            except Exception as e:  # noqa: BLE001 — a probe, not a dependency
                logger.warning("Could not read sandbox core count (%s); loading all %d services at once", e, total)
        if not cores or cores < 1:
            return total

        limit = max(1, min(total, cores * 2))
        logger.info(
            "Load concurrency %d/%d services (sandbox has %d core(s))", limit, total, cores,
        )
        return limit

    async def _bake_snapshot_after_load(
        self, environment_universe_artifact: EnvironmentUniverseArtifact
    ) -> tuple[bool, str | None]:
        """Capture a reusable clean snapshot of what was just loaded.

        Best-effort by contract: the data load already succeeded, and the caller wants a
        seeded env more than it wants a cache entry. Any failure here is logged and
        reported in the result, never raised — otherwise adding a cache would turn
        working loads into failures.

        Returns ``(baked, error)``.
        """
        import asyncio

        from agent_env.env.snapshot_store import EnvSnapshot

        if not self._instance_id:
            # create() resolves the env and its loaded universe from the instance record,
            # so an unregistered env (ad-hoc/local deploy) has nothing to key a snapshot on.
            msg = "no instance_id on this env; snapshot capture needs a registered env instance"
            logger.warning("Skipping snapshot bake for env=%s: %s", self.id, msg)
            return False, msg
        try:
            logger.info(
                "Baking clean snapshot for env=%s universe=%s v%s (instance=%s) so the "
                "next load can skip the re-ingest...",
                self.id, environment_universe_artifact.id,
                environment_universe_artifact.version, self._instance_id,
            )
            snapshot = await EnvSnapshot.create(self._instance_id)
        except asyncio.CancelledError:
            # Cancellation is control flow, not a bake failure. Swallowing it here would
            # let a cancelled/timed-out activity carry on and report success.
            raise
        except Exception as e:  # noqa: BLE001 — a cache miss must not fail a good load
            logger.warning(
                "Snapshot bake failed for env=%s universe=%s: %s: %s (the load itself "
                "succeeded; the next load will take the slow path)",
                self.id, environment_universe_artifact.id, type(e).__name__, e,
                exc_info=True,
            )
            return False, f"{type(e).__name__}: {e}"
        if not snapshot.is_clean:
            # create() derives is_clean from an empty changelog. Landing here means
            # something mutated the data between load and capture, so the snapshot exists
            # but get_clean will never serve it — say so rather than reporting success.
            msg = (
                "captured snapshot is dirty (changelog non-empty), so it will not be "
                "reused; something wrote to the env between load and capture"
            )
            logger.warning("Snapshot bake for env=%s: %s", self.id, msg)
            return False, msg
        logger.info(
            "Baked clean snapshot for env=%s universe=%s v%s: image=%s v%s",
            self.id, environment_universe_artifact.id, environment_universe_artifact.version,
            snapshot.db_image_artifact_id, snapshot.db_image_artifact_version,
        )
        return True, None

    async def load_file_artifact_universe(self, file_artifact_universe: "Any", destination_path: Optional[str] = None,
                                          ) -> "LoadFileArtifactUniverseResult":
        if refusal := host_staging_refusal(self, "Staging files onto the env's host"):
            raise RuntimeError(refusal)
        return await super().load_file_artifact_universe(file_artifact_universe, destination_path)

    async def load_artifact(self, artifact):
        from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact
        if isinstance(artifact, EnvironmentUniverseArtifact):
            return await self.load_environment_universe_artifact(artifact)
        if isinstance(artifact, EnvironmentArtifact):
            return await self.load_environment_artifact(artifact)
        raise ValueError(f"{type(self).__name__} '{self.id}' cannot load artifact '{getattr(artifact, 'id', '?')}' of type '{getattr(artifact, 'type', '?')}' — expected a EnvironmentArtifact or EnvironmentUniverseArtifact")

    async def _load_from_snapshot(self, snapshot) -> None:
        """Swap the servicedb container with a pre-loaded snapshot image. Only supported when both the current deploy and the snapshot has env_state_type set to local_postgres"""
        import asyncio
        from agent_env.artifact import Artifact, DockerImageArtifact
        from agent_env.env.env import Env
        from agent_env.env.envs.service_db import DB_USER
        from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT
        from agent_env.providers import EnvironmentGatewayProvider, MCPServerConfig, WebsiteConfig
        from agent_env.env.envs.service_db import DATABASE_SERVICE_NAME, DB_MCP_SERVICE_NAME, PGWEB_SERVICE_NAME
        from agent_env.providers.env_providers.constants import DOCKER_COMPOSE_PATH, GATEWAY_APP_DIR, GATEWAY_SERVICE_NAME
        from agent_env.providers.env_state import LocalPostgresStateProvider
        from agent_env.config import get_config

        if self._sandbox is None:
            raise RuntimeError("Environment not deployed - call deploy() first")
        if self._sandbox.mode == SANDBOX_MODE_CONTAINER:
            raise NotImplementedError("Snapshot loading not supported on container mode")
        db_image = Artifact.get(snapshot.db_image_artifact_id, snapshot.db_image_artifact_version)
        if not isinstance(db_image, DockerImageArtifact):
            raise RuntimeError(f"Snapshot artifact {snapshot.db_image_artifact_id} is not a DockerImageArtifact")
        logger.info(f"Loading snapshot image {db_image.image_name} into sandbox...")
        await self._sandbox.load_docker_images([db_image])

        # Regenerate docker-compose.yml with the snapshot image (no init-schemas volume needed)
        logger.info("Regenerating docker-compose.yml with snapshot image...")
        config = get_config()
        gateway_env = Env.get(config.default_gateway_env_id)
        service_db = Env.get(config.default_service_db_env_id)
        service_db_config = service_db.to_config()
        service_db_config.db_image = db_image.image_name

        gw = EnvironmentGatewayProvider()
        gw._sandbox = self._sandbox
        gw._state_provider = LocalPostgresStateProvider(service_db_config=service_db_config)
        gw._state_instance = LocalPostgresStateProvider.default_instance()
        mcp_servers = [MCPServerConfig(image=env.docker_image_artifact.image_name, environment_name=env.environment_name) for env in self.mcp_server_envs]
        website_configs = [
            WebsiteConfig(backend_image=env.backend_docker_image_artifact.image_name, frontend_image=env.frontend_docker_image_artifact.image_name, environment_name=env.environment_name)
            for env in self.website_envs
        ]
        if website_configs:
            website_browser_env = Env.get(config.default_website_browser_env_id)
            mcp_servers = list(mcp_servers) + [MCPServerConfig(image=website_browser_env.docker_image_artifact.image_name, environment_name=website_browser_env.environment_name)]

        compose_content = gw.create_docker_compose(
            mcp_servers=mcp_servers,
            gateway_image=gateway_env.docker_image_artifact.image_name,
            gateway_port=AGENT_ENV_GATEWAY_MCP_PORT, website_configs=website_configs or None,
            gateway_mode=self._gateway_mode,
            state_provider=gw._state_provider,  # local-only, set above
            state_instance=gw._state_instance,
            host_ips=self._sandbox.host_ips,
            mcp_server_name=self._mcp_server_name or self.name,
        )
        compose_content = compose_content.replace(
            "    volumes:\n      - ./init-schemas.sql:/docker-entrypoint-initdb.d/init-schemas.sql\n", "",
        )
        write_script = f'''cat > {DOCKER_COMPOSE_PATH} << 'COMPOSE_EOF'
{compose_content}
COMPOSE_EOF'''
        await self._sandbox.exec_script(write_script)

        # Remove old servicedb container and its anonymous volume, then start fresh from snapshot image
        logger.info("Recreating servicedb container with snapshot image...")
        await self._sandbox.exec_script(
            f"cd {GATEWAY_APP_DIR} && docker compose rm -sf -v {DATABASE_SERVICE_NAME} && docker compose up -d {DATABASE_SERVICE_NAME} 2>&1"
        )

        # Wait for servicedb to be healthy
        logger.info("Waiting for servicedb to be healthy...")
        for i in range(60):
            try:
                await self._sandbox.exec_script(
                    f"docker compose -f {DOCKER_COMPOSE_PATH} exec -T {DATABASE_SERVICE_NAME} pg_isready -U {DB_USER}",
                )
                logger.info(f"servicedb healthy after {i + 1}s")
                break
            except Exception:
                if i % 10 == 0:
                    logger.info(f"Still waiting for servicedb... ({i}s)")
                await asyncio.sleep(1)
        else:
            raise RuntimeError("servicedb not healthy after 60s")

        # Recreate non-servicedb services so they reconnect to the new servicedb.
        # Uses `up -d --force-recreate` which respects depends_on ordering
        # (MCP servers start before gateway), unlike `restart` which is unordered.
        environment_names = list(dict.fromkeys(
            [env.environment_name for env in self.mcp_server_envs] + [env.environment_name for env in self.website_envs]
        ))
        svc_list = " ".join(environment_names + [GATEWAY_SERVICE_NAME, PGWEB_SERVICE_NAME, DB_MCP_SERVICE_NAME])
        logger.info(f"Recreating services: {svc_list}")
        await self._sandbox.exec_script(
            f"cd {GATEWAY_APP_DIR} && docker compose up -d --force-recreate {svc_list}"
        )
        await gw._wait_for_gateway(self._sandbox, AGENT_ENV_GATEWAY_MCP_PORT)

        # Install changelog triggers for each service
        for name in environment_names:
            await gw.install_changelog_triggers(name)

    async def validate(self, on_progress: Callable[[str], None] | None = None) -> str:
        """Run the basic env-card validator task and return the task instance ID.

        Deploys the env, fetches + persists its composed EnvironmentCard, then tears down.
        """
        from agent_env.task import Task
        from agent_env.task_step import DeployEnvTaskStep, VerifyEnvironmentCardStep

        task_id = derive_id(self.id, f"validate-v{self.version}")
        task = Task.put(
            id=task_id,
            steps=[
                DeployEnvTaskStep(id=f"{task_id}-deploy", version=None, env_id=self.id, env_version=self.version),
                VerifyEnvironmentCardStep(id=f"{task_id}-envcard", version=None, env_id=self.id),
            ],
        )
        logger.info(f"Created validation task: {task.id} version={task.version}")

        log = on_progress or (lambda msg: None)
        on_start = lambda i, total, step, ctx: log(f"Running step [{i+1}/{total}]: {step.type}...")
        on_complete = lambda i, total, step, ctx, dur: log(f"Completed step [{i+1}/{total}]: {step.type} [{dur:.1f}s]")
        context = await task.run(on_step_start=on_start, on_step_complete=on_complete)

        for deployed_env in context.deployed_envs:
            try:
                await close_deployed(deployed_env, MultiEnv)
            except Exception as e:
                logger.warning(f"Failed to clean up env {deployed_env.env_id}: {e}")
        return context.instance_id

    async def validate_universe_compatibility(self, universe_artifact_id: str, universe_artifact_version: int | None = None, on_progress: Callable[[str], None] | None = None) -> str:
        """Run universe compatibility validation and return the task instance ID.

        Two verdicts are produced: the programmatic verifier (``compare_dicts``) and an LLM
        agent-judge that diffs the original/export universes for *semantic* (major) differences the
        programmatic comparison misses. The round-trip step bundles original/export1/export2 into a
        FileArtifactUniverse, which is staged onto a judge agent. The judge's flagged differences are
        merged into the ``UNIVERSE_COMPATIBILITY`` result as per-service critical issues — so they
        surface (and gate ``compatible``) the same way programmatic issues do.
        """
        from agent_env.task import Task
        from agent_env.task_step import (
            CombineUniverseVerdictsStep,
            DeployAgentTaskStep,
            DeployEnvTaskStep,
            LoadArtifactTaskStep,
            PromptAgentTaskStep,
            RubricsVerifierTaskStep,
            VerifyUniverseLoadExportRoundtripStep,
        )
        from agent_env.task_step.task_steps.multienv_validator.verify_universe_agent_judge import (
            JUDGE_SYSTEM_PROMPT,
            UNIVERSES_DIR,
            build_criteria,
            build_user_prompt,
        )
        from agent_env.artifact import EnvironmentUniverseArtifact

        universe = EnvironmentUniverseArtifact.get(universe_artifact_id, universe_artifact_version)
        task_id = VerifyUniverseLoadExportRoundtripStep.validation_id(
            "validate-universe-compat", self.id, self.version, universe.id, universe.version
        )
        environment_names = [sa.environment_name for sa in universe.get_environment_artifacts()]
        fau_id = VerifyUniverseLoadExportRoundtripStep.file_artifact_universe_id(self.id, self.version, universe.id, universe.version)
        judge_name = "universe-judge"
        judge_prompt_id = f"{task_id}-judge-prompt"
        judge_verifier_id = f"{task_id}-judge-rubric"
        # Steps run sequentially (depends_on=None ⇒ depends on all prior steps), so the FAU exists
        # before it is loaded, and the judge is prompted before it is graded. The round-trip step
        # computes the programmatic verdict but does NOT persist it (persist_result=False); the final
        # CombineUniverseVerdictsStep is the single authoritative writer, so a crash before the judge
        # merges can't leave an ungated half-written result.
        steps = [
            DeployEnvTaskStep(id=f"{task_id}-deploy", version=None, env_id=self.id, env_version=self.version),
            VerifyUniverseLoadExportRoundtripStep(id=f"{task_id}-roundtrip", version=None, env_id=self.id, universe_artifact_id=universe.id, universe_artifact_version=universe.version, emit_file_artifact_universe=True, persist_result=False),
            DeployAgentTaskStep(id=f"{task_id}-judge-deploy", version=None, agent_name=judge_name, system_prompt=JUDGE_SYSTEM_PROMPT, ttl_seconds=1800),
            LoadArtifactTaskStep(id=f"{task_id}-judge-load", version=None, agent_name=judge_name, artifact_id=fau_id, destination_path=UNIVERSES_DIR),
            PromptAgentTaskStep(id=f"{task_id}-judge-prompt", version=None, agent_name=judge_name, prompt=build_user_prompt(), prompt_id=judge_prompt_id),
            RubricsVerifierTaskStep(id=judge_verifier_id, version=None, criteria=build_criteria(environment_names), prompt_id=judge_prompt_id, agent_name=judge_name, verifier_id=judge_verifier_id),
            CombineUniverseVerdictsStep(id=f"{task_id}-combine", version=None, env_id=self.id, universe_artifact_id=universe.id, universe_artifact_version=universe.version, judge_verifier_id=judge_verifier_id),
        ]
        task = Task.put(id=task_id, steps=steps)
        logger.info(f"Created validation task: {task.id} version={task.version}")

        log = on_progress or (lambda msg: None)
        on_start = lambda i, total, step, ctx: log(f"Running step [{i+1}/{total}]: {step.type}...")
        on_complete = lambda i, total, step, ctx, dur: log(f"Completed step [{i+1}/{total}]: {step.type} [{dur:.1f}s]")
        context = await task.run(on_step_start=on_start, on_step_complete=on_complete)

        for deployed_env in context.deployed_envs:
            try:
                await close_deployed(deployed_env, MultiEnv)
            except Exception as e:
                logger.warning(f"Failed to clean up env {deployed_env.env_id}: {e}")
        # Tear down any judge agent (ttl is a backstop if this is skipped).
        for deployed_agent in context.deployed_agents:
            if deployed_agent.sandbox_id:
                try:
                    from agent_env.providers import build_sandbox_provider, get_agent_sandbox_provider
                    provider = build_sandbox_provider(deployed_agent.sandbox_type) if deployed_agent.sandbox_type else get_agent_sandbox_provider()
                    sandbox = await provider.get_sandbox(deployed_agent.sandbox_id)
                    await sandbox.terminate()
                except Exception as e:
                    logger.warning(f"Failed to clean up agent sandbox {deployed_agent.sandbox_id}: {e}")

        # Note: the merged UNIVERSE_COMPATIBILITY result is written by CombineUniverseVerdictsStep
        # (the final task step), so it persists wherever the Task runs — not only via this method.
        return context.instance_id

    async def close(self) -> None:
        await close_replaced(self)
        if self._env_provider is not None:
            try:
                await self._env_provider.close()
                self._env_provider = None
            except Exception as e:
                logger.warning(f"Failed to close env provider, kept for the next close(): {e}")
        if self._sandbox is not None:
            try:
                await self._sandbox.terminate()
            except BaseException as e:
                logger.warning(f"Failed to terminate sandbox {self._sandbox.sandbox_id}: {e}")
            self._sandbox = None


def _hand_provider_to_child_envs(env: MultiEnv) -> None:
    """Point each child env at the MultiEnv's deployment: the provider and URL its loads use, and behind our gateway the sandbox they
    stage into. A plugin's provider leaves them no sandbox, so they load through the record's card."""
    provider = env._env_provider
    builtin = as_builtin(provider)
    for child in env.mcp_server_envs:
        child._sandbox = (builtin.environment_sandbox(child.environment_name) if builtin else None) or env._sandbox
        child._gateway_url = env._gateway_url
        child._env_provider = provider
    for child in env.website_envs:
        child._sandbox = env._sandbox
        child._gateway_url = env._gateway_url
        child._env_provider = provider


def _gateway_state_provider(env: MultiEnv) -> DatabaseStateProvider | None:
    """The state provider behind our gateway, which a snapshot restores into or captures from; None for a deployment without it."""
    from agent_env.providers import EnvironmentGatewayProvider
    return env._env_provider._state_provider if isinstance(env._env_provider, EnvironmentGatewayProvider) else None


def _hand_record_to_child_envs(env: MultiEnv, deployed: DeployedEnv) -> None:
    """Give each child env the record its loads read the card from; a name an MCP server and a website share keeps the live probe, since the card can't tell them apart by name."""
    shared = {c.environment_name for c in env.mcp_server_envs} & {c.environment_name for c in env.website_envs}
    for child in [*env.mcp_server_envs, *env.website_envs]:
        child._deployed = None if child.environment_name in shared else deployed
