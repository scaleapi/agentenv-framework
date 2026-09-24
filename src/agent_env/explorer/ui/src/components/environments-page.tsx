import { useEffect, useState, useCallback, useRef, useMemo } from 'react';
import { AlertTriangle, Search, Sparkles, X } from 'lucide-react';
import {
  BACKEND_URL,
  apiFetch,
  PaginatedResponse,
  formatCellValue,
  TAG_LIMIT,
  TagList,
} from './shared';

// One section per env type core registers. Each drives its own `?type=` fetch.
//
// This listed only `multi` and `mcp_server`, so the two envs `agent-env up`
// bootstraps — `gateway_server` and `service_db` — were invisible: a first run was a
// slow bootstrap followed by two empty sections. Keep this in step with the `type`
// ClassVar on the Env subclasses in src/agent_env/env/envs/.
//
// Types registered by a plugin rather than core (through `[envs] impls`) still
// will not appear. A catch-all section is the real fix; it needs an API that can
// list the distinct types present, which /envs does not expose yet.
export const ENV_SECTIONS = [
  { type: 'multi', label: 'Multi-Service Environments' },
  { type: 'mcp_server', label: 'MCP Server Environments' },
  { type: 'website', label: 'Website Environments' },
  { type: 'service_db', label: 'Service Database Environments' },
  { type: 'gateway_server', label: 'Gateway Environments' },
];

export const ENV_PAGE_SIZE = 10;

// Curated flagship environments surfaced in a "Featured Environments" grid at
// the top of the hub, mirroring FEATURED_AGENTS in agents-hub-page.tsx. Each
// entry carries its env `type` (needed by EnvCard) and an optional brand logo.
//
// Empty by default: curation is deployment-specific. Featured envs are fetched by id via GET /api/v1/envs/{id}; an id that doesn't
// resolve (wrong stage / deleted / renamed) renders a dimmed "Unavailable"
// placeholder tile rather than silently vanishing, so mis-curation stays visible.
// Editing this list is a safe, frontend-only change requiring no backend work.
export const FEATURED_ENVIRONMENTS: Array<{
  id: string;
  type: string;
  logo?: string;
  icon?: string;
  title?: string;
  tagline?: string;
  description?: string;
  recommendedUniverse?: string;
}> = [];

const FEATURED_ENVIRONMENT_IDS = FEATURED_ENVIRONMENTS.map(f => f.id);
const FEATURED_TYPE_BY_ID: Record<string, string> = Object.fromEntries(
  FEATURED_ENVIRONMENTS.map(f => [f.id, f.type]),
);
export const FEATURED_LOGO_BY_ID: Record<string, string | undefined> =
  Object.fromEntries(FEATURED_ENVIRONMENTS.map(f => [f.id, f.logo]));
const FEATURED_ICON_BY_ID: Record<string, string | undefined> =
  Object.fromEntries(FEATURED_ENVIRONMENTS.map(f => [f.id, f.icon]));
// Full featured entry by id — consumed by the env detail header for the curated
// title/tagline/description (mirrors how agent-detail-page reuses FEATURED_*).
export const FEATURED_ENV_BY_ID: Record<
  string,
  (typeof FEATURED_ENVIRONMENTS)[number]
> = Object.fromEntries(FEATURED_ENVIRONMENTS.map(f => [f.id, f]));

export function EnvironmentsPage({
  onSelectEnv,
  selectionMode,
  selectedId,
  onSelect,
}: {
  onSelectEnv?: (envId: string) => void;
  selectionMode?: boolean;
  selectedId?: string | null;
  onSelect?: (id: string, type: string) => void;
}) {
  const [sections, setSections] = useState<
    {
      type: string;
      label: string;
      items: Record<string, unknown>[];
      total: number;
      loading: boolean;
      // A failed fetch must not read as "No environments" — the two are
      // indistinguishable without this, which is how a broken explorer looks empty.
      error: string | null;
      loadingMore: boolean;
    }[]
  >(
    ENV_SECTIONS.map(s => ({
      ...s,
      items: [],
      total: 0,
      loading: true,
      error: null,
      loadingMore: false,
    })),
  );
  const [searchQuery, setSearchQuery] = useState('');
  const [featured, setFeatured] = useState<Record<string, unknown>[]>([]);
  const [featuredLoading, setFeaturedLoading] = useState(true);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Monotonic request id: a search's response is applied only if no newer
  // request has started since, so a slow `id=foo` can't clobber a newer `id=foobar`.
  const reqGenRef = useRef(0);
  const initialLoadDone = useRef(false);

  useEffect(() => {
    let cancelled = false;
    // With no curated ids there is nothing to wait for. Without this the section paints
    // its "Featured Environments" header and "Loading..." on every visit, then vanishes
    // when Promise.all([]) resolves — a flash of a section that will never have content.
    if (FEATURED_ENVIRONMENT_IDS.length === 0) {
      setFeaturedLoading(false);
      return;
    }
    setFeaturedLoading(true);
    Promise.all(
      FEATURED_ENVIRONMENT_IDS.map(id =>
        apiFetch(`${BACKEND_URL}/api/v1/envs/${encodeURIComponent(id)}`)
          .then(res => (res.ok ? res.json() : null))
          .catch(() => null),
      ),
    )
      .then(results => {
        if (cancelled) return;
        setFeatured(
          results.filter(
            (e): e is Record<string, unknown> => e !== null && e !== undefined,
          ),
        );
        setFeaturedLoading(false);
      })
      .catch(() => {
        // Promise.all resolves even on per-fetch failure (each has .catch),
        // but guard the outer chain so featuredLoading can never stick on true.
        if (!cancelled) setFeaturedLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const fetchSections = useCallback((query: string) => {
    if (!initialLoadDone.current) {
      setSections(prev => prev.map(s => ({ ...s, loading: true })));
    }
    const gen = ++reqGenRef.current;
    ENV_SECTIONS.forEach((section, i) => {
      const params = new URLSearchParams({
        type: section.type,
        limit: String(ENV_PAGE_SIZE),
      });
      if (query) params.set('id', query);
      apiFetch(`${BACKEND_URL}/api/v1/envs?${params}`)
        .then(res => {
          if (!res.ok) throw new Error(`Failed to fetch (${res.status})`);
          return res.json();
        })
        .then((json: PaginatedResponse) => {
          if (gen !== reqGenRef.current) return; // superseded by a newer search
          setSections(prev =>
            prev.map((s, j) =>
              j === i
                ? { ...s, items: json.items, total: json.total, loading: false, error: null }
                : s,
            ),
          );
          initialLoadDone.current = true;
        })
        .catch((e: unknown) => {
          if (gen !== reqGenRef.current) return;
          const message = e instanceof Error ? e.message : 'Request failed';
          setSections(prev =>
            prev.map((s, j) =>
              j === i ? { ...s, loading: false, error: message } : s,
            ),
          );
        });
    });
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchSections(searchQuery.trim());
    }, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [searchQuery, fetchSections]);

  const handleShowMore = useCallback(
    (sectionIndex: number) => {
      setSections(prev =>
        prev.map((s, j) =>
          j === sectionIndex ? { ...s, loadingMore: true } : s,
        ),
      );
      const section = sections[sectionIndex];
      if (!section) return;
      const offset = section.items.length;
      const params = new URLSearchParams({
        type: section.type,
        limit: String(ENV_PAGE_SIZE * 3),
        offset: String(offset),
      });
      if (searchQuery.trim()) params.set('id', searchQuery.trim());
      apiFetch(`${BACKEND_URL}/api/v1/envs?${params}`)
        .then(res => res.json())
        .then((json: PaginatedResponse) => {
          setSections(prev =>
            prev.map((s, j) =>
              j === sectionIndex
                ? {
                    ...s,
                    items: [...s.items, ...json.items],
                    loadingMore: false,
                  }
                : s,
            ),
          );
        })
        .catch(() => {
          setSections(prev =>
            prev.map((s, j) =>
              j === sectionIndex ? { ...s, loadingMore: false } : s,
            ),
          );
        });
    },
    [sections, searchQuery],
  );

  const q = searchQuery.trim().toLowerCase();
  // One tile per curated id, in curated order. `env` is the resolved doc, or
  // undefined when the id didn't resolve — those render an "Unavailable"
  // placeholder (see below) instead of being silently dropped.
  const featuredFiltered = useMemo(() => {
    const byId = new Map(featured.map(e => [String(e.id), e]));
    const tiles = FEATURED_ENVIRONMENT_IDS.map(id => ({
      id,
      env: byId.get(id),
    }));
    return q ? tiles.filter(t => t.id.toLowerCase().includes(q)) : tiles;
  }, [featured, q]);
  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold mb-1">Environments Hub</h1>
      <p className="text-sm text-[var(--muted-foreground)] mb-4">
        Discover & Curate RL Environments
      </p>
      <div className="relative mb-6 max-w-lg">
        <Search
          size={14}
          className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted-foreground)]"
        />
        <input
          type="text"
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          placeholder="Search environments..."
          className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] pl-9 pr-8 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-[var(--ring)]"
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery('')}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
          >
            <X size={14} />
          </button>
        )}
      </div>
      {(featuredLoading || featuredFiltered.length > 0) && (
        <section className="mb-8">
          <div className="flex items-center gap-2 mb-3">
            <Sparkles size={16} className="text-purple-500" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
              Featured Environments
            </h2>
          </div>
          {featuredLoading ? (
            <p className="text-sm text-[var(--muted-foreground)]">Loading...</p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
              {featuredFiltered.map(({ id, env }) =>
                env ? (
                  <EnvCard
                    key={`featured-${id}`}
                    item={env}
                    type={FEATURED_TYPE_BY_ID[id] ?? 'multi'}
                    logo={FEATURED_LOGO_BY_ID[id]}
                    icon={FEATURED_ICON_BY_ID[id]}
                    selected={selectionMode && selectedId === id}
                    onClick={() =>
                      selectionMode
                        ? onSelect?.(id, FEATURED_TYPE_BY_ID[id] ?? 'multi')
                        : onSelectEnv?.(id)
                    }
                  />
                ) : (
                  <FeaturedUnavailableCard
                    key={`featured-${id}`}
                    id={id}
                    logo={FEATURED_LOGO_BY_ID[id]}
                    icon={FEATURED_ICON_BY_ID[id]}
                  />
                ),
              )}
            </div>
          )}
        </section>
      )}
      {sections.map((section, sectionIndex) => {
        const hasMore = section.items.length < section.total;
        return (
          <div key={section.type} className="mb-8">
            <h2 className="text-lg font-semibold text-[var(--foreground)] mb-3">
              {section.label}
            </h2>
            {section.loading ? (
              <p className="text-sm text-[var(--muted-foreground)]">
                Loading...
              </p>
            ) : section.error ? (
              <p className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded p-2">
                Could not load environments ({section.error}).
              </p>
            ) : section.items.length === 0 ? (
              <p className="text-sm text-[var(--muted-foreground)]">
                No environments
              </p>
            ) : (
              <>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
                  {section.items.map((item, i) => (
                    <EnvCard
                      key={`${item.id}-${i}`}
                      item={item}
                      type={section.type}
                      selected={selectionMode && selectedId === String(item.id)}
                      onClick={() =>
                        selectionMode
                          ? onSelect?.(String(item.id), section.type)
                          : onSelectEnv?.(String(item.id))
                      }
                    />
                  ))}
                </div>
                {hasMore && (
                  <button
                    onClick={() => handleShowMore(sectionIndex)}
                    disabled={section.loadingMore}
                    className="mt-3 px-4 py-1.5 rounded-md border border-[var(--border)] text-sm font-medium text-[var(--muted-foreground)] hover:bg-[var(--accent)] hover:text-[var(--foreground)] transition-colors disabled:opacity-50"
                  >
                    {section.loadingMore ? 'Loading...' : 'Show more'}
                  </button>
                )}
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}

// Placeholder for a curated env whose id didn't resolve (wrong stage, deleted,
// renamed, or a transient fetch failure). Dimmed + dashed + non-interactive, it
// keeps the mis-curation visible instead of silently dropping the tile.
function FeaturedUnavailableCard({
  id,
  logo,
  icon,
}: {
  id: string;
  logo?: string;
  icon?: string;
}) {
  const banner = logo ?? icon;
  return (
    <div
      title={`"${id}" could not be loaded — it may not exist in this stage, or was deleted/renamed.`}
      className="flex flex-col h-full p-4 rounded-lg border border-dashed border-[var(--border)] opacity-60 cursor-default"
    >
      {banner && (
        <img
          src={banner}
          alt=""
          className={`-mx-4 -mt-4 mb-3 h-24 w-[calc(100%_+_2rem)] rounded-t-lg border-b border-[var(--border)] bg-white grayscale ${
            logo ? 'object-cover' : 'object-contain p-5'
          }`}
        />
      )}
      <div className="flex flex-col gap-2">
        <span className="font-mono text-sm font-semibold text-[var(--muted-foreground)] truncate">
          {id}
        </span>
        <div className="flex items-center gap-1.5 text-xs text-amber-600">
          <AlertTriangle size={12} className="flex-shrink-0" />
          <span>Unavailable</span>
        </div>
        <p className="text-xs text-[var(--muted-foreground)]">
          Not found in this stage.
        </p>
      </div>
    </div>
  );
}

export function EnvCard({
  item,
  type,
  onClick,
  selected,
  logo,
  icon,
}: {
  item: Record<string, unknown>;
  type: string;
  onClick: () => void;
  selected?: boolean;
  logo?: string;
  icon?: string;
}) {
  const mcpServers = (item.mcp_server_envs ?? []) as { id: string }[];
  const websites = (item.website_envs ?? []) as { id: string }[];
  // `logo` = full-bleed illustration; `icon` = centered brand/OS mark.
  const banner = logo ?? icon;

  return (
    <div
      onClick={onClick}
      className={`flex flex-col h-full p-4 rounded-lg border transition-colors cursor-pointer ${
        selected
          ? 'border-violet-600 bg-violet-50'
          : 'border-[var(--border)] hover:bg-[var(--accent)]'
      }`}
    >
      {banner && (
        <img
          src={banner}
          alt=""
          className={`-mx-4 -mt-4 mb-3 h-24 w-[calc(100%_+_2rem)] rounded-t-lg border-b border-[var(--border)] bg-white ${
            logo ? 'object-cover' : 'object-contain p-5'
          }`}
        />
      )}
      <div className="flex flex-col gap-2">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-sm font-semibold text-[var(--foreground)] truncate">
            {String(item.id)}
          </span>
          <span className="text-xs text-[var(--muted-foreground)] flex-shrink-0">
            v{String(item.version)}
          </span>
        </div>
        {(type === 'mcp_server' || type === 'website') &&
          !!item.service_name && (
            <div className="text-xs text-[var(--muted-foreground)]">
              Service:{' '}
              <span className="text-[var(--foreground)]">
                {String(item.service_name)}
              </span>
            </div>
          )}
        {!!item.created_at_utc && (
          <div className="text-xs text-[var(--muted-foreground)]">
            <div className="font-medium">Last Modified</div>
            <div>{formatCellValue('created_at_utc', item.created_at_utc)}</div>
          </div>
        )}
        {type === 'multi' && mcpServers.length > 0 && (
          <div className="mt-1">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-1">
              MCP Servers
            </div>
            <div className="flex flex-wrap gap-1">
              <TagList
                items={mcpServers.map(e => e.id)}
                limit={TAG_LIMIT}
                suffix="MCP server"
              />
            </div>
          </div>
        )}
        {type === 'multi' && websites.length > 0 && (
          <div className="mt-1">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-1">
              Websites
            </div>
            <div className="flex flex-wrap gap-1">
              <TagList
                items={websites.map(e => e.id)}
                limit={TAG_LIMIT}
                suffix="website"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
