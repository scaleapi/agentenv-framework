import { useEffect, useState, useCallback, useRef } from 'react';
import { Search, Sparkles, X } from 'lucide-react';
import {
  BACKEND_URL,
  PaginatedResponse,
  formatCellValue,
  apiFetch,
} from './shared';

const AGENTS_PAGE_SIZE = 20;
// Curated agents pinned to the top of Agents Hub. Each id is fetched via
// GET /api/v1/agents/{id}; one that doesn't resolve renders a dimmed
// "Unavailable" tile rather than vanishing. Add entries as agent artifacts register.
export const FEATURED_AGENTS: Array<{ id: string; logo?: string }> = [
  { id: 'claude-code-cli', logo: '/logos/claude.svg' },
];
const FEATURED_AGENT_IDS = FEATURED_AGENTS.map(f => f.id);
export const FEATURED_LOGO_BY_ID: Record<string, string | undefined> =
  Object.fromEntries(FEATURED_AGENTS.map(f => [f.id, f.logo]));

export function AgentsHubPage({
  onSelectAgent,
}: {
  onSelectAgent: (agentId: string) => void;
}) {
  const [agents, setAgents] = useState<Record<string, unknown>[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [featured, setFeatured] = useState<Record<string, unknown>[]>([]);
  const [featuredLoading, setFeaturedLoading] = useState(true);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Monotonic request id: a search's response is applied only if no newer
  // request has started since, so a slow `id=foo` can't clobber a newer `id=foobar`.
  const reqGenRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    setFeaturedLoading(true);
    Promise.all(
      FEATURED_AGENT_IDS.map(id =>
        apiFetch(`${BACKEND_URL}/api/v1/agents/${encodeURIComponent(id)}`)
          .then(res => (res.ok ? res.json() : null))
          .catch(() => null),
      ),
    ).then(results => {
      if (cancelled) return;
      setFeatured(
        results.filter(
          (a): a is Record<string, unknown> => a !== null && a !== undefined,
        ),
      );
      setFeaturedLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const fetchAgents = useCallback(
    (query: string, offset = 0, append = false) => {
      if (!append) {
        setLoading(true);
      } else {
        setLoadingMore(true);
      }

      const params = new URLSearchParams({
        limit: String(AGENTS_PAGE_SIZE),
        offset: String(offset),
        latest_only: 'true',
      });
      if (query) params.set('id', query);

      const gen = ++reqGenRef.current;
      apiFetch(`${BACKEND_URL}/api/v1/agents?${params}`)
        .then(res => {
          if (!res.ok) throw new Error(`Failed to fetch (${res.status})`);
          return res.json();
        })
        .then((json: PaginatedResponse) => {
          if (gen !== reqGenRef.current) return; // superseded by a newer search
          if (append) {
            setAgents(prev => [...prev, ...json.items]);
          } else {
            setAgents(json.items);
          }
          setTotal(json.total);
          setLoading(false);
          setLoadingMore(false);
        })
        .catch(() => {
          if (gen !== reqGenRef.current) return;
          setLoading(false);
          setLoadingMore(false);
        });
    },
    [],
  );

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchAgents(searchQuery.trim());
    }, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [searchQuery, fetchAgents]);

  const handleShowMore = useCallback(() => {
    fetchAgents(searchQuery.trim(), agents.length, true);
  }, [fetchAgents, searchQuery, agents.length]);

  const hasMore = agents.length < total;

  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold mb-1">Agents Hub</h1>
      <p className="text-sm text-[var(--muted-foreground)] mb-4">
        Browse & Deploy A2A Agents
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
          placeholder="Search agents..."
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
      {(() => {
        const q = searchQuery.trim().toLowerCase();
        const featuredSorted = FEATURED_AGENT_IDS.map(id =>
          featured.find(a => a.id === id),
        ).filter(
          (a): a is Record<string, unknown> => a !== null && a !== undefined,
        );
        const featuredFiltered = q
          ? featuredSorted.filter(a => String(a.id).toLowerCase().includes(q))
          : featuredSorted;
        const filteredAgents = agents;
        const hasAnyResults =
          featuredFiltered.length > 0 || filteredAgents.length > 0;

        return (
          <>
            {(featuredLoading || featuredFiltered.length > 0) && (
              <section className="mb-8">
                <div className="flex items-center gap-2 mb-3">
                  <Sparkles size={16} className="text-purple-500" />
                  <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
                    Featured Agents
                  </h2>
                </div>
                {featuredLoading ? (
                  <p className="text-sm text-[var(--muted-foreground)]">
                    Loading...
                  </p>
                ) : (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
                    {featuredFiltered.map((agent, i) => (
                      <AgentCard
                        key={`featured-${agent.id}-${i}`}
                        agent={agent}
                        logo={FEATURED_LOGO_BY_ID[String(agent.id)]}
                        onClick={() => onSelectAgent(String(agent.id))}
                      />
                    ))}
                  </div>
                )}
              </section>
            )}

            <section>
              <h2 className="text-sm font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-3">
                All Agents
              </h2>
              {loading ? (
                <p className="text-sm text-[var(--muted-foreground)]">
                  Loading...
                </p>
              ) : filteredAgents.length === 0 ? (
                <p className="text-sm text-[var(--muted-foreground)]">
                  {hasAnyResults ? 'No other agents' : 'No agents found'}
                </p>
              ) : (
                <>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
                    {filteredAgents.map((agent, i) => (
                      <AgentCard
                        key={`${agent.id}-${i}`}
                        agent={agent}
                        logo={FEATURED_LOGO_BY_ID[String(agent.id)]}
                        onClick={() => onSelectAgent(String(agent.id))}
                      />
                    ))}
                  </div>
                  {hasMore && (
                    <button
                      onClick={handleShowMore}
                      disabled={loadingMore}
                      className="mt-4 px-4 py-1.5 rounded-md border border-[var(--border)] text-sm font-medium text-[var(--muted-foreground)] hover:bg-[var(--accent)] hover:text-[var(--foreground)] transition-colors disabled:opacity-50"
                    >
                      {loadingMore ? 'Loading...' : 'Show more'}
                    </button>
                  )}
                </>
              )}
            </section>
          </>
        );
      })()}
    </div>
  );
}

export function AgentCard({
  agent,
  logo,
  onClick,
  selected = false,
}: {
  agent: Record<string, unknown>;
  logo?: string;
  onClick: () => void;
  selected?: boolean;
}) {
  const imageArtifact = agent.docker_image_artifact as
    | Record<string, unknown>
    | undefined;
  const metadata = agent.metadata as Record<string, unknown> | undefined;
  const metaEntries: Array<[string, string]> = metadata
    ? Object.entries(metadata).flatMap(([k, v]) =>
        typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
          ? [[k, String(v)] as [string, string]]
          : [],
      )
    : [];

  return (
    <div
      onClick={onClick}
      className={`flex flex-col h-full p-4 rounded-lg border transition-colors cursor-pointer ${
        selected
          ? 'border-violet-500 ring-2 ring-violet-300 bg-violet-50'
          : 'border-[var(--border)] hover:bg-[var(--accent)]'
      }`}
    >
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          {logo && (
            <img
              src={logo}
              alt=""
              className="w-8 h-8 rounded-md border border-[var(--border)] bg-white object-contain p-1 flex-shrink-0"
            />
          )}
          <div className="flex items-baseline gap-2 min-w-0">
            <span className="font-mono text-sm font-semibold text-[var(--foreground)] truncate">
              {String(agent.id)}
            </span>
            <span className="text-xs text-[var(--muted-foreground)] flex-shrink-0">
              v{String(agent.version)}
            </span>
          </div>
        </div>
        {imageArtifact && (
          <div className="text-xs text-[var(--muted-foreground)]">
            Image:{' '}
            <span className="text-[var(--foreground)]">
              {String(imageArtifact.id)}
            </span>
          </div>
        )}
        {!!agent.created_at_utc && (
          <div className="text-xs text-[var(--muted-foreground)]">
            <div className="font-medium">Last Modified</div>
            <div>{formatCellValue('created_at_utc', agent.created_at_utc)}</div>
          </div>
        )}
        {metaEntries.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1 overflow-hidden">
            {metaEntries.slice(0, 3).map(([key, value]) => (
              <span
                key={key}
                className="inline-block max-w-full px-2 py-0.5 rounded text-[11px] font-medium bg-[var(--secondary)] text-[var(--foreground)] truncate"
                title={`${key}: ${value}`}
              >
                {key}: {value}
              </span>
            ))}
            {metaEntries.length > 3 && (
              <span className="inline-block px-2 py-0.5 rounded text-[11px] text-[var(--muted-foreground)]">
                +{metaEntries.length - 3} more
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
