import { useEffect, useState, useCallback, useRef, useMemo } from 'react';
import { Search, X, RefreshCw } from 'lucide-react';
import {
  BACKEND_URL,
  apiFetch,
  PaginatedResponse,
  formatCellValue,
  TAG_LIMIT,
  TagList,
} from './shared';
import {
  ALL_UNIVERSE_QUERY_TYPES,
  normalizeUniverseType,
  universeTypeQueryValues,
  type UniverseType,
} from '../lib/universe-types';

export const UNIVERSE_PAGE_SIZE = 10;

export type { UniverseType };
type TypeFilter = 'all' | UniverseType;

export function UniversesPage({
  onSelectUniverse,
  selectionMode,
  selectedId,
  onSelect,
  universeType,
}: {
  onSelectUniverse?: (id: string, version?: number) => void;
  selectionMode?: boolean;
  selectedId?: string | null;
  onSelect?: (id: string, version?: number, type?: UniverseType) => void;
  /** When set, locks the page to this universe type (no tabs / run-group filter). */
  universeType?: UniverseType;
}) {
  const lockedType = universeType;

  const [items, setItems] = useState<Record<string, unknown>[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [runGroupIdQuery, setRunGroupIdQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState<TypeFilter>(lockedType ?? 'all');
  const [errorToast, setErrorToast] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inFlightRef = useRef<AbortController | null>(null);
  // Backend returns latest matching docs by insert-time without version-dedup, so the rare multi-version id is
  // filtered here. Persists across "Show more"; reset on !append.
  const seenIdsRef = useRef<Set<string>>(new Set());

  const queriedTypes = useMemo<string[]>(() => {
    if (lockedType) return universeTypeQueryValues(lockedType);
    if (typeFilter === 'all') return ALL_UNIVERSE_QUERY_TYPES;
    return universeTypeQueryValues(typeFilter);
  }, [lockedType, typeFilter]);

  const includesFileUniverses = queriedTypes.includes('file_artifact_universe');
  // Unlike the sibling flags, keep tabs in selection mode so a pick can be narrowed by type.
  const showTypeTabs = !lockedType;
  const showRunGroupFilter = includesFileUniverses && !selectionMode;

  const fetchUniverses = useCallback(
    (
      idQuery: string,
      runGroupQuery: string,
      types: string[],
      offset = 0,
      append = false,
    ) => {
      if (append) setLoadingMore(true);
      else setLoading(true);
      const params = new URLSearchParams();
      for (const t of types) params.append('type', t);
      params.set(
        'limit',
        String(append ? UNIVERSE_PAGE_SIZE * 3 : UNIVERSE_PAGE_SIZE),
      );
      params.set('offset', String(offset));
      if (idQuery) params.set('id', idQuery);
      if (runGroupQuery && types.includes('file_artifact_universe')) {
        params.set('run_group_id', runGroupQuery);
      }

      // Cancel any earlier in-flight request so a slow service_universe response can't clobber a newer file_artifact_universe view.
      inFlightRef.current?.abort();
      const controller = new AbortController();
      inFlightRef.current = controller;

      apiFetch(`${BACKEND_URL}/api/v1/artifacts?${params}`, {
        signal: controller.signal,
      })
        .then(res => res.json())
        .then((json: PaginatedResponse) => {
          if (controller.signal.aborted) return;
          if (!append) seenIdsRef.current = new Set();
          const deduped = json.items.filter(item => {
            const id = String(item.id);
            if (seenIdsRef.current.has(id)) return false;
            seenIdsRef.current.add(id);
            return true;
          });
          if (append) {
            setItems(prev => [...prev, ...deduped]);
          } else {
            setItems(deduped);
          }
          setTotal(json.total);
          setLoading(false);
          setLoadingMore(false);
        })
        .catch(err => {
          if (err?.name === 'AbortError') return;
          setLoading(false);
          setLoadingMore(false);
        });
    },
    [],
  );

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchUniverses(searchQuery.trim(), runGroupIdQuery.trim(), queriedTypes);
    }, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [searchQuery, runGroupIdQuery, queriedTypes, fetchUniverses]);

  const handleShowMore = useCallback(() => {
    fetchUniverses(
      searchQuery.trim(),
      runGroupIdQuery.trim(),
      queriedTypes,
      items.length,
      true,
    );
  }, [
    searchQuery,
    runGroupIdQuery,
    queriedTypes,
    items.length,
    fetchUniverses,
  ]);

  const handleRefresh = useCallback(() => {
    fetchUniverses(searchQuery.trim(), runGroupIdQuery.trim(), queriedTypes);
  }, [searchQuery, runGroupIdQuery, queriedTypes, fetchUniverses]);

  const hasMore = items.length < total;

  // Float an exact id match above same-substring universes (backend id filter is substring, newest-first).
  const displayItems = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return items;
    return [...items].sort(
      (a, b) =>
        Number(String(b.id).toLowerCase() === q) -
        Number(String(a.id).toLowerCase() === q),
    );
  }, [items, searchQuery]);

  const heading =
    lockedType === 'file_artifact_universe'
      ? 'File Artifact Universes'
      : 'Universes Hub';
  const subtitle =
    lockedType === 'file_artifact_universe'
      ? 'Artifact bundles produced by collect_artifacts'
      : lockedType === 'service_universe'
      ? 'Explore the worlds underlying RL envs'
      : 'Service universes and file-artifact universes in one place';
  const emptyMessage =
    items.length === 0 && (searchQuery || runGroupIdQuery)
      ? 'No universes match your filters'
      : 'No universes found';
  const searchPlaceholder = 'Search universes by id...';

  return (
    <div className={selectionMode ? 'p-4' : 'p-8'}>
      {!selectionMode && (
        <>
          <h1 className="text-2xl font-semibold mb-1">{heading}</h1>
          <p className="text-sm text-[var(--muted-foreground)] mb-4">
            {subtitle}
          </p>
        </>
      )}

      {showTypeTabs && (
        <div className="inline-flex rounded-md border border-[var(--border)] overflow-hidden text-sm mb-4">
          {(
            [
              ['all', 'All'],
              ['service_universe', 'Services'],
              ['file_artifact_universe', 'Files'],
              ['coding_task_harbor', 'Coding'],
            ] as const
          ).map(([k, label]) => (
            <button
              key={k}
              onClick={() => setTypeFilter(k)}
              className={`px-3 py-1.5 border-r border-[var(--border)] last:border-r-0 transition-colors ${
                typeFilter === k
                  ? 'bg-[var(--secondary)] text-[var(--foreground)] font-medium'
                  : 'text-[var(--muted-foreground)] hover:bg-[var(--accent)] hover:text-[var(--foreground)]'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3 mb-6">
        <div className="relative max-w-lg flex-1 min-w-[240px]">
          <Search
            size={14}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted-foreground)]"
          />
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder={searchPlaceholder}
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

        {showRunGroupFilter && (
          <div className="relative max-w-sm flex-1 min-w-[200px]">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted-foreground)]"
            />
            <input
              type="text"
              value={runGroupIdQuery}
              onChange={e => setRunGroupIdQuery(e.target.value)}
              placeholder="Filter by run group id..."
              className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] pl-9 pr-8 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-[var(--ring)]"
            />
            {runGroupIdQuery && (
              <button
                onClick={() => setRunGroupIdQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
              >
                <X size={14} />
              </button>
            )}
          </div>
        )}

        {!selectionMode && (
          <button
            onClick={handleRefresh}
            disabled={loading}
            title="Refresh"
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-md border border-[var(--border)] text-sm text-[var(--muted-foreground)] hover:bg-[var(--accent)] hover:text-[var(--foreground)] transition-colors disabled:opacity-50"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
        )}
      </div>

      {errorToast && (
        <div className="mb-4 rounded-md border border-red-300 bg-red-50 text-red-800 text-sm px-3 py-2">
          {errorToast}
        </div>
      )}

      {loading ? (
        <p className="text-sm text-[var(--muted-foreground)]">Loading...</p>
      ) : items.length === 0 ? (
        <p className="text-sm text-[var(--muted-foreground)]">{emptyMessage}</p>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
            {displayItems.map((item, i) => {
              const itemType: UniverseType = normalizeUniverseType(item.type);
              const id = String(item.id);
              const v = item.version != null ? Number(item.version) : undefined;
              return (
                <UniverseCard
                  key={`${id}-${i}`}
                  item={item}
                  universeType={itemType}
                  selected={selectionMode && selectedId === id}
                  onClick={() => {
                    // Pin a version only for a run-group-scoped view (each card = a specific version); otherwise undefined → detail defaults to latest.
                    const pinned = runGroupIdQuery.trim() ? v : undefined;
                    if (selectionMode) {
                      onSelect?.(id, v, itemType);
                    } else {
                      onSelectUniverse?.(id, pinned);
                    }
                  }}
                />
              );
            })}
          </div>
          {hasMore && (
            <button
              onClick={handleShowMore}
              disabled={loadingMore}
              className="mt-3 px-4 py-1.5 rounded-md border border-[var(--border)] text-sm font-medium text-[var(--muted-foreground)] hover:bg-[var(--accent)] hover:text-[var(--foreground)] transition-colors disabled:opacity-50"
            >
              {loadingMore ? 'Loading...' : 'Show more'}
            </button>
          )}
        </>
      )}
    </div>
  );
}

function CodingTaskHarborSummary({ item }: { item: Record<string, unknown> }) {
  const mirror = (item.mirror_metadata ?? {}) as Record<string, unknown>;
  const owner = String(mirror.owner ?? item.owner ?? '');
  const repo = String(mirror.repo ?? item.repo ?? '');
  const baseCommit = String(mirror.base_commit ?? item.base_commit ?? '');
  const bundle = (item.bundle ?? {}) as Record<string, unknown>;
  const piecesPresent = (
    [
      ['Dockerfile', 'docker_file'],
      ['Image', 'docker_image'],
      ['Golden patch', 'golden_patch'],
      ['Test patch', 'test_patch'],
      ['Run script', 'run_script'],
      ['Interface', 'interface_md'],
    ] as const
  )
    .filter(([, key]) => !!bundle[key])
    .map(([label]) => label);
  return (
    <>
      {(owner || repo) && (
        <div className="text-xs text-[var(--foreground)] font-mono truncate">
          {owner && repo ? `${owner}/${repo}` : owner || repo}
        </div>
      )}
      {baseCommit && (
        <div className="text-xs text-[var(--muted-foreground)] font-mono">
          {baseCommit.slice(0, 8)}
        </div>
      )}
      {piecesPresent.length > 0 && (
        <div className="mt-1">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-1">
            Bundle ({piecesPresent.length})
          </div>
          <div className="flex flex-wrap gap-1">
            <TagList items={piecesPresent} limit={TAG_LIMIT} suffix="piece" />
          </div>
        </div>
      )}
    </>
  );
}

export function UniverseCard({
  item,
  onClick,
  selected,
  universeType = 'service_universe',
}: {
  item: Record<string, unknown>;
  onClick: () => void;
  selected?: boolean;
  universeType?: UniverseType;
}) {
  const serviceArtifactRefs = (item.service_artifact_refs ?? []) as {
    id: string;
    version: number;
  }[];
  const serviceArtifactIds =
    serviceArtifactRefs.length > 0
      ? serviceArtifactRefs.map(r => r.id)
      : ((item.service_artifact_ids ?? []) as string[]);
  const fileArtifactIds = (item.file_artifact_ids ?? {}) as Record<
    string,
    string
  >;
  const filenames = Object.keys(fileArtifactIds);

  return (
    <div
      onClick={onClick}
      className={`flex flex-col h-full p-4 rounded-lg border transition-colors cursor-pointer ${
        selected
          ? 'border-violet-600 bg-violet-50'
          : 'border-[var(--border)] hover:bg-[var(--accent)]'
      }`}
    >
      <div className="flex flex-col gap-2">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-sm font-semibold text-[var(--foreground)] truncate">
            {String(item.id)}
          </span>
          <span className="text-xs text-[var(--muted-foreground)] flex-shrink-0">
            v{String(item.version)}
          </span>
        </div>
        <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
          {universeType === 'file_artifact_universe'
            ? 'Files'
            : universeType === 'coding_task_harbor'
            ? 'Coding'
            : 'Services'}
        </div>
        {!!item.created_at_utc && (
          <div className="text-xs text-[var(--muted-foreground)]">
            <div className="font-medium">Last Modified</div>
            <div>{formatCellValue('created_at_utc', item.created_at_utc)}</div>
          </div>
        )}
        {universeType === 'service_universe' &&
          serviceArtifactIds.length > 0 && (
            <div className="mt-1">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-1">
                Services ({serviceArtifactIds.length})
              </div>
              <div className="flex flex-wrap gap-1">
                <TagList
                  items={serviceArtifactIds}
                  limit={TAG_LIMIT}
                  suffix="service"
                />
              </div>
            </div>
          )}
        {universeType === 'file_artifact_universe' && filenames.length > 0 && (
          <div className="mt-1">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-1">
              Files ({filenames.length})
            </div>
            <div className="flex flex-wrap gap-1">
              <TagList items={filenames} limit={TAG_LIMIT} suffix="file" />
            </div>
          </div>
        )}
        {universeType === 'coding_task_harbor' && (
          <CodingTaskHarborSummary item={item} />
        )}
      </div>
    </div>
  );
}
