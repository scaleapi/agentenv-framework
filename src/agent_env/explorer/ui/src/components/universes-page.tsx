import { useEffect, useState, useCallback, useRef, useMemo } from 'react';
import { Search, X, RefreshCw } from 'lucide-react';
import {
  useExternalApp,
  type SubmissionItem,
} from '../lib/external-app';
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

/** Iframe-host config via INIT_STATE.payload.inputs.value. All fields optional — omitting it still gives a
 *  working hub. Currently a single pre-filter knob; more picker policies can be added as optional fields. */
interface UniversesConfig {
  type_filter?: UniverseType;
}

/** Shape posted back to the parent as a SUBMISSION `data` item. Same for picked and newly-created universes; `source` distinguishes for host analytics. */
interface UniversePickSubmission {
  universe_id: string;
  version: number;
  type: UniverseType;
  source: 'picked' | 'created';
}

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

  // Iframe-host integration, auto-detected via window.parent !== window. When embedded: card click selects
  // (two-step), a sticky bar emits a SUBMISSION, and Create auto-submits instead of navigating. Standalone unchanged.
  const externalApp = useExternalApp();
  // window.parent !== window is stable for the component's life, but recomputing inline risks a hydration
  // render flipping the value (SSR → client) and retriggering the INIT_STATE effect. Memo'd once.
  const isEmbedded = useMemo(
    () => typeof window !== 'undefined' && window.parent !== window,
    [],
  );
  const [pickedUniverse, setPickedUniverse] =
    useState<UniversePickSubmission | null>(null);
  const lastEmbedInputsRef = useRef<unknown>(undefined);

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

  // Consume INIT_STATE from the host: unwrap `.value` to reach UniversesConfig. React only to new inputs (tracked by ref).
  useEffect(() => {
    if (!isEmbedded || !externalApp.isReady) return;
    if (lastEmbedInputsRef.current === externalApp.receivedInputs) return;
    lastEmbedInputsRef.current = externalApp.receivedInputs;
    const inputs =
      (externalApp.receivedInputs as { value?: UniversesConfig } | null)
        ?.value ?? null;
    if (!inputs) return;
    if (inputs.type_filter && !lockedType) {
      setTypeFilter(inputs.type_filter);
    }
  }, [isEmbedded, externalApp.isReady, externalApp.receivedInputs, lockedType]);

  // Post a SUBMISSION back to the host. Shape matches the task-runner pattern so both pages share the host message handler.
  const submitToHost = useCallback(
    (data: UniversePickSubmission) => {
      const item: SubmissionItem = {
        content: {
          id: 'universe-pick',
          type: 'json',
          data: data as unknown as Record<string, unknown>,
        },
        metadata: { universe_id: data.universe_id, version: data.version },
      };
      externalApp.sendSubmission([item]);
    },
    [externalApp],
  );

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
              // Embed pick mode: a card is "selected" when it matches the picked universe. Selection lives in local state.
              const isPickedHere =
                isEmbedded &&
                pickedUniverse?.universe_id === id &&
                (v == null || pickedUniverse?.version === v);
              return (
                <UniverseCard
                  key={`${id}-${i}`}
                  item={item}
                  universeType={itemType}
                  selected={
                    isPickedHere || (selectionMode && selectedId === id)
                  }
                  onClick={() => {
                    // Pin a version only for a run-group-scoped view (each card = a specific version); otherwise undefined → detail defaults to latest.
                    const pinned = runGroupIdQuery.trim() ? v : undefined;
                    if (isEmbedded) {
                      // Two-step picker: card click toggles selection; the "Use this universe" bar emits the submission.
                      setPickedUniverse(prev =>
                        prev &&
                        prev.universe_id === id &&
                        prev.version === (v ?? 1)
                          ? null
                          : {
                              universe_id: id,
                              version: v ?? 1,
                              type: itemType,
                              source: 'picked',
                            },
                      );
                    } else if (selectionMode) {
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


      {/* Sticky two-step picker action bar — only when embedded and the
          user has a card selected. Floats over the bottom edge of the
          iframe viewport. */}
      {isEmbedded && pickedUniverse && (
        <div className="sticky bottom-0 left-0 right-0 mt-6 -mx-4 -mb-4 sm:-mx-6 sm:-mb-6 border-t border-[var(--border)] bg-[var(--background)]/95 backdrop-blur px-4 sm:px-6 py-3 flex items-center gap-3 flex-wrap shadow-[0_-2px_8px_rgba(0,0,0,0.04)]">
          <div className="flex flex-col min-w-0">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
              Selected
            </span>
            <div className="flex items-baseline gap-2 min-w-0">
              <code className="font-mono text-sm font-semibold truncate">
                {pickedUniverse.universe_id}
              </code>
              <span className="text-xs text-[var(--muted-foreground)]">
                v{pickedUniverse.version}
              </span>
              <span className="text-[10px] font-mono uppercase text-[var(--muted-foreground)]">
                {pickedUniverse.type}
              </span>
            </div>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={() => setPickedUniverse(null)}
              className="px-3 py-1.5 rounded-md border border-[var(--border)] text-sm text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => submitToHost(pickedUniverse)}
              className="px-4 py-1.5 rounded-md bg-violet-600 text-white text-sm font-semibold hover:bg-violet-700 transition-colors"
            >
              Use this universe
            </button>
          </div>
        </div>
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
