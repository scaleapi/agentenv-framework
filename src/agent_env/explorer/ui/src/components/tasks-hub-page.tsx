import { useEffect, useState, useCallback, useRef } from 'react';
import { Search, X } from 'lucide-react';
import {
  BACKEND_URL,
  apiFetch,
  EntityLink,
  PaginatedResponse,
  TAG_LIMIT,
  TagList,
  formatCellValue,
} from './shared';

export const TASK_PAGE_SIZE = 20;

export function TasksHubPage({
  onSelectTask,
}: {
  onSelectTask: (taskId: string) => void;
}) {
  const [items, setItems] = useState<Record<string, unknown>[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Monotonic request id: a search's response is applied only if no newer
  // request has started since, so a slow `id=foo` can't clobber a newer `id=foobar`.
  const reqGenRef = useRef(0);
  const initialLoadDone = useRef(false);
  // The backend fast path returns raw version-rows in insert-time desc order,
  // so an id's newest version comes first. Dedup latest-per-id here, persisting
  // across "Show more" so an older version can't slip in on a later page.
  // Reset on a non-append fetch. Mirrors universes-page.tsx.
  const seenIdsRef = useRef<Set<string>>(new Set());

  const fetchTasks = useCallback(
    (query: string, offset = 0, append = false) => {
      if (append) setLoadingMore(true);
      else if (!initialLoadDone.current) setLoading(true);
      const params = new URLSearchParams({
        latest_only: 'true',
        limit: String(append ? TASK_PAGE_SIZE * 3 : TASK_PAGE_SIZE),
        offset: String(offset),
      });
      if (query) params.set('id', query);
      const gen = ++reqGenRef.current;
      apiFetch(`${BACKEND_URL}/api/v1/tasks?${params}`)
        .then(res => {
          if (!res.ok) throw new Error(`Failed to fetch (${res.status})`);
          return res.json();
        })
        .then((json: PaginatedResponse) => {
          if (gen !== reqGenRef.current) return; // superseded by a newer search
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
          initialLoadDone.current = true;
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
      fetchTasks(searchQuery.trim());
    }, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [searchQuery, fetchTasks]);

  const handleShowMore = useCallback(() => {
    fetchTasks(searchQuery.trim(), items.length, true);
  }, [searchQuery, items.length, fetchTasks]);

  const hasMore = items.length < total;

  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold mb-1">Tasks Hub</h1>
      <p className="text-sm text-[var(--muted-foreground)] mb-4">
        Browse Agent Tasks & Step Sequences
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
          placeholder="Search tasks..."
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
      {loading ? (
        <p className="text-sm text-[var(--muted-foreground)]">Loading...</p>
      ) : items.length === 0 ? (
        <p className="text-sm text-[var(--muted-foreground)]">No tasks found</p>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4">
            {items.map((item, i) => (
              <TaskCard
                key={`${item.id}-${i}`}
                item={item}
                onNavigate={() => onSelectTask(String(item.id))}
              />
            ))}
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

function TaskCard({
  item,
  onNavigate,
}: {
  item: Record<string, unknown>;
  onNavigate: () => void;
}) {
  const steps = (item.steps ?? []) as {
    type: string;
    id: string;
    version: number;
  }[];
  const stepTypes = [...new Set(steps.map(s => s.type))];

  return (
    <EntityLink
      page="task-detail"
      entityId={String(item.id)}
      onNavigate={onNavigate}
      className="flex flex-col h-full p-4 rounded-lg border border-[var(--border)] hover:bg-[var(--accent)] transition-colors cursor-pointer"
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
        <div className="text-xs text-[var(--muted-foreground)]">
          {steps.length} step{steps.length !== 1 ? 's' : ''}
        </div>
        {!!item.created_at_utc && (
          <div className="text-xs text-[var(--muted-foreground)]">
            <div className="font-medium">Created</div>
            <div>{formatCellValue('created_at_utc', item.created_at_utc)}</div>
          </div>
        )}
        {stepTypes.length > 0 && (
          <div className="mt-1">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-1">
              Step Types
            </div>
            <div className="flex flex-wrap gap-1">
              <TagList items={stepTypes} limit={TAG_LIMIT} suffix="type" />
            </div>
          </div>
        )}
      </div>
    </EntityLink>
  );
}
