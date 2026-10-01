import { Home as HomeIcon } from 'lucide-react';
import Link from 'next/link';

/** Base URL for hub API calls. Empty by default → same-origin relative paths that next.config.mjs rewrites to
 * the hub. The `?? ''` is load-bearing: ~144 call sites interpolate it unguarded, so `undefined` would produce
 * `/undefined/api/v1/...` — and the catch-all answers 200 text/html, so it fails quietly. */
export const BACKEND_URL = process.env.NEXT_PUBLIC_AGENT_ENV_HUB_BACKEND_URL ?? '';

export const FRAMEWORK_DOCS_URL = 'https://www.agentenvframework.com/docs';

export function apiFetch(input: string, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers);
  headers.set('x-agent-env-client', 'agent-env-explorer');
  return fetch(input, { ...init, headers });
}

// Same-origin URL proxying an object-store blob through the /objects/content seam — works for any store and, being same-origin, dodges the store's CORS.
export function objectContentUrl(objectUrl: string): string {
  return `${BACKEND_URL}/api/v1/objects/content?object_url=${encodeURIComponent(objectUrl)}`;
}

export const SECTION_HEADER_CLASS =
  'text-sm font-semibold uppercase tracking-wider text-[var(--muted-foreground)]';

export function formatCellValue(col: string, value: unknown): string {
  if (col === 'created_at_utc' && typeof value === 'string') {
    return new Date(value).toLocaleString('en-US', {
      timeZone: 'America/Los_Angeles',
      timeZoneName: 'short',
    });
  }
  return String(value ?? '');
}

export interface PaginatedResponse {
  items: Record<string, unknown>[];
  total: number;
  limit: number;
  offset: number;
}

export type Page =
  | 'home'
  | 'environments'
  | 'env-detail'
  | 'universes'
  | 'universe-detail'
  | 'tasks'
  | 'task-detail'
  | 'agents'
  | 'agent-detail'
  | 'task-runner'
  | 'docs';

export const PAGE_PATH: Partial<Record<Page, string>> = {
  home: '/',
  environments: '/environments',
  'env-detail': '/environments',
  universes: '/universes',
  'universe-detail': '/universes',
  tasks: '/tasks',
  'task-detail': '/tasks',
  agents: '/agents',
  'agent-detail': '/agents',
  'task-runner': '/task-runner',
  docs: '/docs',
};

export function pageToPath(page: Page, entityId?: string | null): string {
  const base = PAGE_PATH[page] ?? '/';
  return entityId ? `${base}/${encodeURIComponent(entityId)}` : base;
}

/** A real anchor for in-place SPA navigation. Rendering as <button> loses link affordances (hover URL,
 * cmd/middle-click new tab, right-click open). Emit an href and bail on modified clicks; keep shallow nav for plain clicks. */
export function EntityLink({
  page,
  entityId,
  onNavigate,
  className,
  title,
  children,
}: {
  page: Page;
  entityId: string;
  onNavigate: () => void;
  className?: string;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <Link
      href={pageToPath(page, entityId)}
      className={className}
      title={title}
      onClick={e => {
        // Modified clicks belong to the browser (new tab / new window).
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0)
          return;
        e.preventDefault();
        onNavigate();
      }}
    >
      {children}
    </Link>
  );
}

export interface NavGroup {
  label: string | null;
  items: { page: Page; label: string; icon: typeof HomeIcon }[];
}

export interface HomeCardGroup {
  label: string;
  cards: {
    page: Page;
    icon: typeof HomeIcon;
    title: string;
    description: string;
  }[];
}

export const MODEL_OPTIONS = [
  'claude-sonnet-4-6',
  'claude-opus-4-7',
  'claude-opus-4-6',
  'claude-opus-5',
  'claude-sonnet-5',
  'claude-sonnet-4-5',
  'claude-haiku-4-5',
  'gpt-4o',
  'gpt-4o-mini',
  'o3',
  'o3-mini',
  'o4-mini',
  'gemini/gemini-3-flash',
  'gemini/gemini-3.6-flash',
  'gemini/gemini-pro-latest',
  'gemini/gemini-2.5-pro',
  'gemini/gemini-2.5-flash',
  'gemini/gemini-2.0-flash',
  'fireworks_ai/kimi-k2p7-code',
];

export const TAG_LIMIT = 3;

export function TagList({
  items,
  limit,
  suffix,
}: {
  items: string[];
  limit: number;
  suffix: string;
}) {
  if (items.length === 0) return null;
  const visible = items.slice(0, limit);
  const overflow = items.length - limit;

  return (
    <>
      {visible.map(id => (
        <span
          key={id}
          title={id}
          className="inline-block max-w-full truncate align-bottom px-2 py-0.5 rounded text-[11px] font-medium bg-[var(--secondary)] text-[var(--foreground)]"
        >
          {id}
        </span>
      ))}
      {overflow > 0 && (
        <span className="inline-block px-2 py-0.5 rounded text-[11px] text-[var(--muted-foreground)]">
          +{overflow} more {suffix}
          {overflow !== 1 ? 's' : ''}
        </span>
      )}
    </>
  );
}

export function MetadataTable({
  metadata,
}: {
  metadata: Record<string, unknown>;
}) {
  if (Object.keys(metadata).length === 0) return null;
  return (
    <div className="mb-6">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
        Metadata
      </h3>
      <div className="rounded-lg border border-[var(--border)] overflow-hidden w-fit">
        <table className="text-sm">
          <tbody>
            {Object.entries(metadata).map(([key, value]) => (
              <tr
                key={key}
                className="border-b border-[var(--border)] last:border-b-0"
              >
                <td className="px-3 py-1.5 text-[var(--muted-foreground)] font-mono whitespace-nowrap">
                  {key}
                </td>
                <td className="px-3 py-1.5 text-[var(--foreground)] font-mono break-all">
                  {typeof value === 'object' && value !== null ? (
                    <details className="text-xs">
                      <summary className="cursor-pointer text-[var(--muted-foreground)] hover:text-[var(--foreground)] select-none">
                        {Array.isArray(value)
                          ? `Array (${value.length})`
                          : `Object (${
                              Object.keys(value as Record<string, unknown>)
                                .length
                            } keys)`}
                      </summary>
                      <pre className="mt-1.5 p-2 rounded-md bg-[var(--secondary)] overflow-auto max-h-80 max-w-xl whitespace-pre text-[var(--foreground)]">
                        {JSON.stringify(value, null, 2)}
                      </pre>
                    </details>
                  ) : typeof value === 'string' &&
                    /^\d{4}-\d{2}-\d{2}T/.test(value) ? (
                    new Date(value).toLocaleString('en-US', {
                      timeZone: 'America/Los_Angeles',
                      month: 'numeric',
                      day: 'numeric',
                      year: 'numeric',
                      hour: 'numeric',
                      minute: '2-digit',
                      second: '2-digit',
                      hour12: true,
                      timeZoneName: 'short',
                    })
                  ) : (
                    String(value)
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export const EXPECTED_VALIDATION_KEYS = [
  'mcp_tool_schema_validation',
  'mcp_tool_correctness_validation',
];

export type ValidationStatus = 'passed' | 'issues' | 'stale' | 'missing';

export function getValidationStatus(
  metadata: Record<string, unknown>,
): ValidationStatus {
  const presentKeys = EXPECTED_VALIDATION_KEYS.filter(k => k in metadata);
  if (presentKeys.length === 0) return 'missing';
  if (presentKeys.length < EXPECTED_VALIDATION_KEYS.length) return 'stale';
  const hasFailed = presentKeys.some(k => {
    const v = metadata[k] as Record<string, unknown> | null;
    return v && v.passed === false;
  });
  return hasFailed ? 'issues' : 'passed';
}
