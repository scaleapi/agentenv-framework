import { useCallback, useEffect, useState } from 'react';
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle,
  ChevronDown,
  ChevronUp,
  FlaskConical,
  Globe,
  Loader2,
  RefreshCw,
  XCircle,
} from 'lucide-react';
import {
  BACKEND_URL,
  apiFetch,
  MetadataTable,
  formatCellValue,
  getValidationStatus,
} from './shared';
import { FEATURED_ENV_BY_ID } from './environments-page';
import {
  AdvertisedEnvironmentCard,
  type EnvironmentCard,
  type EnvironmentCardValidation,
} from './advertised-environment-card';

// environment_card / validated_environment_card are surfaced by the dedicated
// Environment Card section, so keep them out of the generic metadata table.
const HIDDEN_ENV_METADATA_KEYS = new Set([
  'environment_card',
  'validated_environment_card',
]);

// Generic, collapsible JSON dump of the env's resolved references — e.g. a CUA
// env's vm_image_artifact, including the ECR image tag it boots. Deliberately
// schema-agnostic: whatever the backend puts under `resolved_refs` is rendered
// verbatim, so new fields or refs need no frontend change.
function ResolvedRefsSection({ data }: { data: Record<string, unknown> }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="mb-6">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-1.5 mb-2 text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
      >
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        Resolved Details
      </button>
      {open && (
        <pre className="text-xs font-mono bg-[var(--secondary)] rounded-lg p-3 overflow-auto max-h-96 text-[var(--foreground)]">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

export function EnvDetailPage({
  envId,
  onBack,
  onNavigateToEnv,
  onNavigateToUniverse,
}: {
  envId: string;
  onBack: () => void;
  onNavigateToEnv: (id: string) => void;
  onNavigateToUniverse?: (universeId: string) => void;
}) {
  const [env, setEnv] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [childEnvData, setChildEnvData] = useState<
    Record<string, Record<string, unknown>>
  >({});

  const [childValidating, setChildValidating] = useState<
    Record<string, 'running' | 'done' | 'error'>
  >({});

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    apiFetch(`${BACKEND_URL}/api/v1/envs/${envId}`)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to fetch (${res.status})`);
        return res.json();
      })
      .then(data => {
        if (cancelled) return;
        setEnv(data);
        setLoading(false);

        // Fetch child MCP server env data for multi envs
        if (data?.type === 'multi') {
          const mcpIds = ((data.mcp_server_envs ?? []) as { id: string }[]).map(
            e => e.id,
          );
          Promise.all(
            mcpIds.map(async id => {
              try {
                const r = await apiFetch(`${BACKEND_URL}/api/v1/envs/${id}`);
                if (r.ok) return [id, await r.json()] as const;
              } catch {
                /* ignore */
              }
              return null;
            }),
          ).then(results => {
            if (cancelled) return;
            const map: Record<string, Record<string, unknown>> = {};
            for (const r of results) {
              if (r) map[r[0]] = r[1];
            }
            setChildEnvData(map);
          });
        }
      })
      .catch(e => {
        if (!cancelled) {
          setFetchError(e instanceof Error ? e.message : 'Failed to load');
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [envId]);

  const mcpServers = (env?.mcp_server_envs ?? []) as {
    id: string;
    service_name?: string;
  }[];
  const websites = (env?.website_envs ?? []) as {
    id: string;
    service_name?: string;
  }[];
  const envType = String(env?.type ?? '');
  // Curated copy for featured environments (title/tagline/description), shown on
  // the header. undefined for non-featured envs, which render as before.
  const featured = FEATURED_ENV_BY_ID[envId];
  // Featured banner mark shown on the header: a full-bleed illustration (`logo`,
  // MCP Advanced universes) or a centered brand/OS mark (`icon`, desktop CUA).
  const featuredBanner = featured?.logo ?? featured?.icon;
  const vmImageArtifact = env?.vm_image_artifact as
    | { id: string; version: number }
    | undefined;
  const cuaMcpServerEnv = env?.cua_mcp_server_env as
    | { id: string; service_name?: string }
    | undefined;
  // iOS CUA wraps a single MCP server env (the phone-driving MCP image). There
  // is no VM image — the env drives a physical device via a tunneled, Mac-hosted
  // computer server — so the detail view shows just the MCP server.
  const iosCuaMcpServerEnv = env?.ios_cua_mcp_server_env as
    | { id: string; service_name?: string }
    | undefined;
  const metadata = (env?.metadata ?? {}) as Record<string, unknown>;
  const environmentCard = (metadata.environment_card ??
    null) as EnvironmentCard | null;
  const validatedEnvCardRaw = (metadata.validated_environment_card ?? null) as {
    accessible?: boolean;
    required_fields?: Record<string, { present?: boolean }>;
    children_count?: number;
    extensions?: string[];
    error?: string;
  } | null;
  const envCardValidation: EnvironmentCardValidation | undefined =
    validatedEnvCardRaw
      ? {
          accessible: validatedEnvCardRaw.accessible,
          requiredFields: Object.fromEntries(
            Object.entries(validatedEnvCardRaw.required_fields ?? {}).map(
              ([name, info]) => [name, { present: !!info?.present }],
            ),
          ),
          childrenCount: validatedEnvCardRaw.children_count,
          extensions: validatedEnvCardRaw.extensions,
          error: validatedEnvCardRaw.error,
        }
      : undefined;
  const displayedMetadata = Object.fromEntries(
    Object.entries(metadata).filter(
      ([key]) => !HIDDEN_ENV_METADATA_KEYS.has(key),
    ),
  );
  const resolvedRefs = (env?.resolved_refs ?? null) as Record<
    string,
    unknown
  > | null;

  return (
    <div className="p-8 pb-16 flex flex-col h-full overflow-y-auto">
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 text-sm text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors mb-6"
      >
        <ArrowLeft size={14} />
        Back
      </button>

      {loading && (
        <p className="text-sm text-[var(--muted-foreground)]">Loading...</p>
      )}
      {fetchError && <p className="text-sm text-red-500">{fetchError}</p>}

      {env && !loading && (
        <>
          {/* Header */}
          <div className="mb-6">
            <div className="flex items-start gap-4">
              {featuredBanner && (
                <img
                  src={featuredBanner}
                  alt=""
                  className={`w-16 h-16 rounded-lg border border-[var(--border)] bg-white flex-shrink-0 ${
                    featured?.logo ? 'object-cover' : 'object-contain p-2.5'
                  }`}
                />
              )}
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-3 flex-wrap">
                  <h1 className="text-2xl font-semibold font-mono">{envId}</h1>
                  <span className="text-sm text-[var(--muted-foreground)]">
                    v{String(env.version)}
                  </span>
                  <span className="px-2 py-0.5 rounded text-xs font-medium bg-[var(--secondary)] text-[var(--foreground)]">
                    {envType}
                  </span>
                </div>
                {!!env.created_at_utc && (
                  <p className="text-sm text-[var(--muted-foreground)] mt-1">
                    Last Modified:{' '}
                    {formatCellValue('created_at_utc', env.created_at_utc)}
                  </p>
                )}
                {featured && (featured.title || featured.tagline) && (
                  <p className="text-sm font-medium text-[var(--foreground)] mt-2">
                    {[featured.title, featured.tagline]
                      .filter(Boolean)
                      .join(' · ')}
                  </p>
                )}
                {featured?.description && (
                  <p className="text-sm text-[var(--muted-foreground)] mt-1 leading-relaxed max-w-3xl">
                    {featured.description}
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Recommended Universe (curated for MCP Advanced envs) */}
          {featured?.recommendedUniverse && (
            <div className="mb-6">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
                Recommended Universe
              </h3>
              <button
                onClick={() =>
                  onNavigateToUniverse?.(featured.recommendedUniverse!)
                }
                disabled={!onNavigateToUniverse}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-[var(--border)] hover:bg-[var(--accent)] transition-colors disabled:cursor-default disabled:hover:bg-transparent"
              >
                <Globe size={14} className="text-[var(--muted-foreground)]" />
                <span className="font-mono text-sm text-[var(--foreground)]">
                  {featured.recommendedUniverse}
                </span>
              </button>
              <p className="text-xs text-[var(--muted-foreground)] mt-1.5">
                Deploy this environment with this universe for the intended MCP
                Advanced scenario.
              </p>
            </div>
          )}

          {/* Environment Card */}
          {environmentCard && (
            <div className="mb-6">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-3">
                Environment Card
              </h3>
              <AdvertisedEnvironmentCard
                card={environmentCard}
                validation={envCardValidation}
              />
            </div>
          )}
          {!environmentCard &&
            validatedEnvCardRaw &&
            validatedEnvCardRaw.accessible === false && (
              <div className="mb-6">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-3">
                  Environment Card
                </h3>
                <div className="rounded-lg border border-red-200 bg-red-50/40 px-4 py-3 flex items-start gap-2">
                  <AlertTriangle
                    size={14}
                    className="text-red-600 flex-shrink-0 mt-0.5"
                  />
                  <div className="min-w-0 text-sm">
                    <div className="font-medium text-red-700">
                      Environment card inaccessible
                    </div>
                    {validatedEnvCardRaw.error && (
                      <div className="text-xs text-red-700/90 mt-0.5 font-mono break-all">
                        {validatedEnvCardRaw.error}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

          {/* Env details */}
          {(envType === 'mcp_server' || envType === 'website') &&
            !!env.service_name && (
              <div className="mb-6">
                <div className="text-sm text-[var(--muted-foreground)]">
                  Service:{' '}
                  <span className="text-[var(--foreground)] font-medium">
                    {String(env.service_name)}
                  </span>
                </div>
              </div>
            )}

          {envType === 'cua' && (vmImageArtifact || cuaMcpServerEnv) && (
            <div className="mb-6 flex flex-col gap-4">
              {vmImageArtifact && (
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
                    VM Image
                  </h3>
                  <span className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm bg-[var(--secondary)]">
                    <span className="font-medium text-[var(--foreground)]">
                      {vmImageArtifact.id}
                    </span>
                    <span className="text-xs text-[var(--muted-foreground)] font-mono">
                      v{vmImageArtifact.version}
                    </span>
                  </span>
                </div>
              )}
              {cuaMcpServerEnv && (
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
                    CUA MCP Server
                  </h3>
                  <button
                    onClick={() => onNavigateToEnv(cuaMcpServerEnv.id)}
                    className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm bg-[var(--secondary)] hover:bg-[var(--accent)] transition-colors cursor-pointer"
                  >
                    <span className="font-medium text-[var(--foreground)]">
                      {cuaMcpServerEnv.service_name ?? cuaMcpServerEnv.id}
                    </span>
                    {cuaMcpServerEnv.service_name &&
                      cuaMcpServerEnv.service_name !== cuaMcpServerEnv.id && (
                        <span className="text-xs text-[var(--muted-foreground)] font-mono">
                          {cuaMcpServerEnv.id}
                        </span>
                      )}
                  </button>
                </div>
              )}
            </div>
          )}

          {envType === 'ios_cua' && (
            <div className="mb-6 flex flex-col gap-4">
              <p className="text-sm text-[var(--muted-foreground)]">
                Drives a physical iPhone via a tunneled, Mac-hosted computer
                server. No VM image — the device is leased at deploy time.
              </p>
              {iosCuaMcpServerEnv && (
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
                    iOS CUA MCP Server
                  </h3>
                  <button
                    onClick={() => onNavigateToEnv(iosCuaMcpServerEnv.id)}
                    className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm bg-[var(--secondary)] hover:bg-[var(--accent)] transition-colors cursor-pointer"
                  >
                    <span className="font-medium text-[var(--foreground)]">
                      {iosCuaMcpServerEnv.service_name ?? iosCuaMcpServerEnv.id}
                    </span>
                    {iosCuaMcpServerEnv.service_name &&
                      iosCuaMcpServerEnv.service_name !==
                        iosCuaMcpServerEnv.id && (
                        <span className="text-xs text-[var(--muted-foreground)] font-mono">
                          {iosCuaMcpServerEnv.id}
                        </span>
                      )}
                  </button>
                </div>
              )}
            </div>
          )}

          {envType === 'multi' &&
            (mcpServers.length > 0 || websites.length > 0) && (
              <div className="mb-6 flex flex-col gap-4">
                {mcpServers.length > 0 && (
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
                      MCP Servers ({mcpServers.length})
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {mcpServers.map(e => {
                        const child = childEnvData[e.id];
                        const childMeta = (child?.metadata ?? {}) as Record<
                          string,
                          unknown
                        >;
                        const vStatus =
                          childValidating[e.id] === 'running'
                            ? ('running' as const)
                            : getValidationStatus(childMeta);

                        const statusMap = {
                          running: {
                            icon: (
                              <Loader2
                                size={14}
                                className="text-[var(--muted-foreground)] animate-spin flex-shrink-0"
                              />
                            ),
                            text: 'Validating...',
                            color: 'text-[var(--muted-foreground)]',
                          },
                          missing: {
                            icon: (
                              <AlertTriangle
                                size={14}
                                className="text-amber-500 flex-shrink-0"
                              />
                            ),
                            text: 'Missing Validation',
                            color: 'text-amber-500',
                          },
                          stale: {
                            icon: (
                              <AlertTriangle
                                size={14}
                                className="text-amber-500 flex-shrink-0"
                              />
                            ),
                            text: 'Stale Validation',
                            color: 'text-amber-500',
                          },
                          issues: {
                            icon: (
                              <XCircle
                                size={14}
                                className="text-red-500 flex-shrink-0"
                              />
                            ),
                            text: 'Validation Issues Detected',
                            color: 'text-red-500',
                          },
                          passed: {
                            icon: (
                              <CheckCircle
                                size={14}
                                className="text-emerald-500 flex-shrink-0"
                              />
                            ),
                            text: 'Passed Validation',
                            color: 'text-emerald-600',
                          },
                        } as const;
                        const {
                          icon: statusIcon,
                          text: statusText,
                          color: statusColor,
                        } = statusMap[vStatus];

                        return (
                          <button
                            key={e.id}
                            onClick={() => onNavigateToEnv(e.id)}
                            className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm bg-[var(--secondary)] hover:bg-[var(--accent)] transition-colors cursor-pointer w-fit"
                          >
                            {statusIcon}
                            <span className="font-medium text-[var(--foreground)]">
                              {e.service_name ?? e.id}
                            </span>
                            {e.service_name && e.service_name !== e.id && (
                              <span className="text-xs text-[var(--muted-foreground)] font-mono">
                                {e.id}
                              </span>
                            )}
                            <span className={`text-xs ${statusColor}`}>
                              {statusText}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
                {websites.length > 0 && (
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
                      Websites ({websites.length})
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {websites.map(e => (
                        <button
                          key={e.id}
                          onClick={() => onNavigateToEnv(e.id)}
                          className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm bg-[var(--secondary)] hover:bg-[var(--accent)] transition-colors cursor-pointer"
                        >
                          <span className="font-medium text-[var(--foreground)]">
                            {e.service_name ?? e.id}
                          </span>
                          {e.service_name && e.service_name !== e.id && (
                            <span className="text-xs text-[var(--muted-foreground)] font-mono">
                              {e.id}
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

          <MetadataTable metadata={displayedMetadata} />

          {resolvedRefs && Object.keys(resolvedRefs).length > 0 && (
            <ResolvedRefsSection data={resolvedRefs} />
          )}

        </>
      )}
    </div>
  );
}
