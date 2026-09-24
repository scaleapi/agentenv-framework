import { useEffect, useState } from 'react';
import { ArrowLeft, Eye, EyeOff } from 'lucide-react';
import {
  BACKEND_URL,
  apiFetch,
  MetadataTable,
  formatCellValue,
} from './shared';
import {
  AdvertisedAgentCard,
  type AgentCard,
  type AgentCardValidation,
} from './advertised-agent-card';
import { FEATURED_LOGO_BY_ID } from './agents-hub-page';

interface ValidatedMCPMethod {
  supported: boolean;
  options?: Record<string, { supported: boolean }>;
}

interface ValidatedMCPEntry {
  supported: boolean;
  tool_call_count_reported?: boolean;
  methods?: Record<string, ValidatedMCPMethod>;
}

// Shape of `metadata.validated_modalities = { input: {<modality>: outcome}, declared: {...} }`,
// written by the `verify_a2a_modalities` step.
type ModalityReason =
  | 'passed'
  | 'no_ingestion_evidence'
  | 'protocol_reject'
  | 'probe_step_did_not_run';

interface ModalityOutcome {
  supported: boolean;
  reason: ModalityReason;
  error_type?: string;
  probe_response?: string;
}

interface ValidatedModalities {
  input?: Record<string, ModalityOutcome>;
  declared?: {
    default_input_modes?: string[] | null;
    default_output_modes?: string[] | null;
  };
}

function AgentCardView({
  card,
  validation,
  logoUrl,
  modalityValidation,
}: {
  card: AgentCard;
  validation?: AgentCardValidation;
  logoUrl?: string;
  modalityValidation?: Record<string, { supported: boolean }>;
}) {
  return (
    <div className="mb-6">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-3">
        Agent Card
      </h3>

      <div className="mb-4">
        <AdvertisedAgentCard
          card={card}
          validation={validation}
          logoUrl={logoUrl}
          modalityValidation={modalityValidation}
        />
      </div>
    </div>
  );
}

const HIDDEN_METADATA_KEYS = new Set([
  'agent_card',
  'validated_agent_card',
  'validated_a2a_extensions',
  'validated_data_extensions',
  'validated_a2a_protocol',
  // Rendered by <ValidatedModalitiesSection /> so it doesn't double-print
  // as a raw JSON blob in the generic metadata table below.
  'validated_modalities',
]);

const MODALITY_REASON_STYLES: Record<
  ModalityReason,
  { dot: string; pill: string; label: string }
> = {
  passed: {
    dot: 'bg-emerald-500',
    pill: 'border-emerald-300 bg-emerald-50 text-emerald-900',
    label: 'passed',
  },
  no_ingestion_evidence: {
    dot: 'bg-amber-500',
    pill: 'border-amber-300 bg-amber-50 text-amber-900',
    label: 'no ingestion evidence',
  },
  protocol_reject: {
    dot: 'bg-red-500',
    pill: 'border-red-300 bg-red-50 text-red-900',
    label: 'protocol reject',
  },
  probe_step_did_not_run: {
    dot: 'bg-gray-400',
    pill: 'border-gray-300 bg-gray-50 text-gray-700',
    label: 'did not run',
  },
};

function ValidatedModalitiesSection({
  modalities,
}: {
  modalities: ValidatedModalities;
}) {
  const inputs = modalities.input ?? {};
  const entries = Object.entries(inputs);
  if (entries.length === 0) return null;

  const declared = new Set(modalities.declared?.default_input_modes ?? []);

  // Preserve insertion order (agent-env iterates probes in declared order).
  // Sort secondarily: passed first, then by severity, so failures are
  // visually grouped together.
  const reasonRank: Record<ModalityReason, number> = {
    passed: 0,
    no_ingestion_evidence: 1,
    protocol_reject: 2,
    probe_step_did_not_run: 3,
  };
  const sorted = [...entries].sort(
    ([, a], [, b]) => reasonRank[a.reason] - reasonRank[b.reason],
  );

  const passedCount = sorted.filter(([, o]) => o.supported).length;

  return (
    <div className="mb-6">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-3">
        Validated Input Modes{' '}
        <span className="font-normal normal-case text-[var(--muted-foreground)]">
          ({passedCount}/{sorted.length} passed)
        </span>
      </h3>
      <div className="flex flex-wrap gap-2">
        {sorted.map(([modality, outcome]) => {
          const style =
            MODALITY_REASON_STYLES[outcome.reason] ??
            MODALITY_REASON_STYLES.probe_step_did_not_run;
          const wasDeclared = declared.has(modality);
          const title = outcome.error_type
            ? `${style.label} · error_type=${outcome.error_type}`
            : outcome.probe_response
            ? `${style.label} · response: ${outcome.probe_response.slice(
                0,
                160,
              )}`
            : style.label;
          return (
            <span
              key={modality}
              title={title}
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs ${style.pill}`}
            >
              <span className={`h-2 w-2 rounded-full ${style.dot}`} />
              <span className="font-mono">{modality}</span>
              <span className="text-[var(--muted-foreground)]">
                · {style.label}
              </span>
              {!wasDeclared && outcome.supported && (
                <span
                  className="ml-1 rounded bg-emerald-100 px-1 text-[10px] font-semibold uppercase tracking-wide text-emerald-700"
                  title="Modality wasn't in defaultInputModes but the probe passed"
                >
                  bonus
                </span>
              )}
              {wasDeclared && !outcome.supported && (
                <span
                  className="ml-1 rounded bg-red-100 px-1 text-[10px] font-semibold uppercase tracking-wide text-red-700"
                  title="Modality was declared in defaultInputModes but the probe failed"
                >
                  declared
                </span>
              )}
            </span>
          );
        })}
      </div>
    </div>
  );
}

export function AgentDetailPage({
  agentId,
  onBack,
}: {
  agentId: string;
  onBack: () => void;
}) {
  const [agent, setAgent] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  // default_env_vars can carry deployment credentials, so mask values until the
  // user explicitly reveals them (this surface is viewable by anyone with the agent).
  const [revealEnvVars, setRevealEnvVars] = useState(false);

  useEffect(() => {
    setLoading(true);
    setFetchError(null);
    apiFetch(`${BACKEND_URL}/api/v1/agents/${encodeURIComponent(agentId)}`)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to fetch (${res.status})`);
        return res.json();
      })
      .then(data => {
        setAgent(data);
        setLoading(false);
      })
      .catch(e => {
        setFetchError(e instanceof Error ? e.message : 'Failed to load');
        setLoading(false);
      });
  }, [agentId]);

  const defaultEnvVars = (agent?.default_env_vars ?? null) as Record<
    string,
    string
  > | null;
  const envVarEntries = defaultEnvVars ? Object.entries(defaultEnvVars) : [];
  const metadata = (agent?.metadata ?? null) as Record<string, unknown> | null;
  const agentCard = (metadata?.agent_card ?? null) as AgentCard | null;
  const validatedModalities = (metadata?.validated_modalities ??
    null) as ValidatedModalities | null;
  const validatedExtensions = (metadata?.validated_a2a_extensions ??
    {}) as Record<string, ValidatedMCPEntry | undefined>;
  const validatedAgentCardRaw = (metadata?.validated_agent_card ?? null) as {
    accessible?: boolean;
    required_fields?: Record<string, { present?: boolean }>;
    optional_fields?: Record<string, { present?: boolean }>;
  } | null;
  const dataExtensions = (metadata?.validated_data_extensions ?? {}) as Record<
    string,
    { supported: boolean }
  >;
  const protocolMethods = (metadata?.validated_a2a_protocol ?? {}) as Record<
    string,
    { supported: boolean }
  >;
  const hasAnyValidation =
    !!validatedAgentCardRaw ||
    Object.keys(validatedExtensions).length > 0 ||
    Object.keys(dataExtensions).length > 0 ||
    Object.keys(protocolMethods).length > 0;
  const cardValidation: AgentCardValidation | undefined = hasAnyValidation
    ? {
        accessible: validatedAgentCardRaw?.accessible,
        fieldPresence: validatedAgentCardRaw
          ? {
              ...Object.fromEntries(
                Object.entries(validatedAgentCardRaw.required_fields ?? {}).map(
                  ([name, info]) => [
                    name,
                    { present: !!info?.present, kind: 'required' as const },
                  ],
                ),
              ),
              ...Object.fromEntries(
                Object.entries(validatedAgentCardRaw.optional_fields ?? {}).map(
                  ([name, info]) => [
                    name,
                    { present: !!info?.present, kind: 'optional' as const },
                  ],
                ),
              ),
            }
          : undefined,
        validatedExtensions: Object.fromEntries(
          Object.entries(validatedExtensions).map(([uri, val]) => [
            uri,
            {
              supported: val?.supported === true,
              methods: val?.methods,
            },
          ]),
        ),
        dataExtensions,
        protocolMethods,
      }
    : undefined;
  const displayedMetadata: Record<string, unknown> | null = metadata
    ? Object.fromEntries(
        Object.entries(metadata).filter(
          ([key]) => !HIDDEN_METADATA_KEYS.has(key),
        ),
      )
    : null;

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

      {agent && !loading && (
        <>
          <div className="mb-6">
            <div className="flex items-center gap-3 flex-wrap">
              <h1 className="text-2xl font-semibold font-mono">{agentId}</h1>
              <span className="text-sm text-[var(--muted-foreground)]">
                v{String(agent.version)}
              </span>
              <span className="px-2 py-0.5 rounded text-xs font-medium bg-[var(--secondary)] text-[var(--foreground)]">
                a2a_agent
              </span>
            </div>
            {!!agent.created_at_utc && (
              <p className="text-sm text-[var(--muted-foreground)] mt-1">
                Last Modified:{' '}
                {formatCellValue('created_at_utc', agent.created_at_utc)}
              </p>
            )}
          </div>

          {envVarEntries.length > 0 && (
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-2">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
                  Default Env Vars ({envVarEntries.length})
                </h3>
                <button
                  type="button"
                  onClick={() => setRevealEnvVars(v => !v)}
                  aria-label={revealEnvVars ? 'Hide values' : 'Reveal values'}
                  aria-pressed={revealEnvVars}
                  className="inline-flex items-center gap-1 text-[11px] text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
                >
                  {revealEnvVars ? <EyeOff size={12} /> : <Eye size={12} />}
                  {revealEnvVars ? 'Hide' : 'Reveal'}
                </button>
              </div>
              <div className="rounded-lg border border-[var(--border)] overflow-hidden w-fit">
                <table className="text-sm">
                  <tbody>
                    {envVarEntries.map(([key, value]) => (
                      <tr
                        key={key}
                        className="border-b border-[var(--border)] last:border-b-0"
                      >
                        <td className="px-3 py-1.5 text-[var(--muted-foreground)] font-mono whitespace-nowrap">
                          {key}
                        </td>
                        <td className="px-3 py-1.5 text-[var(--foreground)] font-mono break-all">
                          {revealEnvVars ? value : '••••••••'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {!agentCard && !hasAnyValidation && (
            <div className="mb-6">
              <p className="text-xs text-[var(--muted-foreground)]">
                No agent card recorded for this agent.
              </p>
            </div>
          )}

          {agentCard && (
            <AgentCardView
              card={agentCard}
              validation={cardValidation}
              logoUrl={FEATURED_LOGO_BY_ID[agentId]}
              modalityValidation={validatedModalities?.input}
            />
          )}

          {validatedModalities && (
            <ValidatedModalitiesSection modalities={validatedModalities} />
          )}

          {displayedMetadata && <MetadataTable metadata={displayedMetadata} />}
        </>
      )}
    </div>
  );
}
