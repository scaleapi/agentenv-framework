import { useState } from 'react';
import {
  AlertTriangle,
  BookOpen,
  Check,
  ChevronRight,
  ExternalLink,
  Link as LinkIcon,
  Search,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react';
import { safeHref } from '../lib/safe-url';

export interface ValidatedExtensionMethod {
  supported: boolean;
  options?: Record<string, { supported: boolean }>;
}

export interface ValidatedExtension {
  supported: boolean;
  methods?: Record<string, ValidatedExtensionMethod>;
}

export interface AgentCardValidation {
  accessible?: boolean;
  fieldPresence?: Record<
    string,
    { present: boolean; kind: 'required' | 'optional' }
  >;
  validatedExtensions?: Record<string, ValidatedExtension>;
  dataExtensions?: Record<string, { supported: boolean }>;
  protocolMethods?: Record<string, { supported: boolean }>;
}

export interface AgentCardExtension {
  uri: string;
  description?: string;
  required?: boolean;
  params?: Record<string, unknown>;
}

export interface AgentCardSkill {
  id?: string;
  name?: string;
  description?: string;
  tags?: string[];
  examples?: string[];
  inputModes?: string[];
  outputModes?: string[];
  security?: Array<Record<string, string[]>>;
}

export interface AgentCardInterface {
  url: string;
  transport: string;
}

export interface AgentCardSecurityScheme {
  type?: string;
  description?: string;
  scheme?: string;
  bearerFormat?: string;
  in?: string;
  name?: string;
  flows?: Record<string, unknown>;
  openIdConnectUrl?: string;
  [key: string]: unknown;
}

export interface AgentCard {
  name?: string;
  description?: string;
  version?: string;
  protocolVersion?: string;
  url?: string;
  iconUrl?: string;
  documentationUrl?: string;
  preferredTransport?: string;
  additionalInterfaces?: AgentCardInterface[];
  provider?: { organization?: string; url?: string };
  capabilities?: {
    streaming?: boolean;
    pushNotifications?: boolean;
    stateTransitionHistory?: boolean;
    extensions?: AgentCardExtension[];
  };
  skills?: AgentCardSkill[];
  defaultInputModes?: string[];
  defaultOutputModes?: string[];
  securitySchemes?: Record<string, AgentCardSecurityScheme>;
  security?: Array<Record<string, string[]>>;
  supportsAuthenticatedExtendedCard?: boolean;
  [key: string]: unknown;
}

const KNOWN_CARD_KEYS = new Set<string>([
  'name',
  'description',
  'version',
  'protocolVersion',
  'url',
  'iconUrl',
  'documentationUrl',
  'preferredTransport',
  'additionalInterfaces',
  'provider',
  'capabilities',
  'skills',
  'defaultInputModes',
  'defaultOutputModes',
  'securitySchemes',
  'security',
  'supportsAuthenticatedExtendedCard',
]);

function collectAdvertisedOptions(
  advertisedMethod: Record<string, unknown> | undefined,
): string[] {
  if (!advertisedMethod) return [];
  const out = new Set<string>();
  for (const section of ['request', 'response'] as const) {
    const sec = (advertisedMethod[section] ?? {}) as Record<string, unknown>;
    for (const bucket of ['required', 'optional', 'supported'] as const) {
      const vals = sec[bucket];
      if (Array.isArray(vals))
        vals.forEach(v => typeof v === 'string' && out.add(v));
    }
    const oneOf = sec.oneOf;
    if (Array.isArray(oneOf)) {
      for (const entry of oneOf) {
        const required = (entry as Record<string, unknown>)?.required;
        if (Array.isArray(required))
          required.forEach(v => typeof v === 'string' && out.add(v));
      }
    }
  }
  return Array.from(out);
}

export function AgentCardExtensionDetail({
  extension,
  validation,
}: {
  extension: AgentCardExtension;
  validation?: ValidatedExtension;
}) {
  const [paramsOpen, setParamsOpen] = useState(false);
  const label = extension.uri.replace(/^urn:[^:]+:/, '') || extension.uri;
  const params = extension.params;
  const hasParams = params && Object.keys(params).length > 0;

  const extRaw = extension as unknown as Record<string, unknown>;
  const advConfig =
    ((extRaw.config ?? extRaw.params ?? {}) as Record<string, unknown>) || {};
  const advertisedMethods = (advConfig.methods ?? {}) as Record<
    string,
    Record<string, unknown>
  >;
  const validatedMethods = validation?.methods ?? {};
  const methodNames = Array.from(
    new Set([
      ...Object.keys(advertisedMethods),
      ...Object.keys(validatedMethods),
    ]),
  );

  let validationBadge: React.ReactNode = null;
  if (validation?.supported === false) {
    validationBadge = (
      <span className="inline-flex items-center gap-1 text-[9px] font-semibold uppercase px-1.5 py-0.5 rounded bg-red-50 text-red-700 border border-red-200">
        <X size={10} />
        Validation failed
      </span>
    );
  } else if (validation === undefined) {
    validationBadge = (
      <span className="inline-flex items-center gap-1 text-[9px] font-semibold uppercase px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200">
        <AlertTriangle size={10} />
        Not validated
      </span>
    );
  }

  return (
    <div className="rounded-lg border border-indigo-200 bg-indigo-50/40 overflow-hidden">
      <div className="px-3 py-2.5 border-b border-indigo-100">
        <div className="flex items-center gap-2 flex-wrap">
          <code className="text-xs font-mono font-semibold text-indigo-700">
            {label}
          </code>
          {extension.required && (
            <span className="text-[9px] font-semibold uppercase px-1.5 py-0.5 rounded bg-indigo-100 text-indigo-700">
              required
            </span>
          )}
          {validationBadge}
        </div>
        <code className="block text-[10px] font-mono text-[var(--muted-foreground)] mt-1 break-all">
          {extension.uri}
        </code>
        {extension.description && (
          <p className="text-xs text-[var(--foreground)] mt-1.5 leading-relaxed">
            {extension.description}
          </p>
        )}
      </div>

      {methodNames.length > 0 && (
        <div className="px-3 py-3 border-b border-indigo-100 space-y-2">
          <div className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] font-semibold">
            Methods ({methodNames.length})
          </div>
          {methodNames.map(name => {
            const valMethod = validatedMethods[name];
            const advertisedMethod = advertisedMethods[name];
            const advOpts = collectAdvertisedOptions(advertisedMethod);
            const valOpts = Object.entries(valMethod?.options ?? {});
            const optionStatus = new Map<string, 'supported' | 'unsupported'>();
            for (const [n, opt] of valOpts) {
              optionStatus.set(n, opt.supported ? 'supported' : 'unsupported');
            }
            const unionOptions = Array.from(
              new Set([...advOpts, ...valOpts.map(([n]) => n)]),
            );
            const status =
              valMethod?.supported === true
                ? 'supported'
                : valMethod?.supported === false
                ? 'unsupported'
                : 'unknown';
            const httpMethod =
              typeof advertisedMethod?.method === 'string'
                ? (advertisedMethod.method as string)
                : null;
            return (
              <div
                key={name}
                className="rounded-md border border-indigo-100 bg-[var(--background)] p-2"
              >
                <div className="flex items-center gap-2 flex-wrap">
                  {status === 'unsupported' && (
                    <X size={12} className="text-red-500" />
                  )}
                  {status === 'unknown' && (
                    <AlertTriangle size={12} className="text-amber-600" />
                  )}
                  <code className="text-xs font-medium font-mono text-[var(--foreground)]">
                    {name}
                  </code>
                  {httpMethod && (
                    <span className="text-[9px] font-mono uppercase px-1 py-0.5 rounded bg-[var(--secondary)] text-[var(--muted-foreground)]">
                      {httpMethod}
                    </span>
                  )}
                  {status === 'unsupported' && (
                    <span className="text-[9px] uppercase text-red-600">
                      unsupported
                    </span>
                  )}
                  {status === 'unknown' && (
                    <span className="text-[9px] uppercase text-amber-700">
                      not validated
                    </span>
                  )}
                </div>
                {unionOptions.length > 0 && (
                  <div className="mt-2 ml-4">
                    <div className="text-[9px] uppercase tracking-wider text-[var(--muted-foreground)] mb-1">
                      Options
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {unionOptions.map(opt => {
                        const status = optionStatus.get(opt);
                        const cls =
                          status === 'supported'
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            : status === 'unsupported'
                            ? 'bg-red-50 text-red-700 border border-red-200'
                            : 'bg-[var(--secondary)] text-[var(--muted-foreground)] border border-[var(--border)]';
                        return (
                          <span
                            key={opt}
                            title={
                              status === 'supported'
                                ? 'supported'
                                : status === 'unsupported'
                                ? 'not supported'
                                : 'not validated'
                            }
                            className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono ${cls}`}
                          >
                            {status === 'supported' && <Check size={8} />}
                            {status === 'unsupported' && <X size={8} />}
                            {opt}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {hasParams && (
        <>
          <button
            onClick={() => setParamsOpen(v => !v)}
            className="w-full px-3 py-1.5 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] hover:text-[var(--foreground)] bg-indigo-50/30"
          >
            <ChevronRight
              size={11}
              className={`transition-transform ${
                paramsOpen ? 'rotate-90' : ''
              }`}
            />
            {paramsOpen ? 'Hide' : 'View'} params ({Object.keys(params).length})
          </button>
          {paramsOpen && (
            <pre className="px-3 py-2 text-[10px] font-mono leading-relaxed bg-[var(--background)] border-t border-indigo-100 max-h-[300px] overflow-auto whitespace-pre-wrap break-all">
              {JSON.stringify(params, null, 2)}
            </pre>
          )}
        </>
      )}
      {!hasParams && methodNames.length === 0 && (
        <div className="px-3 py-1.5 text-[10px] text-[var(--muted-foreground)] italic">
          No params
        </div>
      )}
    </div>
  );
}

function Chips({ values }: { values: string[] }) {
  if (values.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1">
      {values.map(v => (
        <code
          key={v}
          className="px-1.5 py-0.5 rounded bg-[var(--secondary)] text-[10px] font-mono text-[var(--foreground)]"
        >
          {v}
        </code>
      ))}
    </div>
  );
}

/**
 * Color-coded chip list for declared input modes: green — probe passed, red —
 * probe failed, gray — not validated yet. Reads a `{ supported: boolean }` map.
 */
function ValidatedInputModeChips({
  values,
  validation,
}: {
  values: string[];
  validation?: Record<string, { supported: boolean }>;
}) {
  if (values.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1">
      {values.map(v => {
        const outcome = validation?.[v];
        const cls =
          outcome === undefined
            ? 'bg-[var(--secondary)] text-[var(--foreground)]'
            : outcome.supported
            ? 'bg-emerald-50 text-emerald-900 border border-emerald-200'
            : 'bg-red-50 text-red-900 border border-red-200';
        const title =
          outcome === undefined
            ? 'Not validated'
            : outcome.supported
            ? 'Validated as available'
            : 'Validated as unavailable';
        return (
          <code
            key={v}
            title={title}
            className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${cls}`}
          >
            {v}
          </code>
        );
      })}
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] mb-1">
        {label}
      </div>
      {children}
    </div>
  );
}

function SkillsSection({ skills }: { skills: AgentCardSkill[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  if (skills.length === 0) return null;
  const filtered = skills.filter(
    s =>
      !search ||
      s.name?.toLowerCase().includes(search.toLowerCase()) ||
      s.description?.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <div className="px-5 py-4 border-t border-[var(--border)]">
      <div className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
        Skills ({skills.length})
      </div>
      {skills.length > 3 && (
        <div className="mb-2 relative">
          <Search
            size={12}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--muted-foreground)]"
          />
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search skills"
            className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] text-xs pl-7 pr-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-purple-200"
          />
        </div>
      )}
      <div className="rounded-md border border-[var(--border)] bg-[var(--background)] overflow-hidden max-h-[400px] overflow-y-auto">
        {filtered.map((skill, idx) => {
          const key = skill.id ?? skill.name ?? String(idx);
          const isExp = expanded === key;
          return (
            <div
              key={key}
              className={`border-b border-[var(--border)] last:border-b-0 ${
                isExp ? 'bg-[var(--accent)]/30' : ''
              }`}
            >
              <button
                onClick={() => setExpanded(isExp ? null : key)}
                className="w-full text-left px-3 py-2.5 flex items-start gap-2 hover:bg-[var(--accent)]/30 transition-colors"
              >
                <ChevronRight
                  size={12}
                  className={`flex-shrink-0 mt-1 text-[var(--muted-foreground)] transition-transform ${
                    isExp ? 'rotate-90' : ''
                  }`}
                />
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium">
                    {skill.name ?? skill.id ?? '(unnamed skill)'}
                  </div>
                  {skill.description && (
                    <div
                      className={`text-xs text-[var(--muted-foreground)] mt-1 leading-relaxed ${
                        isExp ? '' : 'line-clamp-1'
                      }`}
                    >
                      {skill.description}
                    </div>
                  )}
                </div>
              </button>
              {isExp && (
                <div className="px-3 pb-2.5 pl-8 space-y-2">
                  {skill.tags && skill.tags.length > 0 && (
                    <div className="text-xs">
                      <span className="text-[10px] uppercase text-[var(--muted-foreground)]">
                        Tags:
                      </span>
                      {skill.tags.map(t => (
                        <span
                          key={t}
                          className="ml-1 inline-block px-1.5 py-0.5 rounded bg-[var(--secondary)] text-[10px] font-mono"
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                  {skill.inputModes && skill.inputModes.length > 0 && (
                    <div className="text-xs text-[var(--muted-foreground)]">
                      <span className="uppercase text-[10px]">
                        Input modes:
                      </span>{' '}
                      {skill.inputModes.join(', ')}
                    </div>
                  )}
                  {skill.outputModes && skill.outputModes.length > 0 && (
                    <div className="text-xs text-[var(--muted-foreground)]">
                      <span className="uppercase text-[10px]">
                        Output modes:
                      </span>{' '}
                      {skill.outputModes.join(', ')}
                    </div>
                  )}
                  {skill.examples && skill.examples.length > 0 && (
                    <div className="text-xs text-[var(--muted-foreground)]">
                      <span className="uppercase text-[10px]">Examples:</span>
                      <ul className="mt-1 space-y-0.5 list-disc list-inside">
                        {skill.examples.map((ex, i) => (
                          <li key={i} className="text-xs">
                            {ex}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const CAPABILITY_DEFS: Array<{
  key: 'streaming' | 'pushNotifications' | 'stateTransitionHistory';
  label: string;
}> = [
  { key: 'streaming', label: 'Streaming' },
  { key: 'pushNotifications', label: 'Push Notifications' },
  { key: 'stateTransitionHistory', label: 'State Transition History' },
];

export function AdvertisedAgentCard({
  card,
  validation,
  logoUrl,
  modalityValidation,
}: {
  card: AgentCard;
  validation?: AgentCardValidation;
  logoUrl?: string;
  /**
   * Per-modality outcomes from the `verify_a2a_modalities` task step.
   * When supplied, declared input-mode chips are colored green (supported)
   * / red (failed) / left gray (no entry → not validated).
   */
  modalityValidation?: Record<string, { supported: boolean }>;
}) {
  const [rawOpen, setRawOpen] = useState(false);
  const [openExtension, setOpenExtension] = useState<string | null>(null);
  const effectiveIconUrl = card.iconUrl ?? logoUrl;

  const validatedExtensions = validation?.validatedExtensions ?? {};
  const dataExtensions = validation?.dataExtensions ?? {};
  const protocolMethods = validation?.protocolMethods ?? {};
  const fieldPresence = validation?.fieldPresence ?? {};
  const missingRequired = Object.entries(fieldPresence)
    .filter(([, info]) => info.kind === 'required' && !info.present)
    .map(([name]) => name);

  const skills = card.skills ?? [];
  const extensions = card.capabilities?.extensions ?? [];
  const providerName = card.provider?.organization;
  const providerUrl = card.provider?.url;
  const inputModes = card.defaultInputModes ?? [];
  const outputModes = card.defaultOutputModes ?? [];
  const additionalInterfaces = card.additionalInterfaces ?? [];
  const securitySchemes = card.securitySchemes ?? {};
  const security = card.security ?? [];
  const caps = card.capabilities ?? {};

  const unknownKeys = Object.keys(card).filter(k => !KNOWN_CARD_KEYS.has(k));

  const capabilityFlags: Array<{
    key: string;
    label: string;
    enabled: boolean;
  }> = CAPABILITY_DEFS.filter(def => caps[def.key] !== undefined).map(def => ({
    key: def.key,
    label: def.label,
    enabled: !!caps[def.key],
  }));

  return (
    <div className="rounded-xl border border-[var(--border)] bg-gradient-to-br from-[var(--background)] to-[var(--secondary)]/40 overflow-hidden">
      <div className="px-5 py-4 border-b border-[var(--border)] bg-[var(--secondary)]/30">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-3 min-w-0 flex-1">
            {effectiveIconUrl ? (
              <img
                src={effectiveIconUrl}
                alt=""
                className="w-10 h-10 rounded-lg border border-[var(--border)] bg-white object-contain p-1.5 flex-shrink-0"
              />
            ) : (
              <div className="w-10 h-10 rounded-lg border border-[var(--border)] bg-[var(--background)] flex items-center justify-center flex-shrink-0">
                <Sparkles size={16} className="text-purple-500" />
              </div>
            )}
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-2 flex-wrap">
                <h4 className="text-lg font-semibold text-[var(--foreground)] truncate">
                  {card.name ?? '(unnamed agent)'}
                </h4>
                {card.version && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-[var(--background)] border border-[var(--border)] text-[var(--muted-foreground)]">
                    card v{card.version}
                  </span>
                )}
                {card.protocolVersion && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-[var(--background)] border border-[var(--border)] text-[var(--muted-foreground)]">
                    A2A {card.protocolVersion}
                  </span>
                )}
                {validation?.accessible === true && (
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                    <Check size={10} />
                    Accessible
                  </span>
                )}
                {validation?.accessible === false && (
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-red-50 text-red-700 border border-red-200">
                    Inaccessible
                  </span>
                )}
                {card.documentationUrl && (
                  <a
                    href={safeHref(card.documentationUrl)}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-[10px] text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
                    onClick={e => e.stopPropagation()}
                  >
                    <BookOpen size={11} />
                    Docs
                  </a>
                )}
              </div>
              {card.description && (
                <p className="text-sm text-[var(--muted-foreground)] mt-1.5 leading-relaxed">
                  {card.description}
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-4 flex-shrink-0">
            <div className="text-center">
              <div className="text-xl font-semibold text-[var(--foreground)]">
                {skills.length}
              </div>
              <div className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)]">
                skill{skills.length !== 1 ? 's' : ''}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xl font-semibold text-[var(--foreground)]">
                {extensions.length}
              </div>
              <div className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)]">
                extension{extensions.length !== 1 ? 's' : ''}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="px-5 py-4 border-b border-[var(--border)]">
        <div className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
          Capabilities
        </div>
        {(() => {
          const hasFlags =
            capabilityFlags.length > 0 ||
            card.supportsAuthenticatedExtendedCard !== undefined;
          const dataExtEntries = Object.entries(dataExtensions);
          const protocolEntries = Object.entries(protocolMethods);
          const hasExtensions = extensions.length > 0;
          const hasDataExtensions = dataExtEntries.length > 0;
          const hasProtocol = protocolEntries.length > 0;
          if (
            !hasFlags &&
            !hasExtensions &&
            !hasDataExtensions &&
            !hasProtocol
          ) {
            return (
              <div className="text-xs text-[var(--muted-foreground)] italic">
                None advertised
              </div>
            );
          }
          return (
            <>
              {hasFlags && (
                <div className="flex flex-wrap gap-2">
                  {capabilityFlags.map(flag => (
                    <span
                      key={flag.key}
                      title={flag.enabled ? 'enabled' : 'not enabled'}
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium ${
                        flag.enabled
                          ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                          : 'bg-[var(--secondary)] text-[var(--muted-foreground)] border border-[var(--border)]'
                      }`}
                    >
                      {flag.label}
                      {!flag.enabled && (
                        <span className="text-[9px] uppercase">
                          not enabled
                        </span>
                      )}
                    </span>
                  ))}
                  {card.supportsAuthenticatedExtendedCard !== undefined && (
                    <span
                      title={
                        card.supportsAuthenticatedExtendedCard
                          ? 'enabled'
                          : 'not enabled'
                      }
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium ${
                        card.supportsAuthenticatedExtendedCard
                          ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                          : 'bg-[var(--secondary)] text-[var(--muted-foreground)] border border-[var(--border)]'
                      }`}
                    >
                      <ShieldCheck size={10} />
                      Authenticated extended card
                      {!card.supportsAuthenticatedExtendedCard && (
                        <span className="text-[9px] uppercase">
                          not enabled
                        </span>
                      )}
                    </span>
                  )}
                </div>
              )}
              {hasExtensions && (
                <div className={hasFlags ? 'mt-3' : ''}>
                  <div className="text-[9px] uppercase tracking-wider text-[var(--muted-foreground)] mb-1.5">
                    Extensions
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {extensions.map(ext => {
                      const label =
                        ext.uri.replace(/^urn:[^:]+:/, '') || ext.uri;
                      const isOpen = openExtension === ext.uri;
                      const extValidation = validatedExtensions[ext.uri];
                      const validationState =
                        extValidation === undefined
                          ? 'unvalidated'
                          : extValidation.supported
                          ? 'ok'
                          : 'failed';
                      return (
                        <button
                          key={ext.uri}
                          type="button"
                          onClick={() =>
                            setOpenExtension(isOpen ? null : ext.uri)
                          }
                          title={
                            ext.description
                              ? `${ext.uri}\n${ext.description}`
                              : ext.uri
                          }
                          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium font-mono border transition-colors ${
                            isOpen
                              ? 'bg-indigo-600 text-white border-indigo-600'
                              : 'bg-indigo-50 text-indigo-700 border-indigo-200 hover:bg-indigo-100'
                          }`}
                        >
                          {validationState === 'failed' && (
                            <X
                              size={10}
                              className={
                                isOpen ? 'text-red-200' : 'text-red-600'
                              }
                            />
                          )}
                          {validationState === 'unvalidated' && (
                            <AlertTriangle
                              size={10}
                              className={
                                isOpen ? 'text-amber-200' : 'text-amber-600'
                              }
                            />
                          )}
                          {label}
                          {ext.required && (
                            <span
                              className={`text-[9px] uppercase ${
                                isOpen ? 'text-indigo-100' : 'text-indigo-500'
                              }`}
                            >
                              required
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
              {hasDataExtensions && (
                <div className={hasFlags || hasExtensions ? 'mt-3' : ''}>
                  <div className="text-[9px] uppercase tracking-wider text-[var(--muted-foreground)] mb-1.5">
                    Data Extensions
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {dataExtEntries.map(([name, info]) => (
                      <span
                        key={`data-${name}`}
                        title={
                          info.supported
                            ? `data extension: ${name}`
                            : `data extension: ${name} — not supported`
                        }
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium font-mono ${
                          info.supported
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            : 'bg-red-50 text-red-700 border border-red-200'
                        }`}
                      >
                        {!info.supported && <X size={10} />}
                        {name}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {hasProtocol && (
                <div
                  className={
                    hasFlags || hasExtensions || hasDataExtensions ? 'mt-3' : ''
                  }
                >
                  <div className="text-[9px] uppercase tracking-wider text-[var(--muted-foreground)] mb-1.5">
                    A2A Protocol
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {protocolEntries.map(([name, info]) => (
                      <span
                        key={`proto-${name}`}
                        title={
                          info.supported
                            ? `A2A method: ${name}`
                            : `A2A method: ${name} — not supported`
                        }
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium font-mono ${
                          info.supported
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                            : 'bg-red-50 text-red-700 border border-red-200'
                        }`}
                      >
                        {!info.supported && <X size={10} />}
                        {name}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </>
          );
        })()}
        {openExtension && (
          <div className="mt-3">
            {(() => {
              const ext = extensions.find(e => e.uri === openExtension);
              return ext ? (
                <AgentCardExtensionDetail
                  extension={ext}
                  validation={validatedExtensions[ext.uri]}
                />
              ) : null;
            })()}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 px-5 py-4 text-xs">
        {providerName && (
          <Field label="Provider">
            <div className="text-sm text-[var(--foreground)] flex items-center gap-1.5">
              {providerName}
              {providerUrl && (
                <a
                  href={safeHref(providerUrl)}
                  target="_blank"
                  rel="noreferrer"
                  className="text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
                  onClick={e => e.stopPropagation()}
                >
                  <ExternalLink size={11} />
                </a>
              )}
            </div>
          </Field>
        )}
        {card.url && (
          <Field label="URL">
            <div className="flex items-center gap-1.5 min-w-0">
              <LinkIcon
                size={11}
                className="text-[var(--muted-foreground)] flex-shrink-0"
              />
              <code className="text-xs font-mono text-[var(--foreground)] truncate">
                {card.url}
              </code>
            </div>
          </Field>
        )}
        {card.preferredTransport && (
          <Field label="Preferred transport">
            <code className="text-xs font-mono text-[var(--foreground)]">
              {card.preferredTransport}
            </code>
          </Field>
        )}
        {inputModes.length > 0 && (
          <Field label="Default input modes">
            <ValidatedInputModeChips
              values={inputModes}
              validation={modalityValidation}
            />
          </Field>
        )}
        {outputModes.length > 0 && (
          <Field label="Default output modes">
            <Chips values={outputModes} />
          </Field>
        )}
        {additionalInterfaces.length > 0 && (
          <Field label="Additional interfaces">
            <div className="space-y-1">
              {additionalInterfaces.map((iface, i) => (
                <div
                  key={`${iface.url}-${i}`}
                  className="flex items-center gap-2 min-w-0"
                >
                  <code className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--secondary)] flex-shrink-0">
                    {iface.transport}
                  </code>
                  <code className="text-xs font-mono text-[var(--foreground)] truncate">
                    {iface.url}
                  </code>
                </div>
              ))}
            </div>
          </Field>
        )}
        {Object.keys(securitySchemes).length > 0 && (
          <Field label="Security schemes">
            <div className="space-y-1">
              {Object.entries(securitySchemes).map(([name, scheme]) => (
                <div key={name} className="flex items-center gap-2">
                  <code className="text-xs font-mono text-[var(--foreground)]">
                    {name}
                  </code>
                  {scheme.type && (
                    <span className="text-[10px] text-[var(--muted-foreground)]">
                      ({scheme.type}
                      {scheme.scheme ? `/${scheme.scheme}` : ''})
                    </span>
                  )}
                </div>
              ))}
            </div>
          </Field>
        )}
        {security.length > 0 && (
          <Field label="Security requirements">
            <div className="space-y-1">
              {security.map((req, i) => (
                <div key={i} className="flex flex-wrap gap-1">
                  {Object.entries(req).map(([scheme, scopes]) => (
                    <span
                      key={scheme}
                      className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-[var(--secondary)] text-[10px] font-mono"
                    >
                      {scheme}
                      {scopes.length > 0 && (
                        <span className="text-[var(--muted-foreground)]">
                          [{scopes.join(', ')}]
                        </span>
                      )}
                    </span>
                  ))}
                </div>
              ))}
            </div>
          </Field>
        )}
        {unknownKeys.map(key => (
          <Field key={key} label={key}>
            <code className="text-[10px] font-mono text-[var(--foreground)] break-all">
              {JSON.stringify(card[key])}
            </code>
          </Field>
        ))}
      </div>

      {card.skills && card.skills.length > 0 && (
        <SkillsSection skills={card.skills} />
      )}

      {missingRequired.length > 0 && (
        <div className="px-5 py-3 border-t border-[var(--border)] bg-amber-50/40 flex items-start gap-2">
          <AlertTriangle
            size={14}
            className="text-amber-600 flex-shrink-0 mt-0.5"
          />
          <div className="min-w-0">
            <div className="text-[11px] font-semibold text-amber-700">
              Missing required fields
            </div>
            <div className="text-xs text-amber-700 mt-0.5 font-mono break-all">
              {missingRequired.join(', ')}
            </div>
          </div>
        </div>
      )}

      <button
        onClick={() => setRawOpen(v => !v)}
        className="w-full px-5 py-2 text-left border-t border-[var(--border)] bg-[var(--secondary)]/20 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
      >
        <ChevronRight
          size={11}
          className={`transition-transform ${rawOpen ? 'rotate-90' : ''}`}
        />
        {rawOpen ? 'Hide' : 'View'} raw card JSON
      </button>
      {rawOpen && (
        <pre className="px-5 py-3 text-[10px] font-mono leading-relaxed bg-[var(--background)] border-t border-[var(--border)] max-h-[400px] overflow-auto whitespace-pre-wrap break-all">
          {JSON.stringify(card, null, 2)}
        </pre>
      )}
    </div>
  );
}
