import { useState } from 'react';
import {
  AlertTriangle,
  Boxes,
  Check,
  ChevronRight,
  Link as LinkIcon,
  Server,
} from 'lucide-react';

export interface EnvironmentCardInterface {
  url: string;
  transport: string;
}

export interface EnvironmentCardExtension {
  uri: string;
  description?: string;
  required?: boolean;
  params?: Record<string, unknown>;
}

export interface EnvironmentCardCapabilities {
  extensions?: EnvironmentCardExtension[];
}

export interface EnvironmentCard {
  name?: string;
  protocolVersion?: string;
  url?: string;
  preferredTransport?: string;
  additionalInterfaces?: EnvironmentCardInterface[];
  capabilities?: EnvironmentCardCapabilities;
  children_environments?: EnvironmentCard[] | null;
  [key: string]: unknown;
}

/** Normalized `validated_environment_card`. Validation is coarse (card reachable +
 *  required fields present); extensions are shown as advertised, not graded. */
export interface EnvironmentCardValidation {
  accessible?: boolean;
  requiredFields?: Record<string, { present: boolean }>;
  childrenCount?: number;
  extensions?: string[];
  error?: string;
}

const KNOWN_ENV_CARD_KEYS = new Set<string>([
  'name',
  'protocolVersion',
  'url',
  'preferredTransport',
  'additionalInterfaces',
  'capabilities',
  'children_environments',
]);

function stripUrn(uri: string): string {
  return uri.replace(/^urn:[^:]+:/, '') || uri;
}

/** Concise type label for a JSON-schema param: primitives verbatim, `T[]`, `"x"`
 *  for a const, `A | B` for unions. Empty string when unknown. */
function paramTypeLabel(schema: unknown): string {
  if (!schema || typeof schema !== 'object') return '';
  const s = schema as Record<string, unknown>;
  if (typeof s.type === 'string') {
    if (s.type === 'array') {
      const item = paramTypeLabel(s.items);
      return item ? `${item}[]` : 'array';
    }
    return s.type;
  }
  if (Array.isArray(s.type)) {
    return (s.type as unknown[]).filter(t => typeof t === 'string').join(' | ');
  }
  if ('const' in s) {
    return typeof s.const === 'string' ? `"${s.const}"` : String(s.const);
  }
  if (Array.isArray(s.enum)) return 'enum';
  const union = (s.anyOf ?? s.oneOf) as unknown[] | undefined;
  if (Array.isArray(union)) {
    return Array.from(new Set(union.map(paramTypeLabel).filter(Boolean))).join(
      ' | ',
    );
  }
  return '';
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

/** Detail panel for one gateway extension: its endpoint, each method's verb +
 *  request params, and a raw-params fallback. */
function EnvCardExtensionDetail({
  extension,
}: {
  extension: EnvironmentCardExtension;
}) {
  const [paramsOpen, setParamsOpen] = useState(false);
  const params = (extension.params ?? {}) as Record<string, unknown>;
  const endpoint = typeof params.endpoint === 'string' ? params.endpoint : null;
  const methods = (params.methods ?? {}) as Record<
    string,
    Record<string, unknown>
  >;
  const methodNames = Object.keys(methods);
  const hasParams = Object.keys(params).length > 0;

  return (
    <div className="rounded-lg border border-indigo-200 bg-indigo-50/40 overflow-hidden">
      <div className="px-3 py-2.5 border-b border-indigo-100">
        <div className="flex items-center gap-2 flex-wrap">
          <code className="text-xs font-mono font-semibold text-indigo-700">
            {stripUrn(extension.uri)}
          </code>
          {extension.required && (
            <span className="text-[9px] font-semibold uppercase px-1.5 py-0.5 rounded bg-indigo-100 text-indigo-700">
              required
            </span>
          )}
          {endpoint && (
            <code className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--background)] border border-indigo-100 text-[var(--muted-foreground)]">
              {endpoint}
            </code>
          )}
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
            const spec = methods[name] ?? {};
            const httpMethod =
              typeof spec.method === 'string' ? (spec.method as string) : null;
            const request = (spec.request ?? {}) as Record<string, unknown>;
            const properties = (request.properties ?? {}) as Record<
              string,
              unknown
            >;
            const required = Array.isArray(request.required)
              ? (request.required as string[])
              : [];
            const fields = Object.keys(properties);
            return (
              <div
                key={name}
                className="rounded-md border border-indigo-100 bg-[var(--background)] p-2"
              >
                <div className="flex items-center gap-2 flex-wrap">
                  <code className="text-xs font-medium font-mono text-[var(--foreground)]">
                    {name}
                  </code>
                  {httpMethod && (
                    <span className="text-[9px] font-mono uppercase px-1 py-0.5 rounded bg-[var(--secondary)] text-[var(--muted-foreground)]">
                      {httpMethod}
                    </span>
                  )}
                </div>
                {fields.length > 0 &&
                  (() => {
                    const requiredFields = fields.filter(f =>
                      required.includes(f),
                    );
                    const optionalFields = fields.filter(
                      f => !required.includes(f),
                    );
                    // Required = filled chip; optional = subtle fill + dashed border.
                    const chips = (names: string[], isRequired: boolean) => (
                      <div className="flex flex-wrap gap-1">
                        {names.map(f => {
                          const type = paramTypeLabel(properties[f]);
                          return (
                            <span
                              key={f}
                              className={`inline-flex items-baseline gap-1.5 px-1.5 py-0.5 rounded text-[10px] font-mono border text-[var(--foreground)] ${
                                isRequired
                                  ? 'bg-[var(--secondary)] font-medium border-[var(--border)]'
                                  : 'bg-[var(--secondary)]/40 border-dashed border-[var(--muted-foreground)]/40'
                              }`}
                            >
                              {f}
                              {type && (
                                <span className="text-[9px] font-sans text-[var(--muted-foreground)]">
                                  {type}
                                </span>
                              )}
                            </span>
                          );
                        })}
                      </div>
                    );
                    const sublabel = (text: string) => (
                      <div className="text-[9px] uppercase tracking-wider text-[var(--muted-foreground)] mb-1">
                        {text}
                      </div>
                    );
                    // Mixed → labeled Required / Optional groups; single kind → flat list.
                    if (
                      requiredFields.length > 0 &&
                      optionalFields.length > 0
                    ) {
                      return (
                        <div className="mt-2 ml-1 space-y-2">
                          <div>
                            {sublabel('Required')}
                            {chips(requiredFields, true)}
                          </div>
                          <div>
                            {sublabel('Optional')}
                            {chips(optionalFields, false)}
                          </div>
                        </div>
                      );
                    }
                    return (
                      <div className="mt-2 ml-1">
                        {sublabel('Parameters')}
                        {chips(fields, requiredFields.length > 0)}
                      </div>
                    );
                  })()}
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
            {paramsOpen ? 'Hide' : 'View'} raw params
          </button>
          {paramsOpen && (
            <pre className="px-3 py-2 text-[10px] font-mono leading-relaxed bg-[var(--background)] border-t border-indigo-100 max-h-[300px] overflow-auto whitespace-pre-wrap break-all">
              {JSON.stringify(params, null, 2)}
            </pre>
          )}
        </>
      )}
    </div>
  );
}

/** Extensions chip row + expandable detail, shared by the card and each child env. */
function ExtensionsBlock({
  extensions,
}: {
  extensions: EnvironmentCardExtension[];
}) {
  const [openExtension, setOpenExtension] = useState<string | null>(null);
  if (extensions.length === 0) {
    return (
      <div className="text-xs text-[var(--muted-foreground)] italic">
        None advertised
      </div>
    );
  }
  const open = extensions.find(e => e.uri === openExtension);
  return (
    <>
      <div className="flex flex-wrap gap-2">
        {extensions.map(ext => {
          const isOpen = openExtension === ext.uri;
          return (
            <button
              key={ext.uri}
              type="button"
              onClick={() => setOpenExtension(isOpen ? null : ext.uri)}
              title={
                ext.description ? `${ext.uri}\n${ext.description}` : ext.uri
              }
              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium font-mono border transition-colors ${
                isOpen
                  ? 'bg-indigo-600 text-white border-indigo-600'
                  : 'bg-indigo-50 text-indigo-700 border-indigo-200 hover:bg-indigo-100'
              }`}
            >
              {stripUrn(ext.uri)}
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
      {open && (
        <div className="mt-3">
          <EnvCardExtensionDetail extension={open} />
        </div>
      )}
    </>
  );
}

/** One row in the Child Environments list — collapsed shows name / url / extension
 *  count; expanded shows the child's URL, transport, and extensions. */
function ChildEnvironmentRow({ child }: { child: EnvironmentCard }) {
  const [expanded, setExpanded] = useState(false);
  const extensions = child.capabilities?.extensions ?? [];
  const additionalInterfaces = child.additionalInterfaces ?? [];
  const grandchildren = child.children_environments ?? [];
  return (
    <div
      className={`border-b border-[var(--border)] last:border-b-0 ${
        expanded ? 'bg-[var(--accent)]/30' : ''
      }`}
    >
      <button
        onClick={() => setExpanded(v => !v)}
        className="w-full text-left px-3 py-2.5 flex items-center gap-2 hover:bg-[var(--accent)]/30 transition-colors"
      >
        <ChevronRight
          size={12}
          className={`flex-shrink-0 text-[var(--muted-foreground)] transition-transform ${
            expanded ? 'rotate-90' : ''
          }`}
        />
        <Server size={13} className="flex-shrink-0 text-indigo-500" />
        <span className="text-sm font-medium text-[var(--foreground)] flex-shrink-0">
          {child.name ?? '(unnamed)'}
        </span>
        <code className="text-[11px] font-mono text-[var(--muted-foreground)] truncate">
          {child.url}
        </code>
        <span className="ml-auto flex items-center gap-1.5 flex-shrink-0">
          {extensions.length > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
              {extensions.length} ext
            </span>
          )}
          {child.preferredTransport && (
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--secondary)] text-[var(--muted-foreground)]">
              {child.preferredTransport}
            </span>
          )}
        </span>
      </button>
      {expanded && (
        <div className="px-3 pb-3 pl-8 space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2 text-xs pt-1">
            <Field label="URL">
              <div className="flex items-center gap-1.5 min-w-0">
                <LinkIcon
                  size={11}
                  className="text-[var(--muted-foreground)] flex-shrink-0"
                />
                <code className="text-xs font-mono text-[var(--foreground)] truncate">
                  {child.url}
                </code>
              </div>
            </Field>
            {child.preferredTransport && (
              <Field label="Preferred transport">
                <code className="text-xs font-mono text-[var(--foreground)]">
                  {child.preferredTransport}
                </code>
              </Field>
            )}
          </div>
          <div>
            <div className="text-[9px] uppercase tracking-wider text-[var(--muted-foreground)] mb-1.5">
              Extensions
            </div>
            <ExtensionsBlock extensions={extensions} />
          </div>
          {additionalInterfaces.length > 0 && (
            <div>
              <div className="text-[9px] uppercase tracking-wider text-[var(--muted-foreground)] mb-1.5">
                Additional interfaces
              </div>
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
            </div>
          )}
          {grandchildren.length > 0 && (
            <div className="text-[10px] text-[var(--muted-foreground)] italic">
              {grandchildren.length} nested child environment
              {grandchildren.length !== 1 ? 's' : ''}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Renders a composed EnvironmentCard: header pills, gateway extensions, a field
 *  grid, and the recursive Child Environments list. */
export function AdvertisedEnvironmentCard({
  card,
  validation,
}: {
  card: EnvironmentCard;
  validation?: EnvironmentCardValidation;
}) {
  const [rawOpen, setRawOpen] = useState(false);
  const extensions = card.capabilities?.extensions ?? [];
  const children = card.children_environments ?? [];
  const additionalInterfaces = card.additionalInterfaces ?? [];
  const requiredFields = validation?.requiredFields ?? {};
  const missingRequired = Object.entries(requiredFields)
    .filter(([, info]) => !info.present)
    .map(([name]) => name);
  const unknownKeys = Object.keys(card).filter(
    k => !KNOWN_ENV_CARD_KEYS.has(k),
  );

  return (
    <div className="rounded-xl border border-[var(--border)] bg-gradient-to-br from-[var(--background)] to-[var(--secondary)]/40 overflow-hidden">
      <div className="px-5 py-4 border-b border-[var(--border)] bg-[var(--secondary)]/30">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div className="w-10 h-10 rounded-lg border border-[var(--border)] bg-[var(--background)] flex items-center justify-center flex-shrink-0">
              <Boxes size={16} className="text-indigo-500" />
            </div>
            <div className="min-w-0 flex-1">
              {/* The composed card's `name` is always the generic gateway id
                  ("AgentEnvGateway"), so it's hidden until the producer names
                  it per-env; the env's identity is the page header above. */}
              <div className="flex items-center gap-2 flex-wrap">
                {card.protocolVersion && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-[var(--background)] border border-[var(--border)] text-[var(--muted-foreground)]">
                    protocol v{card.protocolVersion}
                  </span>
                )}
                {card.preferredTransport && (
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-[var(--background)] border border-[var(--border)] text-[var(--muted-foreground)]">
                    {card.preferredTransport}
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
              </div>
            </div>
          </div>
          <div className="flex items-center gap-4 flex-shrink-0">
            <div className="text-center">
              <div className="text-xl font-semibold text-[var(--foreground)]">
                {extensions.length}
              </div>
              <div className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)]">
                extension{extensions.length !== 1 ? 's' : ''}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xl font-semibold text-[var(--foreground)]">
                {children.length}
              </div>
              <div className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)]">
                child env{children.length !== 1 ? 's' : ''}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="px-5 py-4 border-b border-[var(--border)]">
        <div className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
          Extensions
        </div>
        <ExtensionsBlock extensions={extensions} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 px-5 py-4 text-xs">
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
        {unknownKeys.map(key => (
          <Field key={key} label={key}>
            <code className="text-[10px] font-mono text-[var(--foreground)] break-all">
              {JSON.stringify(card[key])}
            </code>
          </Field>
        ))}
      </div>

      {children.length > 0 && (
        <div className="px-5 py-4 border-t border-[var(--border)]">
          <div className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
            Child environments ({children.length})
          </div>
          <div className="rounded-md border border-[var(--border)] bg-[var(--background)] overflow-hidden max-h-[440px] overflow-y-auto">
            {children.map((child, idx) => (
              <ChildEnvironmentRow
                key={`${child.name ?? 'child'}-${idx}`}
                child={child}
              />
            ))}
          </div>
        </div>
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
