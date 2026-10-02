import { Fragment, useEffect, useMemo, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import { Code2, FileJson, Search, Server, Shapes } from 'lucide-react';
import { liveSpecVersionRows } from '../lib/live-spec-versions';
import { apiFetch, BACKEND_URL, FRAMEWORK_DOCS_URL } from './shared';

type JsonSchema = Record<string, unknown>;

interface OpenApiOperation {
  operationId?: string;
  summary?: string;
  description?: string;
  tags?: string[];
  parameters?: Array<Record<string, unknown>>;
  requestBody?: Record<string, unknown>;
  responses?: Record<string, unknown>;
  [key: string]: unknown;
}

interface OpenApiSpec {
  info?: {
    title?: string;
    version?: string;
    description?: string;
  };
  paths?: Record<string, Partial<Record<HttpMethod, OpenApiOperation>>>;
  components?: {
    schemas?: Record<string, JsonSchema>;
  };
  'x-agent-env-docs'?: Record<string, unknown>;
  'x-agent-env-primitives'?: {
    version?: number;
    artifacts?: AgentEnvPrimitive[];
    envs?: AgentEnvPrimitive[];
    taskSteps?: AgentEnvPrimitive[];
  };
}

interface OpenApiMetadata {
  /** 'live' when served straight from the running hub; absent for a published spec. */
  source?: string;
  bucket?: string;
  commit?: string;
  generated_at?: string;
  latest_key?: string;
  metadata_key?: string;
  openapi_version?: string;
  version_key?: string | null;
  versions?: Record<string, string | null | undefined>;
}

interface AgentEnvPrimitive {
  type: string;
  className?: string;
  module?: string;
  component?: string;
  description?: string;
  aliases?: string[];
  source?: {
    module?: string;
    line?: number;
  };
}

type HttpMethod =
  | 'get'
  | 'post'
  | 'put'
  | 'patch'
  | 'delete'
  | 'options'
  | 'head';

type Selection =
  | { kind: 'overview'; id: 'overview' }
  | { kind: 'endpoint'; id: string }
  | { kind: 'primitive'; id: string }
  | { kind: 'schema'; id: string };

interface EndpointDoc {
  id: string;
  method: HttpMethod;
  path: string;
  group: string;
  operation: OpenApiOperation;
}

interface PrimitiveDoc extends AgentEnvPrimitive {
  id: string;
  group: 'artifacts' | 'envs' | 'taskSteps';
}

type PrimitiveGroup = PrimitiveDoc['group'];
type DocsNavSection = 'endpoints' | PrimitiveGroup | 'schemas';

const HTTP_METHODS: HttpMethod[] = [
  'get',
  'post',
  'put',
  'patch',
  'delete',
  'options',
  'head',
];

const METHOD_CLASS: Record<HttpMethod, string> = {
  get: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  post: 'bg-sky-50 text-sky-700 border-sky-200',
  put: 'bg-amber-50 text-amber-700 border-amber-200',
  patch: 'bg-violet-50 text-violet-700 border-violet-200',
  delete: 'bg-rose-50 text-rose-700 border-rose-200',
  options: 'bg-zinc-50 text-zinc-700 border-zinc-200',
  head: 'bg-zinc-50 text-zinc-700 border-zinc-200',
};

const PRIMITIVE_GROUPS: PrimitiveGroup[] = ['artifacts', 'envs', 'taskSteps'];

const PRIMITIVE_LABEL: Record<PrimitiveGroup, string> = {
  artifacts: 'Artifact Types',
  envs: 'Environment Types',
  taskSteps: 'Task Step Types',
};

const ENDPOINT_NAV_LIMIT = 120;
const SCHEMA_NAV_LIMIT = 160;
const DOCS_CATEGORY_LABEL_HEIGHT = 34;
const ENDPOINTS_STICKY_INDEX = 0;
const PRIMITIVES_STICKY_INDEX_OFFSET = ENDPOINTS_STICKY_INDEX + 1;
const SCHEMAS_STICKY_INDEX =
  PRIMITIVES_STICKY_INDEX_OFFSET + PRIMITIVE_GROUPS.length;
const DOCS_CATEGORY_LABEL_COUNT = SCHEMAS_STICKY_INDEX + 1;

const PRIMITIVE_SELECTOR_PREFIX: Record<PrimitiveGroup, string> = {
  artifacts: 'artifact',
  envs: 'environment',
  taskSteps: 'task-step',
};

function docsSpecUrl(): string {
  const base = BACKEND_URL?.replace(/\/$/, '');
  return base ? `${base}/api/v1/docs/openapi` : '/api/v1/docs/openapi';
}

function docsMetadataUrl(): string {
  const base = BACKEND_URL?.replace(/\/$/, '');
  return base
    ? `${base}/api/v1/docs/openapi/metadata`
    : '/api/v1/docs/openapi/metadata';
}

function shortCommit(commit?: string): string {
  return commit ? commit.slice(0, 12) : '';
}

function formatTimestamp(timestamp?: string): string {
  if (!timestamp) return '';
  return timestamp.replace('T', ' ').replace(/\.\d+Z$/, 'Z');
}

function operationTitle(endpoint: EndpointDoc): string {
  return (
    endpoint.operation.summary ||
    endpoint.operation.operationId ||
    `${endpoint.method.toUpperCase()} ${endpoint.path}`
  );
}

function schemaRefName(ref: unknown): string | null {
  if (typeof ref !== 'string') return null;
  const prefix = '#/components/schemas/';
  return ref.startsWith(prefix) ? ref.slice(prefix.length) : ref;
}

function code(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function extractSchemaTitle(schema: JsonSchema, fallback: string): string {
  return typeof schema.title === 'string' && schema.title
    ? schema.title
    : fallback;
}

function normalizeText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value == null) return '';
  return String(value);
}

function collectEndpoints(spec: OpenApiSpec | null): EndpointDoc[] {
  if (!spec?.paths) return [];
  return Object.entries(spec.paths).flatMap(([path, operations]) =>
    HTTP_METHODS.flatMap(method => {
      const operation = operations[method];
      if (!operation) return [];
      const id = operation.operationId || `${method}:${path}`;
      const group =
        normalizeText(operation['x-agent-env-docs-group']) ||
        operation.tags?.[0] ||
        path.split('/')[3] ||
        'api';
      return [{ id, method, path, group, operation }];
    }),
  );
}

function collectPrimitives(spec: OpenApiSpec | null): PrimitiveDoc[] {
  const primitives = spec?.['x-agent-env-primitives'];
  if (!primitives) return [];
  return PRIMITIVE_GROUPS.flatMap(group =>
    (primitives[group] ?? []).map(item => ({
      ...item,
      id: `${group}:${item.type}`,
      group,
    })),
  );
}

function encodeSelectorPart(value: string): string {
  return encodeURIComponent(value);
}

function decodeSelectorPart(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function primitiveTypeFromId(id: string): string {
  const separatorIndex = id.indexOf(':');
  return separatorIndex === -1 ? id : id.slice(separatorIndex + 1);
}

function endpointSelector(endpoint: EndpointDoc): string {
  return `endpoint-${encodeSelectorPart(endpoint.id)}`;
}

function primitiveSelector(primitive: PrimitiveDoc): string {
  return `${PRIMITIVE_SELECTOR_PREFIX[primitive.group]}-${encodeSelectorPart(
    primitive.type,
  )}`;
}

function schemaSelector(name: string): string {
  return `schema-${encodeSelectorPart(name)}`;
}

function navSectionSelector(section: DocsNavSection): string {
  return `docs-section-${section}`;
}

function selectionSelector(selection: Selection): string {
  if (selection.kind === 'overview') return 'overview';
  if (selection.kind === 'endpoint') {
    return `endpoint-${encodeSelectorPart(selection.id)}`;
  }
  if (selection.kind === 'schema') {
    return schemaSelector(selection.id);
  }

  const group = selection.id.split(':', 1)[0] as PrimitiveGroup;
  const type = primitiveTypeFromId(selection.id);
  return `${PRIMITIVE_SELECTOR_PREFIX[group]}-${encodeSelectorPart(type)}`;
}

function selectionFromSelector({
  endpoints,
  hash,
  primitives,
  schemas,
}: {
  endpoints: EndpointDoc[];
  hash: string;
  primitives: PrimitiveDoc[];
  schemas: Record<string, JsonSchema>;
}): Selection | null {
  const selector = hash.replace(/^#/, '').trim();
  if (!selector) return null;
  if (selector === 'overview') return { kind: 'overview', id: 'overview' };

  if (selector.startsWith('endpoint-')) {
    const id = decodeSelectorPart(selector.slice('endpoint-'.length));
    return endpoints.some(endpoint => endpoint.id === id)
      ? { kind: 'endpoint', id }
      : null;
  }

  for (const group of PRIMITIVE_GROUPS) {
    const prefix = `${PRIMITIVE_SELECTOR_PREFIX[group]}-`;
    if (!selector.startsWith(prefix)) continue;

    const type = decodeSelectorPart(selector.slice(prefix.length));
    const primitive = primitives.find(
      item => item.group === group && item.type === type,
    );
    return primitive ? { kind: 'primitive', id: primitive.id } : null;
  }

  if (selector.startsWith('schema-')) {
    const id = decodeSelectorPart(selector.slice('schema-'.length));
    return schemas[id] ? { kind: 'schema', id } : null;
  }

  return null;
}

function matchQuery(text: string, query: string): boolean {
  return text.toLowerCase().includes(query.trim().toLowerCase());
}

function MethodBadge({ method }: { method: HttpMethod }) {
  return (
    <span
      className={`inline-flex h-5 min-w-[44px] items-center justify-center rounded border px-1.5 text-[11px] font-semibold ${METHOD_CLASS[method]}`}
    >
      {method.toUpperCase()}
    </span>
  );
}

function SectionLabel({
  icon: Icon,
  label,
  count,
  onClick,
  stickyIndex,
}: {
  icon: typeof Server;
  label: string;
  count: number;
  onClick: () => void;
  stickyIndex: number;
}) {
  const stickyStyle: CSSProperties = {
    bottom:
      (DOCS_CATEGORY_LABEL_COUNT - stickyIndex - 1) *
      DOCS_CATEGORY_LABEL_HEIGHT,
    top: stickyIndex * DOCS_CATEGORY_LABEL_HEIGHT,
    zIndex: 20 + stickyIndex,
  };

  return (
    <button
      type="button"
      aria-label={`Jump to ${label} section`}
      onClick={onClick}
      className="sticky -mx-2 flex h-[34px] w-[calc(100%+1rem)] cursor-pointer appearance-none items-center gap-2 border-y border-[var(--border)] bg-[var(--background)] px-5 text-left text-xs font-semibold text-[var(--muted-foreground)] shadow-sm transition-colors hover:bg-[var(--accent)] hover:text-[var(--foreground)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)]"
      style={stickyStyle}
    >
      <Icon size={13} />
      <span>{label}</span>
      <span className="ml-auto tabular-nums">{count}</span>
    </button>
  );
}

function SectionAnchor({ section }: { section: DocsNavSection }) {
  return (
    <div
      id={navSectionSelector(section)}
      className="h-0"
      data-docs-sidebar-section={section}
    />
  );
}

function OverflowHint({ shown, total }: { shown: number; total: number }) {
  if (total <= shown) return null;
  return (
    <div className="px-3 py-2 text-xs text-[var(--muted-foreground)]">
      Showing {shown} of {total}. Refine search for more.
    </div>
  );
}

function NavButton({
  active,
  children,
  onClick,
  selector,
}: {
  active: boolean;
  children: ReactNode;
  onClick: () => void;
  selector?: string;
}) {
  return (
    <button
      id={selector}
      onClick={onClick}
      data-docs-selector={selector}
      className={`w-full rounded-md px-3 py-2 text-left text-sm transition-colors ${
        active
          ? 'bg-[var(--secondary)] text-[var(--foreground)] font-medium'
          : 'text-[var(--muted-foreground)] hover:bg-[var(--accent)] hover:text-[var(--foreground)]'
      }`}
    >
      {children}
    </button>
  );
}

function JsonBlock({ value }: { value: unknown }) {
  return (
    <pre className="max-h-[360px] overflow-auto rounded-md border border-[var(--border)] bg-[var(--secondary)] p-3 text-xs leading-5 text-[var(--foreground)]">
      {code(value)}
    </pre>
  );
}

function OverviewPane({
  spec,
  metadata,
  endpoints,
  primitives,
}: {
  spec: OpenApiSpec;
  metadata: OpenApiMetadata | null;
  endpoints: EndpointDoc[];
  primitives: PrimitiveDoc[];
}) {
  const schemas = spec.components?.schemas ?? {};
  const docs = spec['x-agent-env-docs'] ?? {};
  const primitiveCounts = PRIMITIVE_GROUPS.map(group => ({
    label: PRIMITIVE_LABEL[group],
    count: primitives.filter(item => item.group === group).length,
  }));

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-[var(--foreground)]">
          {spec.info?.title ?? 'AgentEnv Explorer API'}
        </h1>
        <div className="mt-2 text-sm text-[var(--muted-foreground)]">
          Version {spec.info?.version ?? 'unknown'}
          {metadata?.commit && (
            <>
              {' '}
              · Spec commit{' '}
              <code className="text-[var(--foreground)]">
                {shortCommit(metadata.commit)}
              </code>
            </>
          )}
        </div>
        {spec.info?.description && (
          <p className="mt-4 max-w-3xl text-sm leading-6 text-[var(--foreground)]">
            {spec.info.description}
          </p>
        )}
        <p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--muted-foreground)]">
          This page is the explorer&apos;s API reference. For guides to
          environments, artifacts, agents, tasks and plugins, see the{' '}
          <a
            href={FRAMEWORK_DOCS_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--foreground)] underline underline-offset-2"
          >
            AgentEnv Framework docs
          </a>
          .
        </p>
      </div>

      <div className="grid max-w-4xl grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Endpoints" value={endpoints.length} />
        <Stat label="Schemas" value={Object.keys(schemas).length} />
        {primitiveCounts.map(item => (
          <Stat key={item.label} label={item.label} value={item.count} />
        ))}
      </div>

      <div>
        <h2 className="mb-3 text-sm font-semibold text-[var(--foreground)]">
          {metadata?.source === 'live' ? 'This Hub' : 'Published Spec'}
        </h2>
        <DetailGrid
          rows={
            metadata?.source === 'live'
              ? [
                  // A live spec has no object-store provenance; showing blank
                  // Bucket / key rows would imply it failed to load one.
                  ['Source', 'Served live by this hub'],
                  ['Generated', formatTimestamp(metadata?.generated_at)],
                  ['OpenAPI', metadata?.openapi_version ?? ''],
                  ...liveSpecVersionRows(metadata?.versions),
                ]
              : [
                  ['Commit', metadata?.commit ?? ''],
                  ['Generated', formatTimestamp(metadata?.generated_at)],
                  ['Bucket', metadata?.bucket ?? ''],
                  ['Latest key', metadata?.latest_key ?? ''],
                  ['Version key', metadata?.version_key ?? ''],
                  ['OpenAPI', metadata?.openapi_version ?? ''],
                  ['agent-env', metadata?.versions?.['agent-env'] ?? ''],
                ]
          }
        />
      </div>

      <div>
        <h2 className="mb-3 text-sm font-semibold text-[var(--foreground)]">
          OpenAPI Extensions
        </h2>
        <JsonBlock value={docs} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-[var(--border)] px-4 py-3">
      <div className="text-xl font-semibold tabular-nums">{value}</div>
      <div className="mt-1 text-xs text-[var(--muted-foreground)]">{label}</div>
    </div>
  );
}

function EndpointPane({ endpoint }: { endpoint: EndpointDoc }) {
  const parameters = endpoint.operation.parameters ?? [];
  const responses = endpoint.operation.responses ?? {};

  return (
    <div className="space-y-8">
      <div>
        <div className="flex flex-wrap items-center gap-3">
          <MethodBadge method={endpoint.method} />
          <code className="rounded bg-[var(--secondary)] px-2 py-1 text-sm">
            {endpoint.path}
          </code>
        </div>
        <h1 className="mt-4 text-2xl font-semibold text-[var(--foreground)]">
          {operationTitle(endpoint)}
        </h1>
        {endpoint.operation.description && (
          <p className="mt-3 max-w-3xl text-sm leading-6 text-[var(--foreground)]">
            {endpoint.operation.description}
          </p>
        )}
      </div>

      <DetailGrid
        rows={[
          ['Group', endpoint.group],
          ['Operation ID', endpoint.operation.operationId ?? ''],
          [
            'API Version',
            normalizeText(endpoint.operation['x-agent-env-api-version']),
          ],
        ]}
      />

      {parameters.length > 0 && (
        <div>
          <h2 className="mb-3 text-sm font-semibold">Parameters</h2>
          <div className="overflow-hidden rounded-md border border-[var(--border)]">
            {parameters.map((parameter, index) => (
              <div
                key={`${normalizeText(parameter.name)}-${index}`}
                className="grid grid-cols-[160px_100px_1fr] gap-4 border-b border-[var(--border)] px-4 py-3 text-sm last:border-b-0"
              >
                <code>{normalizeText(parameter.name)}</code>
                <span className="text-[var(--muted-foreground)]">
                  {normalizeText(parameter.in)}
                </span>
                <span>{normalizeText(parameter.description)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {endpoint.operation.requestBody && (
        <div>
          <h2 className="mb-3 text-sm font-semibold">Request Body</h2>
          <JsonBlock value={endpoint.operation.requestBody} />
        </div>
      )}

      <div>
        <h2 className="mb-3 text-sm font-semibold">Responses</h2>
        <JsonBlock value={responses} />
      </div>
    </div>
  );
}

function PrimitivePane({
  primitive,
  schemas,
}: {
  primitive: PrimitiveDoc;
  schemas: Record<string, JsonSchema>;
}) {
  const schemaName = schemaRefName(primitive.component);
  const schema = schemaName ? schemas[schemaName] : undefined;

  return (
    <div className="space-y-8">
      <div>
        <div className="text-sm text-[var(--muted-foreground)]">
          {PRIMITIVE_LABEL[primitive.group]}
        </div>
        <h1 className="mt-2 text-2xl font-semibold">{primitive.type}</h1>
        {primitive.description && (
          <p className="mt-3 max-w-3xl text-sm leading-6">
            {primitive.description}
          </p>
        )}
      </div>

      <DetailGrid
        rows={[
          ['Class', primitive.className ?? ''],
          ['Module', primitive.module ?? ''],
          ['Component', primitive.component ?? ''],
          ['Aliases', primitive.aliases?.join(', ') ?? ''],
          [
            'Source',
            primitive.source?.line
              ? `${primitive.source.module}:${primitive.source.line}`
              : primitive.source?.module ?? '',
          ],
        ]}
      />

      {schema && (
        <div>
          <h2 className="mb-3 text-sm font-semibold">Schema</h2>
          <JsonBlock value={schema} />
        </div>
      )}
    </div>
  );
}

function SchemaPane({ name, schema }: { name: string; schema: JsonSchema }) {
  return (
    <div className="space-y-8">
      <div>
        <div className="text-sm text-[var(--muted-foreground)]">Schema</div>
        <h1 className="mt-2 text-2xl font-semibold">
          {extractSchemaTitle(schema, name)}
        </h1>
        {typeof schema.description === 'string' && (
          <p className="mt-3 max-w-3xl text-sm leading-6">
            {schema.description}
          </p>
        )}
      </div>
      <JsonBlock value={schema} />
    </div>
  );
}

function DetailGrid({ rows }: { rows: Array<[string, string]> }) {
  return (
    <div className="max-w-4xl overflow-hidden rounded-md border border-[var(--border)]">
      {rows
        .filter(([, value]) => value)
        .map(([label, value]) => (
          <div
            key={label}
            className="grid grid-cols-[160px_1fr] border-b border-[var(--border)] px-4 py-3 text-sm last:border-b-0"
          >
            <div className="text-[var(--muted-foreground)]">{label}</div>
            <code className="break-all text-[var(--foreground)]">{value}</code>
          </div>
        ))}
    </div>
  );
}

export function DocsPage() {
  const [spec, setSpec] = useState<OpenApiSpec | null>(null);
  const [metadata, setMetadata] = useState<OpenApiMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [selection, setSelection] = useState<Selection>({
    kind: 'overview',
    id: 'overview',
  });

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const loadJson = async <T,>(url: string, label: string): Promise<T> => {
      const response = await apiFetch(url);
      if (!response.ok)
        throw new Error(`${label} request failed: ${response.status}`);
      return (await response.json()) as T;
    };

    Promise.all([
      loadJson<OpenApiSpec>(docsSpecUrl(), 'OpenAPI'),
      loadJson<OpenApiMetadata>(docsMetadataUrl(), 'OpenAPI metadata'),
    ])
      .then(([specData, metadataData]) => {
        if (cancelled) return;
        setSpec(specData);
        setMetadata(metadataData);
        setError(null);
      })
      .catch(err => {
        if (cancelled) return;
        setError(
          err instanceof Error ? err.message : 'Unable to load OpenAPI spec',
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const endpoints = useMemo(() => collectEndpoints(spec), [spec]);
  const primitives = useMemo(() => collectPrimitives(spec), [spec]);
  const schemas = useMemo(() => spec?.components?.schemas ?? {}, [spec]);
  const schemaEntries = useMemo(
    () => Object.entries(schemas).sort(([a], [b]) => a.localeCompare(b)),
    [schemas],
  );

  const filteredEndpoints = useMemo(
    () =>
      endpoints.filter(endpoint =>
        matchQuery(
          `${endpoint.method} ${endpoint.path} ${operationTitle(endpoint)} ${
            endpoint.group
          }`,
          query,
        ),
      ),
    [endpoints, query],
  );
  const filteredPrimitives = useMemo(
    () =>
      primitives.filter(primitive =>
        matchQuery(
          `${primitive.group} ${primitive.type} ${primitive.className ?? ''} ${
            primitive.module ?? ''
          }`,
          query,
        ),
      ),
    [primitives, query],
  );
  const primitiveNavSections = useMemo(
    () =>
      PRIMITIVE_GROUPS.map((group, index) => ({
        group,
        label: PRIMITIVE_LABEL[group],
        items: filteredPrimitives.filter(
          primitive => primitive.group === group,
        ),
        stickyIndex: PRIMITIVES_STICKY_INDEX_OFFSET + index,
      })),
    [filteredPrimitives],
  );
  const filteredSchemas = useMemo(
    () =>
      schemaEntries.filter(([name, schema]) =>
        matchQuery(`${name} ${extractSchemaTitle(schema, name)}`, query),
      ),
    [schemaEntries, query],
  );
  const visibleEndpoints = useMemo(
    () => filteredEndpoints.slice(0, ENDPOINT_NAV_LIMIT),
    [filteredEndpoints],
  );
  const visibleSchemas = useMemo(
    () => filteredSchemas.slice(0, SCHEMA_NAV_LIMIT),
    [filteredSchemas],
  );

  const selectedEndpoint =
    selection.kind === 'endpoint'
      ? endpoints.find(endpoint => endpoint.id === selection.id)
      : undefined;
  const selectedPrimitive =
    selection.kind === 'primitive'
      ? primitives.find(primitive => primitive.id === selection.id)
      : undefined;
  const selectedSchema =
    selection.kind === 'schema' ? schemas[selection.id] : undefined;

  useEffect(() => {
    if (!spec || typeof window === 'undefined') return;

    const applyHashSelection = () => {
      const nextSelection = selectionFromSelector({
        endpoints,
        hash: window.location.hash,
        primitives,
        schemas,
      });
      if (nextSelection) setSelection(nextSelection);
    };

    applyHashSelection();
    window.addEventListener('hashchange', applyHashSelection);
    return () => window.removeEventListener('hashchange', applyHashSelection);
  }, [endpoints, primitives, schemas, spec]);

  useEffect(() => {
    if (typeof document === 'undefined') return;

    document.getElementById(selectionSelector(selection))?.scrollIntoView({
      block: 'nearest',
    });
    document
      .querySelector<HTMLElement>('[data-docs-content]')
      ?.scrollTo({ top: 0 });
  }, [selection]);

  const selectDocsItem = (nextSelection: Selection) => {
    setSelection(nextSelection);
    if (typeof window === 'undefined') return;

    const selector = selectionSelector(nextSelection);
    if (window.location.hash.slice(1) === selector) return;
    window.history.pushState(
      null,
      '',
      `${window.location.pathname}${window.location.search}#${selector}`,
    );
  };

  const scrollNavSectionIntoView = (
    section: DocsNavSection,
    stickyIndex: number,
  ) => {
    if (typeof document === 'undefined') return;

    const anchor = document.getElementById(navSectionSelector(section));
    const scroller = anchor?.closest<HTMLElement>('[data-docs-sidebar-scroll]');
    if (!anchor || !scroller) return;

    const anchorRect = anchor.getBoundingClientRect();
    const scrollerRect = scroller.getBoundingClientRect();
    const stickyOffset = stickyIndex * DOCS_CATEGORY_LABEL_HEIGHT;
    const nextTop =
      scroller.scrollTop + anchorRect.top - scrollerRect.top - stickyOffset;

    scroller.scrollTo({ top: Math.max(0, nextTop), behavior: 'smooth' });
  };

  return (
    <div className="flex h-full min-h-0">
      <aside className="flex w-[320px] flex-shrink-0 flex-col border-r border-[var(--border)]">
        <div className="border-b border-[var(--border)] p-4">
          <div className="flex items-center gap-2 text-base font-semibold">
            <FileJson size={18} />
            Docs
          </div>
          {metadata?.commit && (
            <div className="mt-1 text-xs text-[var(--muted-foreground)]">
              Spec {shortCommit(metadata.commit)}
            </div>
          )}
          <div className="relative mt-4">
            <Search
              size={15}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted-foreground)]"
            />
            <input
              value={query}
              onChange={event => setQuery(event.target.value)}
              placeholder="Search docs"
              className="h-9 w-full rounded-md border border-[var(--border)] bg-transparent pl-9 pr-3 text-sm outline-none focus:border-[var(--ring)]"
            />
          </div>
        </div>

        <div
          className="min-h-0 flex-1 overflow-auto px-2"
          data-docs-sidebar-scroll
        >
          <div className="pt-2">
            <NavButton
              active={selection.kind === 'overview'}
              selector="overview"
              onClick={() =>
                selectDocsItem({ kind: 'overview', id: 'overview' })
              }
            >
              Overview
            </NavButton>
          </div>

          <SectionAnchor section="endpoints" />
          <SectionLabel
            icon={Server}
            label="Endpoints"
            count={filteredEndpoints.length}
            onClick={() =>
              scrollNavSectionIntoView('endpoints', ENDPOINTS_STICKY_INDEX)
            }
            stickyIndex={ENDPOINTS_STICKY_INDEX}
          />
          <div className="space-y-1">
            {visibleEndpoints.map(endpoint => (
              <NavButton
                key={endpoint.id}
                selector={endpointSelector(endpoint)}
                active={
                  selection.kind === 'endpoint' && selection.id === endpoint.id
                }
                onClick={() =>
                  selectDocsItem({ kind: 'endpoint', id: endpoint.id })
                }
              >
                <div className="flex min-w-0 items-center gap-2">
                  <MethodBadge method={endpoint.method} />
                  <span className="truncate">{operationTitle(endpoint)}</span>
                </div>
              </NavButton>
            ))}
            <OverflowHint
              shown={visibleEndpoints.length}
              total={filteredEndpoints.length}
            />
          </div>

          {primitiveNavSections.map(section => (
            <Fragment key={section.group}>
              <SectionAnchor section={section.group} />
              <SectionLabel
                icon={Shapes}
                label={section.label}
                count={section.items.length}
                onClick={() =>
                  scrollNavSectionIntoView(section.group, section.stickyIndex)
                }
                stickyIndex={section.stickyIndex}
              />
              <div className="space-y-1">
                {section.items.map(primitive => (
                  <NavButton
                    key={primitive.id}
                    selector={primitiveSelector(primitive)}
                    active={
                      selection.kind === 'primitive' &&
                      selection.id === primitive.id
                    }
                    onClick={() =>
                      selectDocsItem({ kind: 'primitive', id: primitive.id })
                    }
                  >
                    <div className="truncate">{primitive.type}</div>
                  </NavButton>
                ))}
              </div>
            </Fragment>
          ))}

          <SectionAnchor section="schemas" />
          <SectionLabel
            icon={Code2}
            label="Schemas"
            count={filteredSchemas.length}
            onClick={() =>
              scrollNavSectionIntoView('schemas', SCHEMAS_STICKY_INDEX)
            }
            stickyIndex={SCHEMAS_STICKY_INDEX}
          />
          <div className="space-y-1">
            {visibleSchemas.map(([name, schema]) => (
              <NavButton
                key={name}
                selector={schemaSelector(name)}
                active={selection.kind === 'schema' && selection.id === name}
                onClick={() => selectDocsItem({ kind: 'schema', id: name })}
              >
                <div className="truncate">
                  {extractSchemaTitle(schema, name)}
                </div>
              </NavButton>
            ))}
            <OverflowHint
              shown={visibleSchemas.length}
              total={filteredSchemas.length}
            />
          </div>
        </div>
      </aside>

      <section className="min-w-0 flex-1 overflow-auto p-8" data-docs-content>
        {loading && (
          <div className="text-sm text-[var(--muted-foreground)]">
            Loading docs...
          </div>
        )}
        {error && (
          <div className="rounded-md border border-[var(--border)] p-4 text-sm">
            {error}
          </div>
        )}
        {spec && selection.kind === 'overview' && (
          <OverviewPane
            spec={spec}
            metadata={metadata}
            endpoints={endpoints}
            primitives={primitives}
          />
        )}
        {selectedEndpoint && <EndpointPane endpoint={selectedEndpoint} />}
        {selectedPrimitive && (
          <PrimitivePane primitive={selectedPrimitive} schemas={schemas} />
        )}
        {selection.kind === 'schema' && selectedSchema && (
          <SchemaPane name={selection.id} schema={selectedSchema} />
        )}
      </section>
    </div>
  );
}
