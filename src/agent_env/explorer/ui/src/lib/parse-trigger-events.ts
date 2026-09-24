/** Event-level trigger timeline. Merges three streams on different clocks: env-trigger events from the
 *  aggregation endpoint's state[envId].events (gateway), agent firing-log events from agent_trigger_state
 *  (worker), and conversation turns (Mongo). `seq` is monotonic only within a stream, so the merge sorts on
 *  `ts` and clamps intra-stream regressions. Pure, no React/fetch. */
import { asArray, asRecord, num, str } from './coerce';
import { summarizeWhen } from './parse-triggers';
import type { AuthoredTriggerDetail } from './parse-triggers';

export type TimelineStream = 'env' | 'agent' | 'conversation';

export interface TimelineRow {
  id: string;
  stream: TimelineStream;
  /** Event kind verbatim, plus `message` (conversation) and `gap` (truncation). */
  kind: string;
  seq?: number;
  ts?: string;
  virtualTime?: string;
  envId?: string;
  stepId?: string;
  conversationId?: string;
  triggerId?: string;
  /** Causal chain key: `env:{envId}:{triggerId}` / `agent:{triggerId}`. */
  groupKey?: string;
  actionIndex?: number;
  /** Ordinal of a repeated verify within one action block (the retry loop). */
  attempt?: number;
  turn?: number;
  role?: string;
  summary: string;
  /** Why this trigger fired (`fired` rows only). Absent when the detection that
   *  explains it was evicted from the gateway's head+tail window. */
  reason?: string;
  /** Events evicted between this row and the previous one (`gap` rows only). */
  dropped?: number;
  raw: Record<string, unknown>;
}

export interface TimelineEdge {
  kind: 'co-detection' | 'anchor' | 'cross-engine';
  from: string;
  to: string;
  label: string;
}

/** An armed trigger with a due instant ahead of it. Recurrences re-arm after
 *  every arrival, so this is the only forward-looking signal a live run has. */
export interface PendingTrigger {
  envId: string;
  triggerId: string;
  groupKey: string;
  type?: string;
  status?: string;
  fireCount?: number;
  nextMark: string;
}

export interface TriggerTimeline {
  rows: TimelineRow[];
  edges: TimelineEdge[];
  pending: PendingTrigger[];
  envIds: string[];
  conversationIds: string[];
  /** Envs the endpoint could not read state for — absent events here mean
   *  "unknown", not "none". Only the live path produces these. */
  unavailableEnvs: string[];
  eventsDropped: number;
}

export interface TriggerTimelineInput {
  state?: unknown;
  stateMeta?: unknown;
  stateSources?: unknown;
  agentTriggerState?: unknown;
  conversations?: unknown;
  authored?: Map<string, AuthoredTriggerDetail>;
}

/** `2028-07-11T05:43:08.912294Z` → `2028-07-11 05:43:08Z`. Normalizes to UTC rather than slicing, so a non-`+00:00` offset isn't silently relabeled. */
export function shortStamp(value: string | undefined): string {
  if (!value) return '—';
  const ms = Date.parse(value);
  if (Number.isNaN(ms)) return value;
  return `${new Date(ms).toISOString().slice(0, 19).replace('T', ' ')}Z`;
}

/** Failure kinds all carry a truncated `detail` string; fall back when absent. */
function detailOr(raw: Record<string, unknown>, fallback: string): string {
  return typeof raw.detail === 'string' ? raw.detail : fallback;
}

function changelogNote(raw: Record<string, unknown>): string {
  const before = num(raw.changelog_id_before);
  const after = num(raw.changelog_id_after);
  if (before === undefined || after === undefined) return '';
  return after > before
    ? ` · env changed (changelog ${before}→${after})`
    : ' · no env write';
}

function actionSummary(raw: Record<string, unknown>): string {
  const detail = asRecord(raw.detail);
  if (!detail) return detailOr(raw, 'action ran');
  if (typeof detail.nl === 'string')
    return `instructed executor: “${detail.nl}”`;
  if (typeof detail.tool === 'string') return `called ${detail.tool}`;
  const tools = asArray(detail.tools).filter(t => typeof t === 'string');
  if (detail.action !== undefined) {
    const head = tools.slice(0, 3).join(', ');
    const more = tools.length > 3 ? ` +${tools.length - 3} more` : '';
    return `${str(detail.action) ?? '?'} ${head}${more} for ${
      str(detail.role) ?? '?'
    }`;
  }
  return 'action ran';
}

function envSummary(kind: string, raw: Record<string, unknown>): string {
  const provoking = asRecord(raw.provoking);
  switch (kind) {
    case 'added':
      return 'registered on the gateway';
    case 'removed':
      return 'deregistered';
    case 'detected':
      if (provoking?.source === 'clock') {
        const due = num(raw.due) ?? 0;
        const capped = raw.capped === true ? ' (capped)' : '';
        const window =
          due > 1
            ? ` ${shortStamp(str(raw.first_mark))} → ${shortStamp(
                str(raw.last_mark),
              )}`
            : ` at ${shortStamp(str(raw.first_mark))}`;
        return `clock: ${due} mark${
          due === 1 ? '' : 's'
        } due${window}${capped}`;
      }
      return `condition matched on ${str(provoking?.tool) ?? '?'} (${
        str(provoking?.role) ?? '?'
      })`;
    case 'anchored':
      return `anchored to ${str(raw.anchor) ?? '?'} — due ${shortStamp(
        str(raw.mark),
      )}`;
    case 'reanchored':
      return `clock re-armed (generation ${
        num(raw.generation) ?? '?'
      }) — dropped ${shortStamp(str(raw.dropped_mark))}`;
    case 'action_ok':
      return `${actionSummary(raw)}${changelogNote(raw)}`;
    case 'action_failed':
      return detailOr(raw, 'action failed');
    case 'verify_ok':
      return 'verification passed';
    case 'verify_failed':
      return 'verification failed';
    case 'fired': {
      // The due mark is the firing *reason*, so it rides there, not here.
      const count = num(raw.fire_count);
      return `fired${count !== undefined ? ` (fire #${count})` : ''}`;
    }
    case 'failed':
      return detailOr(
        raw,
        `action #${num(raw.action_index) ?? '?'} rejected — trigger failed`,
      );
    case 'eval_error':
      return detailOr(raw, 'condition evaluation failed');
    default:
      return kind;
  }
}

function agentSummary(kind: string, raw: Record<string, unknown>): string {
  switch (kind) {
    case 'registered': {
      const added = asArray(asRecord(raw.detail)?.added).filter(
        a => typeof a === 'string',
      );
      return `registered ${added.length} agent trigger${
        added.length === 1 ? '' : 's'
      }${added.length ? `: ${added.join(', ')}` : ''}`;
    }
    case 'reset':
      return `fired-set cleared on conversation restart (turn ${
        num(raw.turn) ?? '?'
      })`;
    case 'fired':
      return `fired on turn ${
        num(raw.turn) ?? '?'
      } — injected the next user message`;
    default:
      return kind;
  }
}

function messageText(parts: unknown): string {
  const texts = asArray(parts)
    .map(p => asRecord(p))
    .filter(
      (p): p is Record<string, unknown> =>
        p !== null && typeof p.text === 'string',
    )
    .map(p => String(p.text));
  return texts.join(' ').replace(/\s+/g, ' ').trim();
}

/** The per-trigger entry `state[envId].triggers[]` carries, keyed by groupKey. */
interface TriggerSpec {
  type?: string;
  when?: Record<string, unknown>;
  status?: string;
  fireCount?: number;
  nextMark?: string;
}

function parseEnvRows(state: Record<string, unknown>): {
  rows: TimelineRow[];
  envIds: string[];
  specs: Map<string, TriggerSpec>;
  pending: PendingTrigger[];
  eventsDropped: number;
} {
  const rows: TimelineRow[] = [];
  const envIds: string[] = [];
  const specs = new Map<string, TriggerSpec>();
  const pending: PendingTrigger[] = [];
  let eventsDropped = 0;
  for (const [envId, rawBody] of Object.entries(state)) {
    const body = asRecord(rawBody);
    if (!body) continue;
    envIds.push(envId);
    eventsDropped += num(body.events_dropped) ?? 0;
    for (const rawSpec of asArray(body.triggers)) {
      const spec = asRecord(rawSpec);
      const id = str(spec?.id);
      if (spec && id) {
        const parsed: TriggerSpec = {
          type: str(spec.type),
          when: asRecord(spec.when) ?? undefined,
          status: str(spec.status),
          fireCount: num(spec.fire_count),
          nextMark: str(spec.next_mark),
        };
        specs.set(`env:${envId}:${id}`, parsed);
        // Keyed off the mark, not the status: the gateway advances `next_mark` before running actions and
        // settling `status`, so a mid-`firing` trigger already carries the next instant (gating on "armed" would blank the countdown mid-fire).
        if (parsed.nextMark) {
          pending.push({
            envId,
            triggerId: id,
            groupKey: `env:${envId}:${id}`,
            type: parsed.type,
            status: parsed.status,
            fireCount: parsed.fireCount,
            nextMark: parsed.nextMark,
          });
        }
      }
    }
    // Per (trigger, action_index): a verify that retries emits verify_* twice.
    const attempts = new Map<string, number>();
    let prevSeq: number | undefined;
    for (const rawEvent of asArray(body.events)) {
      const event = asRecord(rawEvent);
      if (!event) continue;
      const kind = str(event.kind) ?? 'unknown';
      const seq = num(event.seq);
      const triggerId = str(event.trigger_id);
      const actionIndex = num(event.action_index);
      if (seq !== undefined && prevSeq !== undefined && seq > prevSeq + 1) {
        rows.push({
          id: `gap:${envId}:${prevSeq}`,
          stream: 'env',
          kind: 'gap',
          seq: prevSeq,
          envId,
          summary: `${seq - prevSeq - 1} events evicted from the gateway log`,
          dropped: seq - prevSeq - 1,
          raw: {},
        });
      }
      if (seq !== undefined) prevSeq = seq;

      let attempt: number | undefined;
      if (triggerId && actionIndex !== undefined) {
        const key = `${triggerId}:${actionIndex}`;
        if (kind.startsWith('verify_')) {
          attempt = (attempts.get(key) ?? 0) + 1;
          attempts.set(key, attempt);
        } else if (kind.startsWith('action_')) {
          // The block closes here, so a recurrence's next pass counts from 1 again.
          attempts.delete(key);
        }
      }
      rows.push({
        id: `env:${envId}:${seq ?? rows.length}`,
        stream: 'env',
        kind,
        seq,
        ts: str(event.ts),
        virtualTime: str(event.virtual_time),
        envId,
        triggerId,
        groupKey: triggerId ? `env:${envId}:${triggerId}` : undefined,
        actionIndex,
        attempt,
        summary: envSummary(kind, event),
        raw: event,
      });
    }
  }
  return { rows, envIds, specs, pending, eventsDropped };
}

function parseAgentRows(
  agentTriggerState: Record<string, unknown>,
): TimelineRow[] {
  const rows: TimelineRow[] = [];
  for (const [stepId, rawLog] of Object.entries(agentTriggerState)) {
    for (const rawEvent of asArray(rawLog)) {
      const event = asRecord(rawEvent);
      if (!event) continue;
      const kind = str(event.kind) ?? 'unknown';
      const seq = num(event.seq);
      const triggerId = str(event.trigger_id);
      rows.push({
        id: `agent:${stepId}:${seq ?? rows.length}`,
        stream: 'agent',
        kind,
        seq,
        ts: str(event.ts),
        stepId,
        triggerId,
        groupKey: triggerId ? `agent:${triggerId}` : undefined,
        turn: num(event.turn),
        summary: agentSummary(kind, event),
        raw: event,
      });
    }
  }
  return rows;
}

function parseConversationRows(conversations: unknown): {
  rows: TimelineRow[];
  conversationIds: string[];
} {
  const rows: TimelineRow[] = [];
  const conversationIds: string[] = [];
  asArray(conversations).forEach((rawConv, convIdx) => {
    const conv = asRecord(rawConv);
    if (!conv) return;
    // Index fallback keeps two id-less conversations in separate streams.
    const convId = str(conv.conversation_id) ?? `conv-${convIdx}`;
    conversationIds.push(convId);
    let turn = 0;
    asArray(conv.messages).forEach((rawMsg, idx) => {
      const msg = asRecord(rawMsg);
      if (!msg) return;
      const role = str(msg.role) ?? 'user';
      // A user message opens a reaction turn; the agent reply stays on it.
      if (role === 'user') turn += 1;
      rows.push({
        id: `conv:${convId}:${idx}`,
        stream: 'conversation',
        kind: 'message',
        seq: idx,
        ts: str(msg.ts),
        conversationId: convId,
        turn,
        role,
        summary: messageText(msg.parts) || '(no text)',
        raw: msg,
      });
    });
  });
  return { rows, conversationIds };
}

const STREAM_RANK: Record<TimelineStream, number> = {
  env: 0,
  agent: 1,
  conversation: 2,
};

/** Sort key per row: `ts` when parseable, else the running max so a timestamp-less row lands after its
 *  predecessor. Each stream is clamped monotonic in `seq`; one bucket per producer, since two conversations run on independent clocks. */
function orderRows(rows: TimelineRow[]): TimelineRow[] {
  const at = new Map<string, number>();
  const byStream = new Map<string, TimelineRow[]>();
  for (const row of rows) {
    const key = `${row.stream}:${
      row.envId ?? row.stepId ?? row.conversationId ?? ''
    }`;
    const list = byStream.get(key);
    if (list) list.push(row);
    else byStream.set(key, [row]);
  }
  for (const list of byStream.values()) {
    let running = 0;
    for (const row of list) {
      const parsed = row.ts ? Date.parse(row.ts) : NaN;
      running = Number.isNaN(parsed) ? running : Math.max(running, parsed);
      at.set(row.id, running);
    }
  }
  return [...rows].sort((a, b) => {
    const delta = (at.get(a.id) ?? 0) - (at.get(b.id) ?? 0);
    if (delta !== 0) return delta;
    const rank = STREAM_RANK[a.stream] - STREAM_RANK[b.stream];
    if (rank !== 0) return rank;
    return (a.seq ?? 0) - (b.seq ?? 0);
  });
}

/** Every `{env_id, trigger_id, status}` an authored agent `when` references. */
function envRefsOf(
  when: unknown,
): { envId: string; triggerId: string; status: string }[] {
  const node = asRecord(when);
  if (!node) return [];
  if (node.type === 'env_trigger') {
    const envId = str(node.env_id);
    const triggerId = str(node.trigger_id);
    return envId && triggerId
      ? [{ envId, triggerId, status: str(node.status) ?? 'fired' }]
      : [];
  }
  return asArray(node.of).flatMap(envRefsOf);
}

type ChainLookup = (
  groupKey: string,
  kind: string,
  before: TimelineRow,
) => TimelineRow | undefined;

/** The most recent row in a chain reaching `kind` before `before`. */
function chainLookup(rows: TimelineRow[]): ChainLookup {
  const order = new Map(rows.map((row, i) => [row.id, i]));
  return (groupKey, kind, before) =>
    rows
      .filter(
        r =>
          r.groupKey === groupKey &&
          r.kind === kind &&
          (order.get(r.id) ?? 0) < (order.get(before.id) ?? 0),
      )
      .pop();
}

/** Why a trigger fired. The `fired` event carries only `fire_count` (+ mark/capped on the time path); the instance-level reason lives on the `detected` that opened the chain. */
function envFiringReason(
  row: TimelineRow,
  spec: TriggerSpec | undefined,
  detected: TimelineRow | undefined,
): string | undefined {
  const mark = str(row.raw.mark);
  if (mark) {
    const capped =
      row.raw.capped === true ? ' (catch-up burst was capped)' : '';
    return `the virtual clock reached ${shortStamp(mark)}${capped}`;
  }
  const provoking = detected && asRecord(detected.raw.provoking);
  const tool = str(provoking?.tool);
  if (!tool) return undefined;
  const role = str(provoking?.role);
  const call = role ? `${tool} (${role})` : tool;
  // Claim a branch only on an affirmatively known type. A state trigger re-evaluates after any non-readonly
  // watched call, so the tool merely prompted the check — claiming "the agent called X" would be an unsupported
  // causal claim. An unknown type degrades to the bare temporal fact.
  if (spec?.type === 'state')
    return `its check matched, re-evaluated after ${call}`;
  if (spec?.type === 'action') return `the agent called ${call}`;
  return `it was detected after ${call}`;
}

/** An agent trigger has no instance data beyond the turn — its authored
 *  condition is the reason. */
function agentFiringReason(
  row: TimelineRow,
  authored: Map<string, AuthoredTriggerDetail> | undefined,
): string | undefined {
  const when = authored?.get(`agent:${row.triggerId}`)?.raw.when;
  if (!when) return undefined;
  const summary = summarizeWhen(when);
  const leaves = summary.leaves.map(l => l.label).join(', ');
  return `its condition was met: ${summary.label}${
    leaves ? ` (${leaves})` : ''
  }`;
}

/** Annotates in place: these rows are built in this module and not yet shared,
 *  so a copy would buy nothing. */
function annotateReasons(
  rows: TimelineRow[],
  specs: Map<string, TriggerSpec>,
  authored: Map<string, AuthoredTriggerDetail> | undefined,
  lastBefore: ChainLookup,
): void {
  for (const row of rows) {
    if (row.kind !== 'fired' || !row.groupKey) continue;
    row.reason =
      row.stream === 'env'
        ? envFiringReason(
            row,
            specs.get(row.groupKey),
            lastBefore(row.groupKey, 'detected', row),
          )
        : agentFiringReason(row, authored);
  }
}

function buildEdges(
  rows: TimelineRow[],
  authored: Map<string, AuthoredTriggerDetail> | undefined,
  lastBefore: ChainLookup,
): TimelineEdge[] {
  const edges: TimelineEdge[] = [];

  // Co-detection: the gateway evaluates every trigger for one tool call in one synchronous pass, so co-detected
  // rows carry consecutive seq (no time window needed). One cursor per env, since seq contiguity is only promised within an env.
  const open = new Map<
    string,
    { head: TimelineRow; tool: string; lastSeq: number }
  >();
  for (const row of rows) {
    if (row.stream !== 'env' || row.kind !== 'detected') continue;
    const envId = row.envId ?? '';
    const tool = str(asRecord(row.raw.provoking)?.tool);
    const run = open.get(envId);
    if (run && tool === run.tool && row.seq === run.lastSeq + 1) {
      edges.push({
        kind: 'co-detection',
        from: run.head.id,
        to: row.id,
        label: `same ${tool} call`,
      });
      run.lastSeq = row.seq;
    } else if (tool && row.seq !== undefined) {
      open.set(envId, { head: row, tool, lastSeq: row.seq });
    } else {
      open.delete(envId);
    }
  }

  // Anchor: an `anchored` row's mark was stamped by its anchor trigger firing.
  for (const row of rows) {
    if (row.kind !== 'anchored') continue;
    const anchor = str(row.raw.anchor);
    const source =
      anchor && lastBefore(`env:${row.envId}:${anchor}`, 'fired', row);
    if (anchor && source) {
      edges.push({
        kind: 'anchor',
        from: source.id,
        to: row.id,
        label: `anchored ${anchor}`,
      });
    }
  }

  // Cross-engine: an agent trigger authored `when: {type: env_trigger, …}`
  // fires off an env trigger reaching that status — exact, from the config.
  for (const row of rows) {
    if (row.stream !== 'agent' || row.kind !== 'fired' || !row.triggerId)
      continue;
    for (const ref of envRefsOf(
      authored?.get(`agent:${row.triggerId}`)?.raw.when,
    )) {
      const source = lastBefore(
        `env:${ref.envId}:${ref.triggerId}`,
        ref.status,
        row,
      );
      if (source) {
        edges.push({
          kind: 'cross-engine',
          from: source.id,
          to: row.id,
          label: `${ref.triggerId} ${ref.status} → ${row.triggerId}`,
        });
      }
    }
  }
  return edges;
}

/** Everything in a `/triggers` payload that changes what the timeline renders — and nothing else. The live
 *  clock reading is EXCLUDED (it changes every poll while events sit still; the header reads it off the payload).
 *  Structural, not a stringify, so a 10k-event log stays cheap. */
export function timelineFingerprint(payload: unknown): string {
  const body = asRecord(payload);
  if (!body) return 'empty';
  const parts = [str(body.status) ?? '', str(body.source) ?? ''];
  for (const [envId, source] of Object.entries(
    asRecord(body.state_sources) ?? {},
  )) {
    parts.push(`${envId}=${str(source) ?? ''}`);
  }
  for (const [envId, rawBody] of Object.entries(asRecord(body.state) ?? {})) {
    const env = asRecord(rawBody);
    if (!env) continue;
    const events = asArray(env.events);
    // Length alone would miss the head+tail window sliding under a capped log,
    // which keeps the count at the cap while the contents move.
    const last = asRecord(events[events.length - 1]);
    parts.push(
      `${envId}:${events.length}:${num(last?.seq) ?? ''}:${
        num(env.events_dropped) ?? 0
      }`,
    );
    for (const rawSpec of asArray(env.triggers)) {
      const spec = asRecord(rawSpec);
      parts.push(
        `${str(spec?.id) ?? ''}/${str(spec?.status) ?? ''}/${
          num(spec?.fire_count) ?? ''
        }/${str(spec?.next_mark) ?? ''}`,
      );
    }
  }
  for (const [stepId, rawLog] of Object.entries(
    asRecord(body.agent_trigger_state) ?? {},
  )) {
    parts.push(`${stepId}:${asArray(rawLog).length}`);
  }
  return parts.join('|');
}

/** Null when the env reported nothing at all — no events, no registered triggers, no unreachable gateway. */
export function parseTriggerTimeline(
  input: TriggerTimelineInput,
): TriggerTimeline | null {
  const env = parseEnvRows(asRecord(input.state) ?? {});
  const agentRows = parseAgentRows(asRecord(input.agentTriggerState) ?? {});
  const unavailableEnvs = Object.entries(asRecord(input.stateSources) ?? {})
    .filter(([, source]) => source === 'unavailable')
    .map(([envId]) => envId);
  const eventRows = [...env.rows.filter(r => r.kind !== 'gap'), ...agentRows];
  // An unreachable gateway means "events unknown" (not "no triggers"), so it still yields a timeline; so does an
  // armed trigger with nothing behind it yet (every live run's first minutes). Test on `specs`, not `pending`:
  // `pending` is keyed off `next_mark`, which only time triggers carry, so gating on it blanks the whole live surface.
  if (
    eventRows.length === 0 &&
    unavailableEnvs.length === 0 &&
    env.pending.length === 0 &&
    env.specs.size === 0
  ) {
    return null;
  }

  const conversations = parseConversationRows(input.conversations);
  const rows = orderRows([...env.rows, ...agentRows, ...conversations.rows]);
  const lastBefore = chainLookup(rows);
  annotateReasons(rows, env.specs, input.authored, lastBefore);
  return {
    rows,
    edges: buildEdges(rows, input.authored, lastBefore),
    pending: env.pending,
    envIds: env.envIds,
    conversationIds: conversations.conversationIds,
    unavailableEnvs,
    eventsDropped: env.eventsDropped,
  };
}

export interface TimelineHighlight {
  /** The selected row's causal chain plus everything an edge connects it to. */
  ids: Set<string>;
  /** Edge labels to show, per row id, for edges wholly inside the selection. */
  labels: Map<string, string[]>;
}

export function highlightFor(
  timeline: TriggerTimeline,
  rowId: string | null,
): TimelineHighlight {
  const ids = new Set<string>();
  const labels = new Map<string, string[]>();
  const row = rowId ? timeline.rows.find(r => r.id === rowId) : undefined;
  if (!row) return { ids, labels };
  ids.add(row.id);
  if (row.groupKey) {
    for (const other of timeline.rows) {
      if (other.groupKey === row.groupKey) ids.add(other.id);
    }
  }
  // One hop out from the chain, deliberately: an edge reached via another edge
  // is not "connected to the selection".
  for (const edge of timeline.edges) {
    if (ids.has(edge.from)) ids.add(edge.to);
    if (ids.has(edge.to)) ids.add(edge.from);
  }
  for (const edge of timeline.edges) {
    if (!ids.has(edge.from) || !ids.has(edge.to)) continue;
    for (const id of [edge.from, edge.to]) {
      labels.set(id, [...(labels.get(id) ?? []), edge.label]);
    }
  }
  return { ids, labels };
}
