/** Authored-trigger graph model for the task detail page. Parses register_env_triggers /
 *  register_agent_triggers out of a task's steps into a left-to-right bipartite graph (one node per trigger,
 *  an anchor pill per group, a cross-layer edge per env_trigger condition). Dagre layout lives here so the
 *  smoke test can assert env-left-of-agent lane ordering. Unknown when/action types pass through raw. */
import dagre from '@dagrejs/dagre';

export interface EnvRef {
  envId: string;
  triggerId: string;
  status: string;
}

export interface WhenLeaf {
  kind: string;
  label: string;
}

export interface WhenSummary {
  kind: string;
  label: string;
  envRefs: EnvRef[];
  /** Ordered leaf summaries for composite (`all`/`any`) conditions. */
  leaves: WhenLeaf[];
}

export interface ActionChip {
  kind: string;
  label: string;
  hasVerify: boolean;
}

export interface TriggerNodeData {
  kind: 'env' | 'agent';
  triggerId: string;
  groupKey: string;
  when: WhenSummary;
  actions: ActionChip[];
  isSensor: boolean;
  notify?: string;
  /** Agent trigger ids whose conditions reference this env trigger. */
  referencedBy: string[];
  raw: Record<string, unknown>;
}

export interface AnchorNodeData {
  kind: 'env' | 'agent';
  label: string;
  triggerCount: number;
  /** The full registration step dict, for the group detail panel. */
  raw: Record<string, unknown>;
}

export interface TriggerGraphNode {
  id: string;
  type: 'trigger' | 'anchor';
  x: number;
  y: number;
  width: number;
  height: number;
  trigger?: TriggerNodeData;
  anchor?: AnchorNodeData;
}

export interface TriggerGraphEdge {
  id: string;
  source: string;
  target: string;
  kind: 'group' | 'env-ref';
}

export interface TriggerGroup {
  kind: 'env' | 'agent';
  key: string;
  stepId: string;
  watchRoles?: string[];
  executorAgentName?: string;
  executorTimeoutSeconds?: number;
}

export interface TriggerGraph {
  nodes: TriggerGraphNode[];
  edges: TriggerGraphEdge[];
  groups: TriggerGroup[];
  triggerCount: number;
  height: number;
}

export const TRIGGER_NODE_W = 240;
const ROW_BASE_H = 78;
const LEAF_ROW_H = 16;
const ANCHOR_H = 30;
const RANK_SEP = 90;
const NODE_SEP = 28;

const CMP_SYMBOLS: Record<string, string> = {
  gte: '≥',
  gt: '>',
  lte: '≤',
  lt: '<',
  eq: '=',
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

const _DURATION_RE =
  /^P(?:(\d+)W)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$/;

/** `PT24H` → `24h`, `P1DT2H` → `1d 2h` (weeks/days/hours/minutes/seconds, no calendar Y/M). Non-duration strings pass through raw. */
export function humanizeDuration(value: string): string {
  const m = _DURATION_RE.exec(value);
  if (!m) return value;
  const units = ['w', 'd', 'h', 'm', 's'];
  const parts = units
    .map((u, i) => (m[i + 1] != null ? `${m[i + 1]}${u}` : null))
    .filter((p): p is string => p !== null);
  return parts.length > 0 ? parts.join(' ') : value;
}

/** Mark values are either absolute RFC3339 stamps or t0-relative durations. */
function markLabel(value: unknown): string {
  const s = String(value ?? '?');
  return s.startsWith('P') ? `t0+${humanizeDuration(s)}` : s;
}

function timeWhenLabel(w: Record<string, unknown>): string {
  const parts: string[] = [];
  if (typeof w.after === 'string') {
    parts.push(
      `after ${w.after} +${humanizeDuration(String(w.offset ?? '?'))}`,
    );
  } else if (w.at != null) {
    parts.push(`at ${markLabel(w.at)}`);
  }
  if (w.every != null) {
    const every = asRecord(w.every);
    let recur = every
      ? `every ~exp(${humanizeDuration(String(every.mean ?? '?'))})`
      : `every ${humanizeDuration(String(w.every))}`;
    if (typeof w.count === 'number') recur += ` ×${w.count}`;
    if (w.until != null) recur += ` until ${markLabel(w.until)}`;
    parts.push(recur);
  }
  return parts.length > 0 ? parts.join(', ') : 'time';
}

export function summarizeWhen(when: unknown): WhenSummary {
  const w = asRecord(when);
  if (!w) return { kind: 'unknown', label: 'unknown', envRefs: [], leaves: [] };
  const kind = String(w.type ?? 'unknown');

  if (kind === 'all' || kind === 'any') {
    const children = Array.isArray(w.of) ? w.of.map(summarizeWhen) : [];
    return {
      kind,
      label: `${kind.toUpperCase()} of ${children.length}`,
      envRefs: children.flatMap(c => c.envRefs),
      leaves: children.map(c => ({ kind: c.kind, label: c.label })),
    };
  }

  if (kind === 'env_trigger') {
    const ref: EnvRef = {
      envId: String(w.env_id ?? ''),
      triggerId: String(w.trigger_id ?? ''),
      status: String(w.status ?? 'fired'),
    };
    return {
      kind,
      label: `${ref.triggerId} ${ref.status}`,
      envRefs: [ref],
      leaves: [],
    };
  }

  let label: string;
  switch (kind) {
    case 'action':
      label = String(w.tool ?? '?');
      break;
    case 'state': {
      const steps = asRecord(w.check)?.steps;
      label = Array.isArray(steps) ? `${steps.length}-step check` : 'check';
      break;
    }
    case 'step':
      label = `turn ${
        CMP_SYMBOLS[String(w.cmp)] ?? String(w.cmp ?? '≥')
      } ${String(w.turn ?? '?')}`;
      break;
    case 'time':
      label = timeWhenLabel(w);
      break;
    case 'conversational':
      label = 'conversational';
      break;
    default:
      // Unknown kind (future barrier / …): raw type string as the label;
      // still descend `of` if present so env refs survive.
      label = kind;
  }
  const nestedRefs = Array.isArray(w.of)
    ? w.of.map(summarizeWhen).flatMap(c => c.envRefs)
    : [];
  return { kind, label, envRefs: nestedRefs, leaves: [] };
}

export function summarizeActions(actions: unknown): ActionChip[] {
  if (!Array.isArray(actions)) return [];
  return actions.map(a => {
    const act = asRecord(a);
    if (!act) return { kind: 'unknown', label: 'unknown', hasVerify: false };
    const kind = String(act.type ?? 'unknown');
    let label: string;
    switch (kind) {
      case 'permission': {
        const action = String(act.action ?? 'permission');
        if (act.tools === '*') {
          label = `${action} all tools`;
        } else {
          const tools = Array.isArray(act.tools) ? act.tools.length : 0;
          label = `${action} ${tools} tool${tools === 1 ? '' : 's'}`;
        }
        break;
      }
      case 'nl':
        label = 'nl';
        break;
      case 'say':
        label = 'say';
        break;
      case 'end':
        label = 'end';
        break;
      case 'tool':
        label = String(act.tool ?? 'tool');
        break;
      default:
        label = kind;
    }
    return { kind, label, hasVerify: Boolean(act.verify) };
  });
}

// Rough greedy estimate of wrapped action-chip rows, so wrapping grows the card instead of clipping.
const CHIP_CONTENT_W = TRIGGER_NODE_W - 50;
const CHIP_ROW_H = 22;

function estimateActionRows(actions: ActionChip[]): number {
  let rows = 1;
  let x = 0;
  for (const a of actions) {
    const w = Math.min(a.label.length, 18) * 6 + 34;
    if (x > 0 && x + w > CHIP_CONTENT_W) {
      rows += 1;
      x = w;
    } else {
      x += w + 4;
    }
  }
  return rows;
}

function triggerNodeHeight(when: WhenSummary, actions: ActionChip[]): number {
  return (
    ROW_BASE_H +
    when.leaves.length * LEAF_ROW_H +
    (estimateActionRows(actions) - 1) * CHIP_ROW_H
  );
}

function anchorWidth(label: string): number {
  return Math.min(220, Math.max(96, Math.round(label.length * 6.5) + 28));
}

export interface AuthoredTriggerDetail {
  kind: 'env' | 'agent';
  triggerId: string;
  groupKey: string;
  /** The full authored trigger dict (when/actions incl. say/nl/tools/verify, notify), rendered via ConfigDetail. */
  raw: Record<string, unknown>;
}

/** Per-trigger authored-config index for the badge popovers. Keys env:{envId}:{triggerId} / agent:{triggerId}
 *  (agent firings don't carry the name, so agent triggers key on id alone). Slim step refs yield an empty index. */
export function indexAuthoredTriggers(
  steps: unknown[] | undefined,
): Map<string, AuthoredTriggerDetail> {
  const out = new Map<string, AuthoredTriggerDetail>();
  for (const s of steps ?? []) {
    const step = asRecord(s);
    if (!step) continue;
    const kind =
      step.type === 'register_env_triggers'
        ? ('env' as const)
        : step.type === 'register_agent_triggers'
        ? ('agent' as const)
        : null;
    if (!kind) continue;
    const groupKey = String(
      (kind === 'env' ? step.env_id : step.agent_name) ?? kind,
    );
    const triggers = Array.isArray(step.triggers) ? step.triggers : [];
    for (const t of triggers) {
      const trig = asRecord(t);
      if (!trig) continue;
      const triggerId = String(trig.id ?? '');
      const key =
        kind === 'env' ? `env:${groupKey}:${triggerId}` : `agent:${triggerId}`;
      if (out.has(key)) continue;
      out.set(key, { kind, triggerId, groupKey, raw: trig });
    }
  }
  return out;
}

export function parseTriggerGraph(
  steps: Array<Record<string, unknown>>,
): TriggerGraph | null {
  const envSteps = steps.filter(s => s.type === 'register_env_triggers');
  const agentSteps = steps.filter(s => s.type === 'register_agent_triggers');
  if (envSteps.length === 0 && agentSteps.length === 0) return null;

  const groups: TriggerGroup[] = [];
  const nodes: TriggerGraphNode[] = [];
  const edges: TriggerGraphEdge[] = [];
  const envNodeIds = new Map<string, string>(); // `${envId}:${triggerId}` → node id
  const usedIds = new Set<string>();

  const uniqueId = (base: string): string => {
    let id = base;
    let n = 2;
    while (usedIds.has(id)) id = `${base}-${n++}`;
    usedIds.add(id);
    return id;
  };

  const envAnchorIds: string[] = [];
  const agentAnchorIds: string[] = [];
  const timeAfterRefs: Array<{
    envKey: string;
    dependentNodeId: string;
    after: string;
  }> = [];

  for (const step of envSteps) {
    const key = String(step.env_id ?? 'env');
    groups.push({
      kind: 'env',
      key,
      stepId: String(step.id ?? ''),
      watchRoles: Array.isArray(step.watch_roles)
        ? step.watch_roles.map(String)
        : undefined,
      executorAgentName:
        step.executor_agent_name != null
          ? String(step.executor_agent_name)
          : undefined,
      executorTimeoutSeconds:
        typeof step.executor_timeout_seconds === 'number'
          ? step.executor_timeout_seconds
          : undefined,
    });
    const anchorId = uniqueId(`anchor-env-${key}`);
    envAnchorIds.push(anchorId);
    nodes.push({
      id: anchorId,
      type: 'anchor',
      x: 0,
      y: 0,
      width: anchorWidth(key),
      height: ANCHOR_H,
      anchor: {
        kind: 'env',
        label: key,
        triggerCount: Array.isArray(step.triggers) ? step.triggers.length : 0,
        raw: step,
      },
    });
    const triggers = Array.isArray(step.triggers) ? step.triggers : [];
    for (const t of triggers) {
      const trig = asRecord(t);
      if (!trig) continue;
      const triggerId = String(trig.id ?? '');
      const when = summarizeWhen(trig.when);
      const actions = summarizeActions(trig.actions);
      const nodeId = uniqueId(`env:${key}:${triggerId}`);
      envNodeIds.set(`${key}:${triggerId}`, nodeId);
      nodes.push({
        id: nodeId,
        type: 'trigger',
        x: 0,
        y: 0,
        width: TRIGGER_NODE_W,
        height: triggerNodeHeight(when, actions),
        trigger: {
          kind: 'env',
          triggerId,
          groupKey: key,
          when,
          actions,
          // Actionless triggers are observe-only sensors — except time
          // triggers, whose actionless form is a scheduled marker.
          isSensor: when.kind !== 'time' && actions.length === 0,
          notify: trig.notify != null ? String(trig.notify) : undefined,
          referencedBy: [],
          raw: trig,
        },
      });
      edges.push({
        id: `e-${anchorId}-${nodeId}`,
        source: anchorId,
        target: nodeId,
        kind: 'group',
      });
      if (when.kind === 'time') {
        const after = asRecord(trig.when)?.after;
        if (typeof after === 'string' && after) {
          timeAfterRefs.push({ envKey: key, dependentNodeId: nodeId, after });
        }
      }
    }
  }

  const nodeById = new Map(nodes.map(n => [n.id, n]));

  // A time trigger's `after` anchors its mark to another env trigger's firing — same cross-trigger dependency
  // as an env_trigger condition, so same edge + referencedBy treatment. Dangling anchors keep the node, drop the edge.
  for (const { envKey, dependentNodeId, after } of timeAfterRefs) {
    const sourceId = envNodeIds.get(`${envKey}:${after}`);
    if (!sourceId || sourceId === dependentNodeId) continue;
    edges.push({
      id: `e-${sourceId}-${dependentNodeId}`,
      source: sourceId,
      target: dependentNodeId,
      kind: 'env-ref',
    });
    const dependent = nodeById.get(dependentNodeId)?.trigger?.triggerId;
    if (dependent)
      nodeById.get(sourceId)?.trigger?.referencedBy.push(dependent);
  }

  for (const step of agentSteps) {
    const key = String(step.agent_name ?? 'agent');
    groups.push({ kind: 'agent', key, stepId: String(step.id ?? '') });
    const anchorId = uniqueId(`anchor-agent-${key}`);
    agentAnchorIds.push(anchorId);
    nodes.push({
      id: anchorId,
      type: 'anchor',
      x: 0,
      y: 0,
      width: anchorWidth(key),
      height: ANCHOR_H,
      anchor: {
        kind: 'agent',
        label: key,
        triggerCount: Array.isArray(step.triggers) ? step.triggers.length : 0,
        raw: step,
      },
    });
    const triggers = Array.isArray(step.triggers) ? step.triggers : [];
    for (const t of triggers) {
      const trig = asRecord(t);
      if (!trig) continue;
      const triggerId = String(trig.id ?? '');
      const when = summarizeWhen(trig.when);
      const actions = summarizeActions(trig.actions);
      const nodeId = uniqueId(`agent:${key}:${triggerId}`);
      nodes.push({
        id: nodeId,
        type: 'trigger',
        x: 0,
        y: 0,
        width: TRIGGER_NODE_W,
        height: triggerNodeHeight(when, actions),
        trigger: {
          kind: 'agent',
          triggerId,
          groupKey: key,
          when,
          actions,
          isSensor: false,
          referencedBy: [],
          raw: trig,
        },
      });
      edges.push({
        id: `e-${anchorId}-${nodeId}`,
        source: anchorId,
        target: nodeId,
        kind: 'group',
      });
      // A composite may reference the same env trigger more than once — dedupe so edge ids and referencedBy stay unique.
      const seenSources = new Set<string>();
      for (const ref of when.envRefs) {
        const sourceId = envNodeIds.get(`${ref.envId}:${ref.triggerId}`);
        // Dangling reference (env trigger not registered in this task):
        // keep the node, drop the edge.
        if (!sourceId || seenSources.has(sourceId)) continue;
        seenSources.add(sourceId);
        edges.push({
          id: `e-${sourceId}-${nodeId}`,
          source: sourceId,
          target: nodeId,
          kind: 'env-ref',
        });
        nodeById.get(sourceId)?.trigger?.referencedBy.push(triggerId);
      }
    }
  }

  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: 'LR', nodesep: NODE_SEP, ranksep: RANK_SEP });
  g.setDefaultEdgeLabel(() => ({}));
  nodes.forEach(n => g.setNode(n.id, { width: n.width, height: n.height }));
  edges.forEach(e => g.setEdge(e.source, e.target));
  // Lane pin: agent anchors (and all agent triggers) rank strictly right of every env trigger, even with no env-ref edge.
  for (const envAnchor of envAnchorIds) {
    for (const agentAnchor of agentAnchorIds) {
      g.setEdge(envAnchor, agentAnchor, { minlen: 2 });
    }
  }
  dagre.layout(g);

  let maxY = 0;
  const positioned = nodes.map(n => {
    const p = g.node(n.id);
    const x = p.x - n.width / 2;
    const y = p.y - n.height / 2;
    maxY = Math.max(maxY, y + n.height);
    return { ...n, x, y };
  });

  const triggerCount = positioned.filter(n => n.type === 'trigger').length;
  return {
    nodes: positioned,
    edges,
    groups,
    triggerCount,
    height: maxY,
  };
}
