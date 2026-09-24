/**
 * Parser for Claude Code CLI stream-json trajectories.
 *
 * Source: the `claude` CLI's `--output-format stream-json` records, forwarded
 * verbatim.
 *
 * Records are NOT OTel spans. They have shapes like:
 *   { type: "system", subtype: "init"|"task_started"|"task_progress"|..., ... }
 *   { type: "assistant", message: { model, content: [thinking|text|tool_use], usage }, parent_tool_use_id }
 *   { type: "user", message: { content: [tool_result] }, tool_use_result, parent_tool_use_id, timestamp }
 *   { type: "result", subtype: "success"|..., duration_ms, num_turns, result, modelUsage, ... }
 *
 * Sub-agent activity (Claude Code's Task/Agent tool) appears as separate
 * `system/task_started` + `system/task_progress` (one per inner tool use)
 * + `system/task_notification` records, each carrying a `tool_use_id` that
 * matches the parent assistant message's tool_use block. The sub-agent's
 * own assistant/user messages also appear in the stream with their
 * `parent_tool_use_id` set to that same id. We fold all that into a
 * `subAgentSummary` attached to the parent tool call rather than
 * emitting separate top-level events.
 */
import {
  ParsedTrajectory,
  ToolResultInfo,
  TrajectoryEvent,
  detectOutputError,
  groupEventsIntoSteps,
  parseToolName,
} from './parse-trajectory';

// ---------------------------------------------------------------------------
// Input types (minimal — we only declare the fields we read)
// ---------------------------------------------------------------------------

export interface ClaudeCliRecord {
  type: string;
  subtype?: string;
  // Common
  session_id?: string;
  uuid?: string;
  parent_tool_use_id?: string | null;
  timestamp?: string;
  // system/init
  model?: string;
  // system/task_*
  task_id?: string;
  tool_use_id?: string;
  description?: string;
  task_type?: string;
  prompt?: string;
  last_tool_name?: string | null;
  usage?: {
    total_tokens?: number;
    tool_uses?: number;
    duration_ms?: number;
  };
  // assistant
  message?: AssistantOrUserMessage;
  // user (tool_use_result block — agent SDK-specific metadata)
  tool_use_result?: {
    success?: boolean;
    [k: string]: unknown;
  };
  // result
  is_error?: boolean;
  duration_ms?: number;
  num_turns?: number;
  result?: string;
}

interface AssistantOrUserMessage {
  role?: string;
  model?: string;
  content?: ContentBlock[];
  usage?: Record<string, unknown>;
}

type ContentBlock =
  | { type: 'thinking'; thinking?: string; signature?: string }
  | { type: 'text'; text?: string }
  | {
      type: 'tool_use';
      id?: string;
      name?: string;
      input?: Record<string, unknown>;
    }
  | {
      type: 'tool_result';
      tool_use_id?: string;
      content?: unknown;
      is_error?: boolean;
    }
  | { type: string; [k: string]: unknown };

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

export function looksLikeClaudeCliStreamJson(
  records: ReadonlyArray<unknown>,
): boolean {
  if (records.length === 0) return false;
  const first = records[0] as Partial<ClaudeCliRecord> | null;
  return (
    !!first &&
    typeof first === 'object' &&
    first.type === 'system' &&
    first.subtype === 'init'
  );
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

export function parseClaudeCliStreamJson(
  records: ClaudeCliRecord[],
): ParsedTrajectory {
  const initRecord = records.find(
    r => r.type === 'system' && r.subtype === 'init',
  );
  const resultRecord = records.find(r => r.type === 'result');

  // --- Pass 1: index tool_results, sub-agent metadata, sub-agent progress
  const toolResults = new Map<string, ToolResultInfo>();
  const subAgentStarted = new Map<
    string,
    { description: string; prompt: string; taskType: string }
  >();
  // We keep only the *last* task_progress / task_notification per tool_use_id
  // for the aggregate counters — task_notification supersedes when present.
  const subAgentLatest = new Map<
    string,
    {
      totalTokens: number;
      toolUses: number;
      durationMs: number;
      lastToolName: string | null;
    }
  >();

  for (const r of records) {
    if (r.type === 'user') {
      // Tool results — both main-agent (parent_tool_use_id === null) and
      // sub-agent ones (parent_tool_use_id set). Index every block; the
      // sub-agent ones won't be referenced from top-level tool_use blocks
      // but harmless to include.
      for (const block of r.message?.content ?? []) {
        if (block.type !== 'tool_result') continue;
        const tb = block as Extract<ContentBlock, { type: 'tool_result' }>;
        const id = tb.tool_use_id;
        if (!id) continue;
        const output = stringifyResultContent(tb.content);
        const isError =
          tb.is_error === true ||
          r.tool_use_result?.success === false ||
          detectOutputError(output);
        toolResults.set(id, {
          output,
          startTime: r.timestamp ?? '',
          endTime: r.timestamp ?? '',
          durationMs: 0,
          isError,
        });
      }
    } else if (r.type === 'system' && r.subtype === 'task_started') {
      const id = r.tool_use_id;
      if (!id) continue;
      subAgentStarted.set(id, {
        description: r.description ?? '',
        prompt: r.prompt ?? '',
        taskType: r.task_type ?? '',
      });
    } else if (
      r.type === 'system' &&
      (r.subtype === 'task_progress' || r.subtype === 'task_notification')
    ) {
      const id = r.tool_use_id;
      if (!id) continue;
      const usage = r.usage ?? {};
      // Records are in chronological order, so unconditionally updating
      // gives last-wins semantics; the final `task_notification` (if
      // present) lands last and supersedes the running task_progress
      // snapshots. Fall back to prior values for any field a later
      // record happens to omit.
      const existing = subAgentLatest.get(id);
      subAgentLatest.set(id, {
        totalTokens: usage.total_tokens ?? existing?.totalTokens ?? 0,
        toolUses: usage.tool_uses ?? existing?.toolUses ?? 0,
        durationMs: usage.duration_ms ?? existing?.durationMs ?? 0,
        lastToolName: r.last_tool_name ?? existing?.lastToolName ?? null,
      });
    }
  }

  // --- Pass 2: walk assistant messages recursively
  // Top-level: parent_tool_use_id === null
  // Sub-agent at level N: parent_tool_use_id === parent's tool_use_id
  // The walk also accumulates the top-level toolCallCount + serviceCounts;
  // sub-agent inner tools are added on top so the header badges reflect
  // total work done across all nested levels.
  const ctx: WalkContext = {
    records,
    toolResults,
    subAgentStarted,
    subAgentLatest,
    toolCallCount: 0,
    serviceCounts: {},
  };
  const events = walkAssistantEvents(ctx, null);
  let model = initRecord?.model ?? 'unknown';
  // Resolve the top-level model from the first non-sub-agent assistant
  // record (so we report the parent's model even when a sub-agent runs on
  // a different one).
  for (const r of records) {
    if (r.type === 'assistant' && !r.parent_tool_use_id && r.message?.model) {
      model = r.message.model;
      break;
    }
  }
  const toolCallCount = ctx.toolCallCount;
  const serviceCounts = ctx.serviceCounts;

  // --- Metadata
  const finalResponse =
    resultRecord?.result ?? lastTextOf(events) ?? '';
  const totalDurationMs = resultRecord?.duration_ms ?? 0;
  const numTurns =
    resultRecord?.num_turns ??
    records.filter(r => r.type === 'assistant' && !r.parent_tool_use_id).length;

  const steps = groupEventsIntoSteps(events, finalResponse);

  return {
    model,
    // Stream-json doesn't echo the human's input prompt back into the
    // event stream — the prompt is the *input* to the CLI, not an event.
    // Surfacing it would require plumbing it through fetchTrajectory
    // separately. Empty string is acceptable; the viewer renders it as
    // a (collapsed) empty pre-block.
    userPrompt: '',
    finalResponse,
    events,
    steps,
    totalDurationMs,
    numTurns,
    toolCallCount,
    serviceCounts,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface WalkContext {
  records: ClaudeCliRecord[];
  toolResults: Map<string, ToolResultInfo>;
  subAgentStarted: Map<
    string,
    { description: string; prompt: string; taskType: string }
  >;
  subAgentLatest: Map<
    string,
    {
      totalTokens: number;
      toolUses: number;
      durationMs: number;
      lastToolName: string | null;
    }
  >;
  // Accumulators — only mutated when walking the top level (parentToolUseId
  // === null). Sub-agent inner tools are added on top of the top-level
  // count so the trajectory header reflects total work; sub-agent
  // serviceCounts feed into the same map for the same reason.
  toolCallCount: number;
  serviceCounts: Record<string, number>;
}

/**
 * Walk assistant messages whose `parent_tool_use_id` matches
 * `parentToolUseId` (null for top-level main-agent activity). Recursively
 * descends into each `tool_use` block that has a matching sub-agent
 * summary, producing nested `subAgentSummary.events` arrays.
 *
 * `toolCallCount` / `serviceCounts` on `ctx` aggregate across the entire
 * tree so the trajectory header counts reflect everything that ran.
 */
function walkAssistantEvents(
  ctx: WalkContext,
  parentToolUseId: string | null,
): TrajectoryEvent[] {
  const events: TrajectoryEvent[] = [];
  for (const r of ctx.records) {
    if (r.type !== 'assistant') continue;
    const myParent = r.parent_tool_use_id ?? null;
    if (myParent !== parentToolUseId) continue;

    for (const block of r.message?.content ?? []) {
      if (block.type === 'thinking') {
        const tb = block as Extract<ContentBlock, { type: 'thinking' }>;
        const text = tb.thinking ?? '';
        if (text) events.push({ type: 'thinking', text });
      } else if (block.type === 'text') {
        const tb = block as Extract<ContentBlock, { type: 'text' }>;
        const text = tb.text ?? '';
        if (text) events.push({ type: 'text', text });
      } else if (block.type === 'tool_use') {
        const tb = block as Extract<ContentBlock, { type: 'tool_use' }>;
        const id = tb.id ?? '';
        const name = tb.name ?? 'tool';
        const input = (tb.input ?? {}) as Record<string, unknown>;
        const result = ctx.toolResults.get(id) ?? null;

        const started = ctx.subAgentStarted.get(id);
        const latest = ctx.subAgentLatest.get(id);
        const hasSubAgent = !!started || !!latest;
        const subAgentEvents = hasSubAgent
          ? walkAssistantEvents(ctx, id)
          : [];
        const subAgentSummary = hasSubAgent
          ? {
              description: started?.description ?? '',
              prompt: started?.prompt ?? '',
              taskType: started?.taskType ?? '',
              totalTokens: latest?.totalTokens ?? 0,
              toolUses: latest?.toolUses ?? 0,
              durationMs: latest?.durationMs ?? 0,
              lastToolName: latest?.lastToolName ?? null,
              events: subAgentEvents,
            }
          : undefined;

        events.push({
          type: 'tool_call',
          id,
          name,
          input,
          result,
          ...(subAgentSummary ? { subAgentSummary } : {}),
        });
        ctx.toolCallCount += 1;
        const { service } = parseToolName(name);
        ctx.serviceCounts[service] = (ctx.serviceCounts[service] ?? 0) + 1;
      }
    }
  }
  return events;
}

function stringifyResultContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (content == null) return '';
  try {
    return JSON.stringify(content, null, 2);
  } catch {
    return String(content);
  }
}

function lastTextOf(events: TrajectoryEvent[]): string {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e && e.type === 'text' && e.text.trim()) return e.text;
  }
  return '';
}
