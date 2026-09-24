/**
 * Codex CLI stream-json (NDJSON) trajectory parser.
 *
 * Source: the Codex CLI's `codex exec --json` output, stored verbatim.
 *
 * Shape (flat array of records, no `attributes`):
 *
 *   thread.started
 *     { type: 'thread.started', thread_id }
 *
 *   item.completed (the bulk of the stream)
 *     { type: 'item.completed', item: { type, ... } }
 *
 *     item.type values:
 *       - 'agent_message'       → assistant text turn (item.text)
 *       - 'command_execution'   → bash command run (tool)
 *       - 'mcp_tool_call'       → MCP tool invocation (tool)
 *       - 'file_change'         → file create/edit/delete (tool)
 *       - 'web_search'          → web search (tool)
 *
 * The CLI's per-item field names aren't fully documented and may evolve, so this
 * parser reads each item's fields defensively (multiple candidate names) and falls
 * back to a JSON dump for anything unrecognized — render imperfectly rather than crash.
 *
 * Differences from Claude Code and Gemini stream-json:
 *   - No streaming deltas — each item.completed is whole.
 *   - No user-message echoes — user prompt isn't included in the
 *     stream (it's the prompt arg to `codex exec`).
 *   - Tools and their results are folded into a single record (vs.
 *     Gemini's separate tool_use / tool_result pair).
 *   - No per-turn aggregate `result` record — totals live in the
 *     wider envelope outside this trajectory array.
 */
import {
  type ParsedTrajectory,
  type ToolResultInfo,
  type TrajectoryEvent,
  detectOutputError,
  groupEventsIntoSteps,
  parseToolName,
} from './parse-trajectory';

// ---------------------------------------------------------------------------
// Record shapes
// ---------------------------------------------------------------------------

export interface CodexThreadStartedRecord {
  type: 'thread.started';
  thread_id?: string;
}

export interface CodexAgentMessageItem {
  type: 'agent_message';
  text?: string;
  // Defensive read: some Codex CLI versions surface model on the item.
  model?: string;
}

export interface CodexCommandExecutionItem {
  type: 'command_execution';
  // CLI's actual key varies by build; accept the common candidates.
  command?: string;
  cmd?: string;
  argv?: string[];
  // Output / exit info — read defensively.
  output?: string;
  stdout?: string;
  stderr?: string;
  exit_code?: number;
  status?: string;
  duration_ms?: number;
}

export interface CodexMcpToolCallItem {
  type: 'mcp_tool_call';
  server?: string;
  server_name?: string;
  tool?: string;
  tool_name?: string;
  name?: string;
  arguments?: Record<string, unknown>;
  args?: Record<string, unknown>;
  input?: Record<string, unknown>;
  output?: unknown;
  result?: unknown;
  status?: string;
  duration_ms?: number;
}

export interface CodexFileChangeItem {
  type: 'file_change';
  path?: string;
  // create / edit / delete / move, etc.
  kind?: string;
  diff?: string;
  before?: string;
  after?: string;
}

export interface CodexWebSearchItem {
  type: 'web_search';
  query?: string;
  results?: unknown;
  output?: unknown;
}

export type CodexItem =
  | CodexAgentMessageItem
  | CodexCommandExecutionItem
  | CodexMcpToolCallItem
  | CodexFileChangeItem
  | CodexWebSearchItem
  | { type: string; [key: string]: unknown };

export interface CodexItemCompletedRecord {
  type: 'item.completed';
  item?: CodexItem;
}

export type CodexRecord =
  | CodexThreadStartedRecord
  | CodexItemCompletedRecord
  | { type: string; [key: string]: unknown };

const TOOL_ITEM_TYPES = new Set([
  'command_execution',
  'mcp_tool_call',
  'file_change',
  'web_search',
]);

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/**
 * Heuristic: a Codex CLI trajectory always opens with a
 * `{type:'thread.started', thread_id:'...'}` record. Distinct from:
 *   - Claude CLI's `system/init` (has `type:'system', subtype:'init'`)
 *   - Gemini's `init` (has `type:'init', model:'...'`)
 *   - OTel spans (have `attributes`)
 */
export function looksLikeCodexStreamJson(records: unknown[]): boolean {
  if (!Array.isArray(records) || records.length === 0) return false;
  const first = records[0] as Record<string, unknown> | null;
  if (!first || typeof first !== 'object') return false;
  return first.type === 'thread.started';
}

// ---------------------------------------------------------------------------
// Per-item rendering helpers
// ---------------------------------------------------------------------------

function stringifyToolOutput(output: unknown): string {
  if (output == null) return '';
  if (typeof output === 'string') return output;
  try {
    return JSON.stringify(output, null, 2);
  } catch {
    return String(output);
  }
}

interface ToolCallShape {
  name: string;
  input: Record<string, unknown>;
  result: ToolResultInfo | null;
}

/**
 * Map a single `item.completed.item` (one of the tool-type items) into
 * the `ToolCallEvent`-shaped name / input / result trio. We collapse
 * Codex's "completed action" record into a single tool call rather
 * than splitting use vs. result, since the CLI already pairs them.
 */
function itemToToolCall(item: CodexItem): ToolCallShape {
  const itemType = (item.type as string) || 'tool';
  // Defensive field reads — see the type defs above for the candidates.
  switch (itemType) {
    case 'command_execution': {
      const ci = item as CodexCommandExecutionItem;
      const cmd =
        ci.command ??
        ci.cmd ??
        (Array.isArray(ci.argv) ? ci.argv.join(' ') : undefined);
      const stdout = ci.stdout ?? '';
      const stderr = ci.stderr ?? '';
      const exitCode = typeof ci.exit_code === 'number' ? ci.exit_code : null;
      const isError =
        ci.status === 'error' ||
        (exitCode !== null && exitCode !== 0) ||
        detectOutputError(stdout || stderr);
      // Compose a single output blob with exit / stderr / stdout so
      // the existing ToolCard render path (which shows `result.output`
      // in a <pre>) surfaces all three. If the item already has a
      // single `output` field, prefer that as-is.
      const output =
        typeof ci.output === 'string' && ci.output.trim()
          ? ci.output
          : [
              exitCode !== null ? `exit=${exitCode}` : null,
              stderr ? `stderr:\n${stderr}` : null,
              stdout ? `stdout:\n${stdout}` : null,
            ]
              .filter(Boolean)
              .join('\n\n');
      return {
        name: 'Bash',
        input: { command: cmd ?? '' },
        result: output
          ? {
              output,
              startTime: '',
              endTime: '',
              durationMs:
                typeof ci.duration_ms === 'number' ? ci.duration_ms : 0,
              isError,
            }
          : null,
      };
    }
    case 'mcp_tool_call': {
      const mi = item as CodexMcpToolCallItem;
      const server = mi.server ?? mi.server_name ?? '';
      const tool = mi.tool ?? mi.tool_name ?? mi.name ?? 'tool';
      const name = server ? `${server}::${tool}` : tool;
      const input =
        (mi.arguments as Record<string, unknown> | undefined) ??
        (mi.args as Record<string, unknown> | undefined) ??
        (mi.input as Record<string, unknown> | undefined) ??
        {};
      const rawResult = mi.output ?? mi.result;
      const output = stringifyToolOutput(rawResult);
      return {
        name,
        input,
        result:
          rawResult !== undefined
            ? {
                output,
                startTime: '',
                endTime: '',
                durationMs:
                  typeof mi.duration_ms === 'number' ? mi.duration_ms : 0,
                isError: mi.status === 'error' || detectOutputError(output),
              }
            : null,
      };
    }
    case 'file_change': {
      const fi = item as CodexFileChangeItem;
      const kind = fi.kind ?? 'change';
      const name =
        kind === 'create' ? 'Write' : kind === 'delete' ? 'Delete' : 'Edit';
      const input: Record<string, unknown> = { path: fi.path ?? '' };
      if (fi.kind) input.kind = fi.kind;
      // Codex emits either a pre-formed `diff`, or raw `before`/`after`
      // snapshots (for some file_change kinds). Synthesize a unified-
      // diff-ish blob from the snapshots so the content still renders
      // instead of disappearing into a blank tool card.
      const diffContent =
        fi.diff ??
        (fi.before !== undefined || fi.after !== undefined
          ? `--- before\n${fi.before ?? ''}\n+++ after\n${fi.after ?? ''}`
          : '');
      return {
        name,
        input,
        result: diffContent
          ? {
              output: diffContent,
              startTime: '',
              endTime: '',
              durationMs: 0,
              isError: false,
            }
          : null,
      };
    }
    case 'web_search': {
      const wi = item as CodexWebSearchItem;
      const output = stringifyToolOutput(wi.results ?? wi.output);
      return {
        name: 'WebSearch',
        input: { query: wi.query ?? '' },
        result: output
          ? {
              output,
              startTime: '',
              endTime: '',
              durationMs: 0,
              isError: false,
            }
          : null,
      };
    }
    default: {
      // Unknown tool item type — best-effort fallback so it still renders.
      const generic = item as Record<string, unknown>;
      const { type: _t, ...rest } = generic;
      const output = stringifyToolOutput(rest);
      return {
        name: itemType,
        input: rest as Record<string, unknown>,
        result: output
          ? {
              output,
              startTime: '',
              endTime: '',
              durationMs: 0,
              isError: false,
            }
          : null,
      };
    }
  }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

export function parseCodexStreamJson(records: CodexRecord[]): ParsedTrajectory {
  const events: TrajectoryEvent[] = [];
  let toolCallCount = 0;
  const serviceCounts: Record<string, number> = {};
  let model = 'unknown';

  // Walk in order: agent_message items become text events; tool-type
  // items become tool_call events with the result inlined. Codex
  // doesn't emit a separate user-message echo, so we leave
  // `userPrompt` empty (it lives on the wrapping prompt_response in
  // the instance context).
  for (const r of records) {
    if (r.type !== 'item.completed') continue;
    const itemCompleted = r as CodexItemCompletedRecord;
    const item = itemCompleted.item;
    if (!item || typeof item !== 'object') continue;
    const itemType = item.type;
    if (itemType === 'agent_message') {
      const msg = item as CodexAgentMessageItem;
      if (msg.model) model = msg.model;
      const text = msg.text ?? '';
      if (text) events.push({ type: 'text', text });
    } else if (typeof itemType === 'string' && TOOL_ITEM_TYPES.has(itemType)) {
      const { name, input, result } = itemToToolCall(item);
      events.push({
        type: 'tool_call',
        // No stable id is emitted per item; synthesize a positional
        // one so React keys don't collide.
        id: `codex-${events.length}`,
        name,
        input,
        result,
      });
      toolCallCount += 1;
      const { service } = parseToolName(name);
      serviceCounts[service] = (serviceCounts[service] ?? 0) + 1;
    }
    // Unknown item.type (e.g. future additions) silently skipped —
    // better to under-render than blow up the timeline.
  }

  // Final response = last non-empty text event. The CLI doesn't emit a
  // dedicated "final" record (no `result` envelope like Claude/Gemini),
  // so the heuristic matches Gemini's behavior.
  let finalResponse = '';
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e && e.type === 'text' && e.text.trim()) {
      finalResponse = e.text;
      break;
    }
  }

  // numTurns ≈ assistant text events. Codex doesn't emit per-turn
  // boundaries explicitly; one agent_message per turn is the
  // observed pattern.
  const numTurns = events.filter(e => e.type === 'text').length;

  // totalDurationMs unavailable from the trajectory itself — Codex
  // doesn't emit a wrap-up record carrying it. Sum per-item
  // durations when present as the next best thing.
  const totalDurationMs = records
    .filter(
      (r): r is CodexItemCompletedRecord => r.type === 'item.completed',
    )
    .map(r => {
      const item = r.item as { duration_ms?: unknown } | undefined;
      return typeof item?.duration_ms === 'number' ? item.duration_ms : 0;
    })
    .reduce((a, b) => a + b, 0);

  const steps = groupEventsIntoSteps(events, finalResponse);

  return {
    model,
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
