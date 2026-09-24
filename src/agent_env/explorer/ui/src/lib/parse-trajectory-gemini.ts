/**
 * Gemini CLI stream-json trajectory parser.
 *
 * Source: the `gemini` CLI's `--output-format stream-json` events, stored
 * verbatim.
 *
 * Shape (flat array of records, no `attributes`):
 *   init        — { type:'init', timestamp, session_id, model }
 *   message     — { type:'message', timestamp, role, content, delta? }
 *                 role ∈ {'user','assistant'}; assistant turns stream as
 *                 multiple `delta:true` chunks whose `content` is the
 *                 INCREMENTAL fragment (concat to assemble full text).
 *                 A non-delta message represents the fully-assembled text.
 *   tool_use    — { type:'tool_use', timestamp, tool_name, tool_id, parameters }
 *   tool_result — { type:'tool_result', timestamp, tool_id, status, output }
 *                 Paired to tool_use by exact `tool_id` (string, not
 *                 positional).
 *   result      — { type:'result', timestamp, status, stats }
 *
 * Differences from Anthropic Claude Code's stream-json (handled by
 * parse-trajectory-claude-cli.ts):
 *   - Flat `type` with no `subtype` (Claude has `system/init`,
 *     `system/task_started`, etc.).
 *   - `message.content` is plain text; Claude's is an array of typed
 *     content blocks.
 *   - Streams with `delta:true` chunks; Claude batches turns whole.
 *   - No sub-agent / Task tool events.
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

export interface GeminiInitRecord {
  type: 'init';
  timestamp?: string;
  session_id?: string;
  model?: string;
}

export interface GeminiMessageRecord {
  type: 'message';
  timestamp?: string;
  role: 'user' | 'assistant';
  content?: string;
  delta?: boolean;
}

export interface GeminiToolUseRecord {
  type: 'tool_use';
  timestamp?: string;
  tool_name?: string;
  tool_id?: string;
  parameters?: Record<string, unknown>;
}

export interface GeminiToolResultRecord {
  type: 'tool_result';
  timestamp?: string;
  tool_id?: string;
  status?: 'success' | 'error' | string;
  output?: unknown;
}

export interface GeminiResultRecord {
  type: 'result';
  timestamp?: string;
  status?: string;
  stats?: {
    total_tokens?: number;
    input_tokens?: number;
    output_tokens?: number;
    tool_calls?: number;
    models?: string[];
    duration_ms?: number;
  };
}

export type GeminiRecord =
  | GeminiInitRecord
  | GeminiMessageRecord
  | GeminiToolUseRecord
  | GeminiToolResultRecord
  | GeminiResultRecord;

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/**
 * Heuristic for "is this a Gemini-CLI trajectory?". Cheap shape check on
 * the first record; precise enough to disambiguate from:
 *   - OTel/OpenInference spans (have `.attributes`, no `.type`)
 *   - CUA trajectories (driven by `envType==='cua'` flag, not shape)
 *   - Claude Code CLI stream-json (`type:'system', subtype:'init'`)
 */
export function looksLikeGeminiStreamJson(records: unknown[]): boolean {
  if (!Array.isArray(records) || records.length === 0) return false;
  const first = records[0] as Record<string, unknown> | null;
  if (!first || typeof first !== 'object') return false;
  return first.type === 'init' && typeof first.model === 'string';
}

// ---------------------------------------------------------------------------
// Parser
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

export function parseGeminiStreamJson(
  records: GeminiRecord[],
): ParsedTrajectory {
  const initRecord = records.find(
    (r): r is GeminiInitRecord => r.type === 'init',
  );
  const resultRecord = records.find(
    (r): r is GeminiResultRecord => r.type === 'result',
  );

  // --- Pass 1: index tool_results by tool_id
  const toolResults = new Map<string, ToolResultInfo>();
  for (const r of records) {
    if (r.type !== 'tool_result') continue;
    const id = r.tool_id;
    if (!id) continue;
    const output = stringifyToolOutput(r.output);
    const isError = r.status === 'error' || detectOutputError(output);
    toolResults.set(id, {
      output,
      startTime: r.timestamp ?? '',
      endTime: r.timestamp ?? '',
      // Per-tool latency isn't surfaced by the CLI; only the global
      // `result.stats.duration_ms` is. Leave 0 — the existing trajectory
      // viewer hides the ms badge when durationMs is 0.
      durationMs: 0,
      isError,
    });
  }

  // --- Pass 2: walk records, coalesce streamed assistant chunks, emit events
  const events: TrajectoryEvent[] = [];
  let toolCallCount = 0;
  const serviceCounts: Record<string, number> = {};
  let userPrompt = '';
  let firstUserSeen = false;

  // Accumulator for the in-progress assistant turn. Deltas are incremental fragments, so concatenate; a
  // non-delta record marks a fully-assembled turn and REPLACES the accumulated deltas (the CLI emits one or the other).
  let pendingChunks: string[] = [];

  const flushAssistant = (override?: string) => {
    let text: string;
    if (typeof override === 'string') {
      text = override;
    } else if (pendingChunks.length > 0) {
      text = pendingChunks.join('');
    } else {
      pendingChunks = [];
      return;
    }
    pendingChunks = [];
    if (text) events.push({ type: 'text', text });
  };

  for (const r of records) {
    if (r.type === 'message') {
      if (r.role === 'user') {
        flushAssistant();
        if (!firstUserSeen) {
          // First user message is the original prompt that kicked off
          // the session. Subsequent ones (if any) are conversation turns
          // the user injected — fold them into the timeline as separate
          // text events so they remain visible in steps.
          userPrompt = r.content ?? '';
          firstUserSeen = true;
        } else if (r.content) {
          events.push({ type: 'text', text: r.content });
        }
        continue;
      }
      // role === 'assistant'
      if (r.delta === true) {
        if (r.content) pendingChunks.push(r.content);
      } else {
        flushAssistant(r.content ?? '');
      }
    } else if (r.type === 'tool_use') {
      flushAssistant();
      const id = r.tool_id ?? '';
      const name = r.tool_name ?? 'tool';
      const input = (r.parameters ?? {}) as Record<string, unknown>;
      const result = toolResults.get(id) ?? null;
      events.push({ type: 'tool_call', id, name, input, result });
      toolCallCount += 1;
      const { service } = parseToolName(name);
      serviceCounts[service] = (serviceCounts[service] ?? 0) + 1;
    }
    // init, tool_result, result — handled out-of-band above
  }
  flushAssistant();

  // --- Metadata
  const model =
    initRecord?.model ?? resultRecord?.stats?.models?.[0] ?? 'unknown';
  const totalDurationMs = resultRecord?.stats?.duration_ms ?? 0;

  // The CLI doesn't emit a `num_turns` field; approximate by counting
  // assistant text events. Tool calls within a turn aren't counted as
  // separate turns. Matches roughly what Claude CLI's `num_turns` means
  // in the result record.
  const numTurns = events.filter(e => e.type === 'text').length;

  // Final response = last non-empty text event. The CLI's `result` record
  // doesn't carry the final assistant text directly (Claude's does);
  // pulling from the last text event is the closest equivalent.
  let finalResponse = '';
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e && e.type === 'text' && e.text.trim()) {
      finalResponse = e.text;
      break;
    }
  }

  const steps = groupEventsIntoSteps(events, finalResponse);

  return {
    model,
    userPrompt,
    finalResponse,
    events,
    steps,
    totalDurationMs,
    numTurns,
    toolCallCount,
    serviceCounts,
  };
}
