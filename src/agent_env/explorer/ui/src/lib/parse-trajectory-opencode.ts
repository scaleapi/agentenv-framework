/**
 * OpenCode CLI stream-json trajectory parser.
 *
 * Source: the `opencode run --format json` events, a flat array of
 * `{ type, timestamp, sessionID, part }`:
 *
 *   step_start   part.type='step-start'   — a model step boundary
 *   tool_use     part.type='tool'         — a tool call, self-contained:
 *                  part.tool, part.callID, part.state={status,input,output,time}
 *   text         part.type='text'         — assistant text (part.text)
 *   step_finish  part.type='step-finish'  — step end (part.reason, part.tokens)
 *   (reasoning   part.type='reasoning'    — thinking, when the model emits it)
 *
 * Unlike Gemini/Claude stream-json, tool calls are self-contained (the final
 * `state` carries both input and output), and each callID appears exactly once,
 * so there's no use/result pairing to do. The user prompt isn't echoed (it's
 * the message arg), and the model lives on the wrapping prompt_response
 * (threaded via opts.modelHint), matching the OpenClaw parser.
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

interface OpenCodeToolState {
  status?: string;
  input?: Record<string, unknown>;
  output?: unknown;
  metadata?: Record<string, unknown>;
  title?: string;
  time?: { start?: number; end?: number };
}

export interface OpenCodePart {
  type?: string; // 'step-start' | 'tool' | 'text' | 'step-finish' | 'reasoning'
  text?: string;
  tool?: string;
  callID?: string;
  id?: string;
  state?: OpenCodeToolState;
  reason?: string; // step-finish: 'stop' | 'tool-calls'
  tokens?: unknown;
  providerID?: string;
  modelID?: string;
  [key: string]: unknown;
}

export interface OpenCodeRecord {
  type?: string; // 'step_start' | 'tool_use' | 'text' | 'step_finish'
  timestamp?: number;
  sessionID?: string;
  part?: OpenCodePart;
  [key: string]: unknown;
}

const EVENT_TYPES = new Set(['step_start', 'tool_use', 'text', 'step_finish']);

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/**
 * Heuristic: an OpenCode `--format json` trajectory is a flat array whose
 * records carry a `part` object and a top-level `type` in the OpenCode event
 * set. Distinct from:
 *   - Claude CLI  (`type:'system', subtype:'init'`, no `part`)
 *   - Gemini      (`type:'init'`, no `part`)
 *   - Codex       (`type:'thread.started'`, no `part`)
 *   - OpenClaw    (`tool_call`/`thinking`/`tool_result` dicts, no `part`)
 *   - OTel spans  (have `attributes`)
 */
export function looksLikeOpenCodeStreamJson(records: unknown[]): boolean {
  if (!Array.isArray(records) || records.length === 0) return false;
  const first = records[0] as Record<string, unknown> | null;
  if (!first || typeof first !== 'object') return false;
  const part = first.part as Record<string, unknown> | undefined;
  return (
    typeof first.type === 'string' &&
    EVENT_TYPES.has(first.type) &&
    !!part &&
    typeof part === 'object'
  );
}

// ---------------------------------------------------------------------------
// Helpers
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

function eventType(r: OpenCodeRecord): string {
  return r.type ?? r.part?.type ?? '';
}

/**
 * OpenCode names MCP tools `mcp_<server>_<tool>` (single underscores), but the
 * shared `parseToolName` — and the viewer's service badge / grouping / copy
 * button (`stripMcpPrefix`) — expect the Anthropic MCP shape
 * `mcp__<server>__<tool>`. Rewrite to that shape so MCP tools classify by their
 * real service (QuickBooks, CRM, Linear, …) instead of collapsing to a generic
 * "Tool". Built-in tools (`todowrite`, `task`) have no `mcp_` prefix and pass
 * through unchanged.
 */
function canonicalizeToolName(raw: string): string {
  const m = /^mcp_([^_]+)_(.+)$/.exec(raw);
  return m ? `mcp__${m[1]}__${m[2]}` : raw;
}

/** Best-effort model id from the stream; opts.modelHint is the reliable source. */
function sniffModel(records: OpenCodeRecord[]): string | null {
  for (const r of records) {
    const p = r.part ?? {};
    const modelID = (p.modelID ??
      (p.state?.metadata as Record<string, unknown> | undefined)?.modelID) as
      | string
      | undefined;
    if (modelID) {
      const provider = p.providerID as string | undefined;
      return provider ? `${provider}/${modelID}` : modelID;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

export function parseOpenCodeStreamJson(
  records: OpenCodeRecord[],
  opts?: { modelHint?: string },
): ParsedTrajectory {
  const events: TrajectoryEvent[] = [];
  let toolCallCount = 0;
  const serviceCounts: Record<string, number> = {};

  for (const r of records) {
    const t = eventType(r);
    const part = r.part ?? {};

    if (part.type === 'reasoning') {
      // Check reasoning before text: opencode tags a reasoning part's
      // top-level `type` as 'text' too, so the text branch would swallow it.
      const text = part.text ?? '';
      if (text.trim()) events.push({ type: 'thinking', text });
    } else if (t === 'text' || part.type === 'text') {
      const text = part.text ?? '';
      if (text.trim()) events.push({ type: 'text', text });
    } else if (t === 'tool_use' || part.type === 'tool') {
      const name = canonicalizeToolName(part.tool ?? 'tool');
      const state = part.state ?? {};
      const input = (state.input as Record<string, unknown> | undefined) ?? {};
      const hasOutput = state.output !== undefined && state.output !== null;
      const output = stringifyToolOutput(state.output);
      const time = state.time ?? {};
      const durationMs =
        typeof time.start === 'number' && typeof time.end === 'number'
          ? Math.max(0, time.end - time.start)
          : 0;
      const result: ToolResultInfo | null = hasOutput
        ? {
            output,
            startTime: '',
            endTime: '',
            durationMs,
            isError: state.status === 'error' || detectOutputError(output),
          }
        : null;
      events.push({
        type: 'tool_call',
        id: part.callID ?? part.id ?? `opencode-${events.length}`,
        name,
        input,
        result,
      });
      toolCallCount += 1;
      const { service } = parseToolName(name);
      serviceCounts[service] = (serviceCounts[service] ?? 0) + 1;
    }
    // step_start / step_finish are boundaries — used for turns/duration below.
  }

  // Final response = last non-empty text event (OpenCode has no dedicated
  // terminal record; its last assistant text is the summary).
  let finalResponse = '';
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e && e.type === 'text' && e.text.trim()) {
      finalResponse = e.text;
      break;
    }
  }

  // A turn ≈ a step that finishes with reason 'stop' (tool-call steps continue
  // the same turn). Fall back to 1 when the stream carried any content.
  const stopCount = records.filter(
    r => eventType(r) === 'step_finish' && r.part?.reason === 'stop',
  ).length;
  const numTurns = stopCount || (events.length ? 1 : 0);

  // Duration from the event timestamps (ms epoch on each record).
  const stamps = records
    .map(r => (typeof r.timestamp === 'number' ? r.timestamp : null))
    .filter((n): n is number => n !== null);
  const firstStamp = stamps[0];
  const lastStamp = stamps[stamps.length - 1];
  const totalDurationMs =
    firstStamp !== undefined && lastStamp !== undefined
      ? Math.max(0, lastStamp - firstStamp)
      : 0;

  const model = opts?.modelHint || sniffModel(records) || 'unknown';
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
