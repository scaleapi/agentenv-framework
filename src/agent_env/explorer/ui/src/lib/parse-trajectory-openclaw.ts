/**
 * OpenClaw A2A trajectory parser.
 *
 * Source: the OpenClaw session transcript, as a flat list of event dicts:
 *   tool_call      — { type:'tool_call', tool, input, tool_use_id }
 *                    `input` is `str(arguments)` (Python repr, truncated to
 *                    500 chars) — usually NOT valid JSON.
 *   tool_result    — { type:'tool_result', tool_use_id, tool_name, output, is_error }
 *                    paired to its call by exact `tool_use_id`.
 *   thinking       — { type:'thinking', content }
 *   final_response — { type:'final_response', content, stop_reason, latency_ms }
 *                    always appended last (even on error).
 *
 * Distinct from the CLI stream-json parsers (claude/gemini/codex) and OTEL:
 *   - flat list, snake_case keys, a terminal `final_response` record
 *   - no `model`, no `init`/`result` envelope, no `.attributes`
 *   - the agent's reasoning surfaces as `thinking` records — there are no
 *     plain assistant `text` records, which is why the generic OTEL path
 *     reports "0 turns / Model: unknown" for OpenClaw runs.
 */
import {
  type ParsedTrajectory,
  type ThinkingEvent,
  type ToolCallEvent,
  type ToolResultInfo,
  type TrajectoryEvent,
  type TrajectoryStep,
  detectOutputError,
  parseToolName,
} from './parse-trajectory';

// ---------------------------------------------------------------------------
// Record shapes
// ---------------------------------------------------------------------------

export interface OpenClawToolCallRecord {
  type: 'tool_call';
  tool?: string;
  input?: string;
  tool_use_id?: string;
}

export interface OpenClawToolResultRecord {
  type: 'tool_result';
  tool_use_id?: string;
  tool_name?: string;
  output?: string;
  is_error?: boolean;
}

export interface OpenClawThinkingRecord {
  type: 'thinking';
  content?: string;
}

export interface OpenClawFinalRecord {
  type: 'final_response';
  content?: string;
  stop_reason?: string | null;
  latency_ms?: number;
}

export type OpenClawRecord =
  | OpenClawToolCallRecord
  | OpenClawToolResultRecord
  | OpenClawThinkingRecord
  | OpenClawFinalRecord;

const OPENCLAW_TYPES = new Set([
  'tool_call',
  'tool_result',
  'thinking',
  'final_response',
]);

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/**
 * Heuristic for "is this an OpenClaw trajectory?". Every record must be an
 * object whose `type` is one of OpenClaw's four, AND at least one record must
 * carry an OpenClaw-unique signal — the terminal `final_response` type, a
 * `tool_call` with a snake_case `tool_use_id`, or a `tool_result` with
 * `tool_name`/`is_error`. This disambiguates from:
 *   - OTel/OpenInference spans (have `.attributes`, no `.type`)
 *   - Gemini `tool_result` (uses `tool_id`/`status`/`output`, not
 *     `tool_use_id`/`tool_name`/`is_error`)
 *   - Claude CLI / Codex stream-json (`system`/`thread.started`/… types)
 */
export function looksLikeOpenClawTrajectory(records: unknown[]): boolean {
  if (!Array.isArray(records) || records.length === 0) return false;
  let sawSignature = false;
  for (const r of records) {
    if (!r || typeof r !== 'object') return false;
    const rec = r as Record<string, unknown>;
    if ('attributes' in rec && !('type' in rec)) return false; // OTEL span
    const t = rec.type;
    if (typeof t !== 'string' || !OPENCLAW_TYPES.has(t)) return false;
    if (t === 'final_response') sawSignature = true;
    else if (t === 'tool_call' && 'tool_use_id' in rec) sawSignature = true;
    else if (t === 'tool_result' && ('is_error' in rec || 'tool_name' in rec))
      sawSignature = true;
  }
  return sawSignature;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/**
 * `input` is `str(arguments)` — a Python repr, usually not
 * JSON (single-quoted keys). Try JSON first (some tools emit real JSON); fall
 * back to surfacing the raw string verbatim under `arguments` so the viewer
 * still shows the call's inputs.
 */
function parseToolInput(input: string | undefined): Record<string, unknown> {
  if (!input) return {};
  try {
    const v = JSON.parse(input);
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      return v as Record<string, unknown>;
    }
  } catch {
    // python-repr string, not JSON — fall through
  }
  return { arguments: input };
}

/**
 * OpenClaw `thinking` blocks frequently open with a markdown bold title
 * (e.g. `**Initiating Bootstrap Sequence**`). Use that as the step label when
 * present, else the first line, truncated.
 */
function deriveStepLabel(thinkingText: string): string {
  const trimmed = thinkingText.trim();
  const bold = trimmed.match(/^\*\*(.+?)\*\*/);
  if (bold && bold[1]) return bold[1].trim();
  const firstLine = trimmed.split('\n', 1)[0] ?? '';
  return firstLine.length > 80 ? firstLine.slice(0, 77) + '…' : firstLine;
}

export function parseOpenClawTrajectory(
  records: OpenClawRecord[],
  opts?: { modelHint?: string },
): ParsedTrajectory {
  // --- Pass 1: index tool_results by tool_use_id
  const toolResults = new Map<string, ToolResultInfo>();
  for (const r of records) {
    if (r.type !== 'tool_result') continue;
    const id = r.tool_use_id;
    if (!id) continue;
    const output = r.output ?? '';
    toolResults.set(id, {
      output,
      startTime: '',
      endTime: '',
      // No per-tool latency is recorded; only the run-level
      // `final_response.latency_ms` exists. 0 makes the viewer hide the badge.
      durationMs: 0,
      isError: r.is_error === true || detectOutputError(output),
    });
  }

  // --- Pass 2: walk records in order, emit events
  const events: TrajectoryEvent[] = [];
  let toolCallCount = 0;
  const serviceCounts: Record<string, number> = {};
  let finalResponse = '';
  let totalDurationMs = 0;

  for (const r of records) {
    if (r.type === 'thinking') {
      const text = r.content ?? '';
      if (text) {
        const ev: ThinkingEvent = { type: 'thinking', text };
        events.push(ev);
      }
    } else if (r.type === 'tool_call') {
      const id = r.tool_use_id ?? '';
      const name = r.tool ?? 'tool';
      const result = (id && toolResults.get(id)) || null;
      const ev: ToolCallEvent = {
        type: 'tool_call',
        id,
        name,
        input: parseToolInput(r.input),
        result,
      };
      events.push(ev);
      toolCallCount += 1;
      const { service } = parseToolName(name);
      serviceCounts[service] = (serviceCounts[service] ?? 0) + 1;
    } else if (r.type === 'final_response') {
      finalResponse = r.content ?? '';
      totalDurationMs = typeof r.latency_ms === 'number' ? r.latency_ms : 0;
    }
    // tool_result handled out-of-band in pass 1
  }

  // --- Steps: the shared `groupEventsIntoSteps` keys off `text` events, which
  // OpenClaw never emits — so build reasoning→action steps on `thinking`
  // boundaries instead. Each thinking block starts a step (labeled by its
  // title) and the tool calls that follow it form the step body. Tool calls
  // before any thinking fall into a leading "Agent actions" step.
  const steps: TrajectoryStep[] = [];
  let current: TrajectoryStep | null = null;
  const flush = () => {
    if (current && (current.events.length > 0 || current.label)) {
      current.toolCount = current.events.filter(
        e => e.type === 'tool_call',
      ).length;
      steps.push(current);
    }
    current = null;
  };
  for (const e of events) {
    if (e.type === 'thinking') {
      flush();
      current = { label: deriveStepLabel(e.text), events: [e], toolCount: 0 };
    } else {
      if (!current) current = { label: 'Agent actions', events: [], toolCount: 0 };
      current.events.push(e);
    }
  }
  flush();

  return {
    // The trajectory carries no model; the caller threads it from the
    // prompt-response (`pr.model`). Falls back to 'unknown'.
    model: opts?.modelHint || 'unknown',
    // The user/task prompt lives on the prompt-response, not in the
    // trajectory — the viewer renders it from there separately.
    userPrompt: '',
    finalResponse,
    events,
    steps,
    totalDurationMs,
    // OpenClaw is chat-first and flattens its execute() turns into one
    // transcript with no turn markers, so a literal turn count isn't
    // recoverable. The reasoning→action step count is the closest meaningful
    // proxy (and avoids the misleading "0 turns" the generic parser shows).
    numTurns: steps.length,
    toolCallCount,
    serviceCounts,
  };
}
