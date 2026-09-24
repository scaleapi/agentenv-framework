/**
 * Parse OTel trajectory spans into a structured timeline.
 *
 * The main `parseOtelTrajectory` keys off the OTEL GenAI semantic
 * conventions (`gen_ai.*`) emitted by claude_code and codex agents.
 * For openai_agents_sdk runs (which emit OpenInference-shaped spans via
 * the SDK's instrumentor), `parse-trajectory-openinference.ts` provides
 * an in-place adapter that synthesizes equivalent `gen_ai.*` attributes;
 * we run it as a pre-pass so the rest of this file stays unchanged.
 */
import { normalizeOpenInferenceSpan } from './parse-trajectory-openinference';
import {
  ClaudeCliRecord,
  looksLikeClaudeCliStreamJson,
  parseClaudeCliStreamJson,
} from './parse-trajectory-claude-cli';
import {
  GeminiRecord,
  looksLikeGeminiStreamJson,
  parseGeminiStreamJson,
} from './parse-trajectory-gemini';
import {
  CodexRecord,
  looksLikeCodexStreamJson,
  parseCodexStreamJson,
} from './parse-trajectory-codex';
import {
  OpenClawRecord,
  looksLikeOpenClawTrajectory,
  parseOpenClawTrajectory,
} from './parse-trajectory-openclaw';
import {
  OpenCodeRecord,
  looksLikeOpenCodeStreamJson,
  parseOpenCodeStreamJson,
} from './parse-trajectory-opencode';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface OtelSpan {
  name: string;
  context: { trace_id: string; span_id: string; trace_state?: string };
  kind: string;
  parent_id: string | null;
  start_time: string;
  end_time: string;
  status: { status_code: string };
  attributes: Record<string, string>;
  events: unknown[];
  links: unknown[];
  resource: { attributes: Record<string, string> };
}

export interface ToolResultInfo {
  output: string;
  startTime: string;
  endTime: string;
  durationMs: number;
  isError: boolean;
  screenshot?: string; // base64 PNG, post-action (CUA only)
  /**
   * base64 frame with the action marker baked in by the harness — the pre-action
   * frame the gesture was measured on, with the ring/dot/line drawn onto the
   * pixels (`cua.acted_screenshot_annotated`). The viewer shows this directly as
   * the action-location frame. (CUA only; absent for non-gesture tools.)
   */
  actedScreenshotAnnotated?: string;
}

export interface ThinkingEvent {
  type: 'thinking';
  text: string;
}

export interface TextEvent {
  type: 'text';
  text: string;
}

/**
 * Summary of a sub-agent invocation, attached to the parent `Task`-named
 * ToolCallEvent when present. Populated only by the Claude Code CLI
 * stream-json parser today — the source format encodes sub-agent activity
 * as separate system records keyed by `tool_use_id`, which the parser
 * reattaches to the parent tool call. OTel/CUA paths leave this undefined.
 */
export interface SubAgentSummary {
  description: string;
  prompt: string;
  taskType: string;
  totalTokens: number;
  toolUses: number;
  durationMs: number;
  lastToolName: string | null;
  /**
   * The sub-agent's own timeline (its assistant/user messages reattached
   * by `parent_tool_use_id`). Recurses naturally: a sub-agent that spawns
   * another sub-agent gets its own `subAgentSummary.events` populated.
   * Empty when the source format doesn't carry per-message sub-agent
   * detail (e.g. OTel paths).
   */
  events: TrajectoryEvent[];
}

export interface ToolCallEvent {
  type: 'tool_call';
  id: string;
  name: string;
  input: Record<string, unknown>;
  result: ToolResultInfo | null;
  subAgentSummary?: SubAgentSummary;
}

export type TrajectoryEvent = ThinkingEvent | TextEvent | ToolCallEvent;

export interface TrajectoryStep {
  label: string;
  events: TrajectoryEvent[];
  toolCount: number;
}

export interface ParsedTrajectory {
  model: string;
  userPrompt: string;
  finalResponse: string;
  events: TrajectoryEvent[];
  steps: TrajectoryStep[];
  totalDurationMs: number;
  numTurns: number;
  toolCallCount: number;
  serviceCounts: Record<string, number>;
  /**
   * base64 screenshot of the starting state — the screen the agent saw before
   * its first action (CUA only; from the harness `initial_screenshot` span).
   * Rendered at the top of the trajectory so the "before any action" state is
   * visible, since execute_tool spans only carry post-action frames.
   */
  initialScreenshot?: string;
}

// ---------------------------------------------------------------------------
// Service registry — known overrides only; unknown services get auto-labeled
// ---------------------------------------------------------------------------

const SERVICE_MAP: Record<string, [string, string]> = {
  linear: ['Linear', '#5E6AD2'],
  slack: ['Slack', '#4A154B'],
  email: ['Email', '#2563EB'],
  send_email: ['Email', '#2563EB'],
  search_emails: ['Email', '#2563EB'],
  list_emails: ['Email', '#2563EB'],
  get_email: ['Email', '#2563EB'],
  reply_to_email: ['Email', '#2563EB'],
  contacts: ['Contacts', '#0D9488'],
  calendar: ['Calendar', '#EA580C'],
  crm: ['CRM', '#DC2626'],
  cab: ['Rides', '#CA8A04'],
  flights: ['Flights', '#0284C7'],
  airbnb: ['Airbnb', '#FF5A5F'],
  messaging: ['Messaging', '#4F46E5'],
  reminder: ['Reminders', '#DB2777'],
  channels: ['Slack', '#4A154B'],
  conversations: ['Slack', '#4A154B'],
  computer: ['Computer', '#6366F1'],
};

const DEFAULT_TOOL_COLOR = '#6B7280';

const BUILTIN_TOOLS = new Set([
  'Read',
  'Write',
  'Edit',
  'MultiEdit',
  'Bash',
  'Grep',
  'Glob',
  'Task',
  'TaskOutput',
  'WebFetch',
  'WebSearch',
  'NotebookEdit',
  'TodoWrite',
  'ExitPlanMode',
  'EnterPlanMode',
  'AskUserQuestion',
]);

const EMAIL_PREFIXES = [
  'send_email',
  'search_emails',
  'list_emails',
  'get_email',
  'reply_to_email',
];

export interface ParsedToolName {
  service: string;
  color: string;
  action: string;
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export function parseToolName(rawName: string): ParsedToolName {
  if (BUILTIN_TOOLS.has(rawName)) {
    return { service: 'Code', color: DEFAULT_TOOL_COLOR, action: rawName };
  }

  const parts = rawName.split('__');
  if (parts.length >= 3) {
    const fullAction = parts.slice(2).join('__');
    const segments = fullAction.split('_');
    let serviceKey = segments[0] ?? '';
    let actionParts = segments.slice(1);

    // Check multi-word email prefixes first
    for (const prefix of EMAIL_PREFIXES) {
      if (fullAction.startsWith(prefix)) {
        serviceKey = prefix;
        const rest = fullAction.slice(prefix.length).replace(/^_/, '');
        actionParts = rest ? rest.split('_') : prefix.split('_');
        break;
      }
    }

    const action = actionParts.filter(Boolean).map(capitalize).join(' ');

    const entry = SERVICE_MAP[serviceKey];
    if (entry) {
      const [display, color] = entry;
      return { service: display, color, action };
    }

    // Unknown service — derive display name from key
    return {
      service: capitalize(serviceKey),
      color: DEFAULT_TOOL_COLOR,
      action: action || capitalize(serviceKey),
    };
  }

  return { service: 'Tool', color: DEFAULT_TOOL_COLOR, action: rawName };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function durationMs(startIso: string, endIso: string): number {
  try {
    return new Date(endIso).getTime() - new Date(startIso).getTime();
  } catch {
    return 0;
  }
}

function tryUnescapeJson(text: string): string {
  try {
    const parsed = JSON.parse(text);
    if (typeof parsed === 'object' && parsed !== null && 'result' in parsed) {
      try {
        const inner = JSON.parse(parsed.result as string);
        return JSON.stringify(inner, null, 2);
      } catch {
        return String(parsed.result);
      }
    }
    return JSON.stringify(parsed, null, 2);
  } catch {
    return text;
  }
}

export function detectOutputError(output: string): boolean {
  try {
    const parsed = JSON.parse(output);
    if (typeof parsed === 'object' && parsed !== null && 'error' in parsed) {
      return true;
    }
  } catch {
    // Not JSON — check if the raw text starts with "Error:"
    if (output.startsWith('Error:') || output.startsWith('error:')) {
      return true;
    }
  }
  return false;
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}m ${Math.round(rem)}s`;
}

// ---------------------------------------------------------------------------
// Main parser
// ---------------------------------------------------------------------------

export function parseOtelTrajectory(
  spans: OtelSpan[],
  envType?: string,
  opts?: { modelHint?: string },
): ParsedTrajectory {
  // ios_cua emits the same screenshot-bearing OTel span shape as desktop cua
  // (chat spans + execute_tool spans with a base64 screenshot), so it renders
  // through the same parser — screenshots-between-actions, no phone viewer.
  if (envType === 'cua' || envType === 'ios_cua')
    return parseCuaTrajectory(spans);

  // Claude Code CLI A2A agents emit Anthropic stream-json events, not
  // OTel/OpenInference spans. The records have no `attributes` field, so
  // the normalizer below would crash. Detect by shape (first record is a
  // `system/init` event) and dispatch to the dedicated parser.
  if (looksLikeClaudeCliStreamJson(spans)) {
    return parseClaudeCliStreamJson(spans as unknown as ClaudeCliRecord[]);
  }

  // Gemini CLI A2A agents emit a different stream-json shape (first
  // record is `{type:'init', model:'…'}`, no subtype) — dispatch to its
  // own parser. Mutually exclusive with the Claude CLI shape above.
  if (looksLikeGeminiStreamJson(spans)) {
    return parseGeminiStreamJson(spans as unknown as GeminiRecord[]);
  }

  // Codex CLI A2A agents emit NDJSON from `codex exec --json`. The first
  // record is `{type:'thread.started', thread_id:'…'}`; subsequent
  // records are `{type:'item.completed', item:{type, …}}` with item.type
  // ∈ {agent_message, command_execution, mcp_tool_call, file_change,
  // web_search}. Distinct from both Claude CLI and Gemini shapes above.
  if (looksLikeCodexStreamJson(spans)) {
    return parseCodexStreamJson(spans as unknown as CodexRecord[]);
  }

  // OpenClaw A2A agents emit a flat list of {tool_call, tool_result, thinking,
  // final_response} dicts — no `init`/`result` envelope, no `model`, no
  // `.attributes`. Detect by shape and dispatch to its parser; the model is
  // threaded from the prompt-response via opts.modelHint.
  if (looksLikeOpenClawTrajectory(spans)) {
    return parseOpenClawTrajectory(spans as unknown as OpenClawRecord[], {
      modelHint: opts?.modelHint,
    });
  }

  // OpenCode CLI A2A agents emit `opencode run --format json` events — a flat
  // list of {type, timestamp, part} records (part.type ∈ step-start / tool /
  // text / step-finish). No init/result envelope, no `.attributes`; the model
  // is threaded via opts.modelHint. Distinct from all shapes above.
  if (looksLikeOpenCodeStreamJson(spans)) {
    return parseOpenCodeStreamJson(spans as unknown as OpenCodeRecord[], {
      modelHint: opts?.modelHint,
    });
  }

  // Pre-pass: synthesize gen_ai.* attributes on OpenInference-shaped spans
  // (emitted by openai_agents_sdk) so the existing classification + extraction
  // logic below works for both schemes.
  for (const span of spans) {
    normalizeOpenInferenceSpan(span);
  }

  let root: OtelSpan | null = null;
  const turns: OtelSpan[] = [];
  const toolSpans: OtelSpan[] = [];

  for (const span of spans) {
    // Defensive: a non-OTel record that slipped past the dispatchers above
    // (Claude CLI / Gemini / CUA) would otherwise crash here with
    // "Cannot read properties of undefined (reading 'gen_ai.operation.name')".
    // Skip rather than throw — better an empty timeline than a hard error.
    if (!span.attributes) continue;
    const opName = span.attributes['gen_ai.operation.name'];
    const noParent = span.parent_id === null || span.parent_id === undefined;

    // Root span detection — three accepted patterns:
    //   1. Explicit "chain" operation (claude_code / codex)
    //   2. Top-level invoke_agent without parent (openai_agents_sdk via OpenInference)
    //   3. Legacy: parent_id=null with no operation name set
    if (opName === 'chain') {
      root = span;
    } else if (opName === 'invoke_agent' && noParent && !root) {
      root = span;
    } else if (noParent && !opName && !root) {
      root = span;
    }

    // Turn spans: Claude uses name, OpenAI Agents uses gen_ai.operation.name
    if (span.name === 'claude.assistant.turn' || opName === 'chat') {
      turns.push(span);
    } else if (opName === 'execute_tool') {
      // OpenAI Agents harness: tool spans tagged by operation name
      toolSpans.push(span);
    } else if (span.parent_id && opName !== 'chain') {
      // Legacy: non-root, non-turn child spans are tools
      toolSpans.push(span);
    }
  }

  turns.sort((a, b) => a.start_time.localeCompare(b.start_time));
  toolSpans.sort((a, b) => a.start_time.localeCompare(b.start_time));

  // Build FIFO queues of tool results keyed by tool name
  const toolResultQueues: Record<string, ToolResultInfo[]> = {};
  for (const ts of toolSpans) {
    const name = ts.name;
    const completionRaw = ts.attributes['gen_ai.completion'] ?? '{}';
    let output: string;
    try {
      const completion = JSON.parse(completionRaw);
      output = tryUnescapeJson(String(completion.output ?? completionRaw));
    } catch {
      output = completionRaw;
    }

    if (!toolResultQueues[name]) toolResultQueues[name] = [];
    toolResultQueues[name].push({
      output,
      startTime: ts.start_time,
      endTime: ts.end_time,
      durationMs: durationMs(ts.start_time, ts.end_time),
      isError: ts.status.status_code === 'ERROR' || detectOutputError(output),
    });
  }

  // Extract metadata from root span
  let model = 'unknown';
  let userPrompt = '';
  let finalResponse = '';
  let totalDurationMs = 0;
  let numTurns = 0;

  if (root) {
    const attrs = root.attributes;
    model =
      attrs['langsmith.metadata.model'] ||
      attrs['gen_ai.request.model'] ||
      'unknown';
    totalDurationMs = parseInt(
      attrs['langsmith.metadata.duration_ms'] || '0',
      10,
    );
    if (!totalDurationMs && spans.length >= 2) {
      const sorted = [...spans].sort((a, b) =>
        a.start_time.localeCompare(b.start_time),
      );
      totalDurationMs = durationMs(
        sorted[0]!.start_time,
        sorted[sorted.length - 1]!.end_time,
      );
    }
    numTurns = parseInt(attrs['langsmith.metadata.num_turns'] || '0', 10);
    if (!numTurns) numTurns = turns.length;

    try {
      const promptData = JSON.parse(attrs['gen_ai.prompt'] || '{}');
      userPrompt = promptData.prompt || '';
      if (!userPrompt && Array.isArray(promptData.messages)) {
        const userMsg = promptData.messages.find(
          (m: { role?: string }) => m.role === 'user',
        );
        if (userMsg) userPrompt = userMsg.content || '';
      }
    } catch {
      /* ignore */
    }

    try {
      const compData = JSON.parse(attrs['gen_ai.completion'] || '{}');
      const content = compData.content;
      if (Array.isArray(content)) {
        for (let i = content.length - 1; i >= 0; i--) {
          const block = content[i];
          if (block?.type === 'text') {
            finalResponse = block.text || '';
            break;
          }
        }
      }
    } catch {
      /* ignore */
    }
  }

  // Walk turns to build timeline
  const events: TrajectoryEvent[] = [];

  for (const span of turns) {
    const completionRaw = span.attributes['gen_ai.completion'] ?? '{}';
    let content: unknown[];
    try {
      const comp = JSON.parse(completionRaw);
      content = Array.isArray(comp.content)
        ? comp.content
        : Array.isArray(comp)
        ? comp
        : [];
    } catch {
      continue;
    }

    for (const block of content) {
      if (typeof block !== 'object' || block === null) continue;
      const b = block as Record<string, unknown>;
      const btype = b.type as string;

      if (btype === 'thinking') {
        events.push({ type: 'thinking', text: (b.thinking as string) || '' });
      } else if (btype === 'text') {
        events.push({ type: 'text', text: (b.text as string) || '' });
      } else if (btype === 'tool_use') {
        const toolName = (b.name as string) || '';
        const toolId = (b.id as string) || '';
        const toolInput = (b.input as Record<string, unknown>) || {};

        let resultInfo: ToolResultInfo | null = null;
        const queue = toolResultQueues[toolName];
        if (queue && queue.length > 0) {
          resultInfo = queue.shift()!;
        }

        events.push({
          type: 'tool_call',
          id: toolId,
          name: toolName,
          input: toolInput,
          result: resultInfo,
        });
      }
    }
  }

  // Fallback final response
  if (!finalResponse) {
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      if (e && e.type === 'text' && e.text.trim()) {
        finalResponse = e.text;
        break;
      }
    }
  }

  // Service counts
  const serviceCounts: Record<string, number> = {};
  let toolCallCount = 0;
  for (const e of events) {
    if (e.type === 'tool_call') {
      toolCallCount++;
      const { service } = parseToolName(e.name);
      serviceCounts[service] = (serviceCounts[service] || 0) + 1;
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

// ---------------------------------------------------------------------------
// CUA trajectory parser
// ---------------------------------------------------------------------------

function parseCuaTrajectory(spans: OtelSpan[]): ParsedTrajectory {
  const sorted = [...spans].sort((a, b) =>
    a.start_time.localeCompare(b.start_time),
  );

  const chatSpans = sorted.filter(s => s.name === 'chat');
  const toolSpans = sorted.filter(s => s.name === 'execute_tool');

  // Starting-state screenshot (harness `initial_screenshot` span) — the screen
  // before any action. Rendered ahead of the events.
  let initialScreenshot: string | undefined;
  const initSpan = sorted.find(s => s.name === 'initial_screenshot');
  if (initSpan) {
    try {
      const shot = JSON.parse(
        initSpan.attributes['gen_ai.completion'] ?? '{}',
      ).screenshot;
      if (typeof shot === 'string' && shot) initialScreenshot = shot;
    } catch {
      /* no starting frame */
    }
  }

  // Build a FIFO queue of tool results
  const toolResultQueue: ToolResultInfo[] = toolSpans.map(ts => {
    const completionRaw = ts.attributes['gen_ai.completion'] ?? '{}';
    let output: string;
    let screenshot: string | undefined;
    try {
      const parsed = JSON.parse(completionRaw);
      screenshot =
        typeof parsed.screenshot === 'string' ? parsed.screenshot : undefined;
      // Exclude screenshot from the displayed output
      if (screenshot) {
        const { screenshot: _, ...rest } = parsed;
        output = JSON.stringify(rest, null, 2);
      } else {
        output = JSON.stringify(parsed, null, 2);
      }
    } catch {
      output = completionRaw;
    }
    // Harness-baked annotated frame (the pre-action frame with the gesture
    // marker drawn into the pixels). Shown directly as the action-location frame.
    const annotatedRaw = ts.attributes['cua.acted_screenshot_annotated'];
    const actedScreenshotAnnotated =
      typeof annotatedRaw === 'string' && annotatedRaw
        ? annotatedRaw
        : undefined;
    return {
      output,
      startTime: ts.start_time,
      endTime: ts.end_time,
      durationMs: durationMs(ts.start_time, ts.end_time),
      isError: ts.status.status_code === 'ERROR' || detectOutputError(output),
      screenshot,
      actedScreenshotAnnotated,
    };
  });

  // Metadata from first chat span
  const firstChat = chatSpans[0];
  const model = firstChat?.attributes['gen_ai.request.model'] ?? 'unknown';

  // Total duration from first to last span
  const firstSpan = sorted[0];
  const lastSpan = sorted[sorted.length - 1];
  const totalDurationMs =
    firstSpan && lastSpan
      ? durationMs(firstSpan.start_time, lastSpan.end_time)
      : 0;

  // User prompt from first span
  let userPrompt = '';
  if (firstChat) {
    try {
      const promptData = JSON.parse(
        firstChat.attributes['gen_ai.prompt'] || '{}',
      );
      userPrompt = promptData.prompt || '';
    } catch {
      /* ignore */
    }
  }

  // Walk chat spans to build events
  const events: TrajectoryEvent[] = [];

  for (const span of chatSpans) {
    const completionRaw = span.attributes['gen_ai.completion'] ?? '[]';
    let content: unknown[];
    try {
      const comp = JSON.parse(completionRaw);
      // CUA format: completion is a JSON array directly (not {content: [...]})
      content = Array.isArray(comp)
        ? comp
        : Array.isArray(comp.content)
        ? comp.content
        : [];
    } catch {
      continue;
    }

    for (const block of content) {
      if (typeof block !== 'object' || block === null) continue;
      const b = block as Record<string, unknown>;
      const btype = b.type as string;

      if (btype === 'thinking') {
        events.push({
          type: 'thinking',
          text: (b.thinking as string) || '',
        });
      } else if (btype === 'text') {
        events.push({ type: 'text', text: (b.text as string) || '' });
      } else if (btype === 'tool_use') {
        const toolName = (b.name as string) || 'computer';
        const toolId = (b.id as string) || '';
        const toolInput = (b.input as Record<string, unknown>) || {};

        // Pair with next tool result from queue
        const resultInfo =
          toolResultQueue.length > 0 ? toolResultQueue.shift()! : null;

        events.push({
          type: 'tool_call',
          id: toolId,
          name: toolName,
          input: toolInput,
          result: resultInfo,
        });
      }
    }
  }

  // Final response from last text event
  let finalResponse = '';
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e && e.type === 'text' && e.text.trim()) {
      finalResponse = e.text;
      break;
    }
  }

  // Service counts
  const serviceCounts: Record<string, number> = {};
  let toolCallCount = 0;
  for (const e of events) {
    if (e.type === 'tool_call') {
      toolCallCount++;
      const { service } = parseToolName(e.name);
      serviceCounts[service] = (serviceCounts[service] || 0) + 1;
    }
  }

  const steps = groupCuaEventsIntoSteps(events, finalResponse);

  return {
    model,
    userPrompt,
    finalResponse,
    events,
    steps,
    totalDurationMs,
    numTurns: chatSpans.length,
    toolCallCount,
    serviceCounts,
    initialScreenshot,
  };
}

// ---------------------------------------------------------------------------
// Step grouping
// ---------------------------------------------------------------------------

export function groupEventsIntoSteps(
  events: TrajectoryEvent[],
  finalResponse: string,
): TrajectoryStep[] {
  const steps: TrajectoryStep[] = [];
  let currentEvents: TrajectoryEvent[] = [];
  let currentLabel = '';

  for (const event of events) {
    if (event.type === 'text') {
      // Skip the final response — it's rendered separately
      if (event.text === finalResponse) continue;

      // A new text block starts a new step — flush the previous one
      if (currentLabel || currentEvents.length > 0) {
        steps.push({
          label: currentLabel || 'Initial reasoning',
          events: currentEvents,
          toolCount: currentEvents.filter(e => e.type === 'tool_call').length,
        });
      }

      currentLabel = event.text;
      currentEvents = [];
    } else {
      currentEvents.push(event);
    }
  }

  // Flush remaining events
  if (currentLabel || currentEvents.length > 0) {
    steps.push({
      label: currentLabel || 'Initial reasoning',
      events: currentEvents,
      toolCount: currentEvents.filter(e => e.type === 'tool_call').length,
    });
  }

  return steps;
}

function firstLine(text: string, maxLen = 100): string {
  const line = (text || '')
    .split('\n')
    .map(l => l.trim())
    .find(l => l.length > 0);
  if (!line) return '';
  return line.length > maxLen ? line.slice(0, maxLen) + '…' : line;
}

// The gemini passthrough prepends garbled fragments to the model's visible text on reasoning-heavy turns
// (a leaked "thought\n" marker, a leading non-Latin run, redaction tails). Clean them so step labels are
// human-readable. Conservative — does NOT strip ambiguous glued-Latin prefixes (e.g. "gMon"), which risk
// eating real words (iPhone, eBay); those fall through to the synthesized label below.
const INTENT_MARKERS = [
  '[Earlier thought process continues internally...]',
  '…[truncated]',
  '...[truncated]',
];
function cleanIntentText(text: string): string {
  if (!text) return '';
  let t = text;
  for (const m of INTENT_MARKERS) t = t.split(m).join(' ');
  t = t.replace(/\s+/g, ' ').trim();
  t = t.replace(/^\S{0,12}?thought[\s:]+/, ''); // glued "<junk>thought " boundary
  t = t.replace(/^[^\x00-\x7F]+\s*/, ''); // leading non-ASCII junk run
  t = t.replace(/\s*[^\x00-\x7F]+$/, ''); // trailing non-ASCII junk run (proxy appends it too)
  return t.trim();
}
function isLowSignalIntent(text: string): boolean {
  if (text.length < 8) return true;
  if (
    text.includes('.png') ||
    text.includes('.jpg') ||
    text.includes('omitted]')
  )
    return true;
  if ((text.match(/]/g)?.length ?? 0) > (text.match(/\[/g)?.length ?? 0))
    return true;
  if (/(.)\1{5,}/.test(text)) return true; // long single-char filler run
  const letters = text.match(/[A-Za-z]/g)?.length ?? 0;
  return letters / text.length < 0.45;
}
// A human-readable step label from model text, or '' when the text is too garbled
// (caller then falls back to the action name).
function cleanLabel(text: string): string {
  const c = cleanIntentText(text);
  return c && !isLowSignalIntent(c) ? firstLine(c) : '';
}

export function groupCuaEventsIntoSteps(
  events: TrajectoryEvent[],
  finalResponse: string,
): TrajectoryStep[] {
  const steps: TrajectoryStep[] = [];
  let pending: TrajectoryEvent[] = [];
  let lastText = '';
  let lastThinking = '';

  for (const event of events) {
    if (event.type === 'text') {
      if (event.text === finalResponse) continue;
      lastText = event.text;
      pending.push(event);
    } else if (event.type === 'thinking') {
      lastThinking = event.text;
      pending.push(event);
    } else {
      const label =
        cleanLabel(lastText) ||
        cleanLabel(lastThinking) ||
        parseToolName(event.name).action ||
        'Action';
      steps.push({ label, events: [...pending, event], toolCount: 1 });
      pending = [];
      lastText = '';
      lastThinking = '';
    }
  }

  if (pending.length > 0) {
    steps.push({
      label: cleanLabel(lastText) || cleanLabel(lastThinking) || 'Reasoning',
      events: pending,
      toolCount: 0,
    });
  }

  return steps;
}
