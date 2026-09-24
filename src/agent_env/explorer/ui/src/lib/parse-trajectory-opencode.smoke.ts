/**
 * Smoke test: feed an OpenCode CLI `opencode run --format json` trajectory
 * through `parseOtelTrajectory` (which auto-dispatches via
 * `looksLikeOpenCodeStreamJson`) and assert the event extraction contract.
 *
 * Default fixture is inline + synthetic but faithful to the real record shape
 * (verified against the live run `task-v2env-186-778c8c2c-opencode-y5hm9olj`:
 * records are `{type, timestamp, sessionID, part}`; tool calls are
 * self-contained with `part.state={status,input,output,time}`; a turn ends on a
 * `step_finish` whose `part.reason === 'stop'`). Point at a real pulled
 * trajectory with `OPENCODE_FIXTURE=<path>`.
 *
 * Runner: plain TS, throws on assertion failure. Run with any TS executor:
 *   npx tsx src/lib/parse-trajectory-opencode.smoke.ts
 */
import * as fs from 'fs';

import { OtelSpan, parseOtelTrajectory } from './parse-trajectory';
import {
  OpenCodeRecord,
  looksLikeOpenCodeStreamJson,
  parseOpenCodeStreamJson,
} from './parse-trajectory-opencode';

function assert(cond: unknown, msg: string): void {
  if (!cond) {
    console.error(`✗ ${msg}`);
    throw new Error(msg);
  }
  console.log(`✓ ${msg}`);
}

// A faithful miniature of the real stream: one turn that reasons, calls two
// tools (one ok, one error), emits assistant text, and ends with reason 'stop'.
const INLINE_FIXTURE: OpenCodeRecord[] = [
  {
    type: 'step_start',
    timestamp: 1000,
    sessionID: 'ses_abc',
    part: { type: 'step-start' },
  },
  {
    type: 'reasoning',
    timestamp: 1050,
    sessionID: 'ses_abc',
    part: { type: 'reasoning', text: 'Let me gather the emails first.' },
  },
  {
    type: 'tool_use',
    timestamp: 1100,
    sessionID: 'ses_abc',
    part: {
      type: 'tool',
      tool: 'mcp_ea2100eb_search_emails',
      callID: 'call_1',
      state: {
        status: 'completed',
        input: { query: 'Sunbelt' },
        output: '8 emails found',
        time: { start: 1100, end: 1151 },
      },
    },
  },
  {
    type: 'tool_use',
    timestamp: 1200,
    sessionID: 'ses_abc',
    part: {
      type: 'tool',
      tool: 'mcp_ea2100eb_crm_search_companies',
      callID: 'call_2',
      state: {
        status: 'error',
        input: { name: 'Palmetto' },
        output: 'Error: company not found',
        time: { start: 1200, end: 1210 },
      },
    },
  },
  {
    type: 'step_finish',
    timestamp: 1250,
    sessionID: 'ses_abc',
    part: {
      type: 'step-finish',
      reason: 'tool-calls',
      providerID: 'anthropic',
      modelID: 'claude-sonnet-4-6',
    },
  },
  {
    type: 'step_start',
    timestamp: 1300,
    sessionID: 'ses_abc',
    part: { type: 'step-start' },
  },
  {
    type: 'text',
    timestamp: 1400,
    sessionID: 'ses_abc',
    part: { type: 'text', text: 'Here is the full summary of what happened.' },
  },
  {
    type: 'step_finish',
    timestamp: 1500,
    sessionID: 'ses_abc',
    part: {
      type: 'step-finish',
      reason: 'stop',
      providerID: 'anthropic',
      modelID: 'claude-sonnet-4-6',
    },
  },
];

function loadFixture(): OpenCodeRecord[] {
  const path = process.env.OPENCODE_FIXTURE;
  if (!path) {
    console.log('fixture: <inline synthetic>');
    return INLINE_FIXTURE;
  }
  console.log(`fixture: ${path}`);
  return JSON.parse(fs.readFileSync(path, 'utf-8')) as OpenCodeRecord[];
}

function main(): void {
  const records = loadFixture();
  assert(records.length > 0, `loaded ${records.length} records`);

  // ---- 1. Detection ----
  assert(
    looksLikeOpenCodeStreamJson(records),
    'looksLikeOpenCodeStreamJson() detects opencode stream-json input',
  );
  assert(
    !looksLikeOpenCodeStreamJson([]),
    'looksLikeOpenCodeStreamJson([]) returns false for empty input',
  );
  assert(
    !looksLikeOpenCodeStreamJson([{ type: 'span', attributes: {} } as never]),
    'looksLikeOpenCodeStreamJson() returns false for OTel-shaped input',
  );
  assert(
    !looksLikeOpenCodeStreamJson([
      { type: 'system', subtype: 'init' } as never,
    ]),
    'looksLikeOpenCodeStreamJson() returns false for claude-cli input (no part)',
  );

  // ---- 2. Auto-dispatch through parseOtelTrajectory ----
  // The viewer always calls parseOtelTrajectory; verify it routes here with a
  // model hint (opencode's stream carries no top-level model envelope).
  const viaDispatch = parseOtelTrajectory(
    records as unknown as OtelSpan[],
    undefined,
    { modelHint: 'anthropic/claude-sonnet-4-6' },
  );
  assert(
    viaDispatch.model === 'anthropic/claude-sonnet-4-6',
    `dispatched parse honours modelHint (got "${viaDispatch.model}")`,
  );
  assert(
    viaDispatch.toolCallCount > 0,
    `dispatched parse extracts tool calls (${viaDispatch.toolCallCount})`,
  );

  // ---- 3. Direct parse — verify the contract ----
  const parsed = parseOpenCodeStreamJson(records);

  // toolCallCount == number of tool_use / part.type==='tool' records
  const totalTools = records.filter(
    r => r.type === 'tool_use' || r.part?.type === 'tool',
  ).length;
  assert(
    parsed.toolCallCount === totalTools,
    `toolCallCount (${parsed.toolCallCount}) === tool records (${totalTools})`,
  );

  const toolCallEvents = parsed.events.filter(e => e.type === 'tool_call');
  assert(
    toolCallEvents.length === totalTools,
    `tool_call events (${toolCallEvents.length}) === tool records (${totalTools})`,
  );

  // Self-contained results: every tool with a state.output gets a result.
  const withResult = toolCallEvents.filter(
    e => e.type === 'tool_call' && e.result !== null,
  );
  assert(
    withResult.length > 0,
    `tool_call events carry self-contained results (${withResult.length}/${toolCallEvents.length})`,
  );

  // Error status → isError on the result.
  const errorResults = toolCallEvents.filter(
    e => e.type === 'tool_call' && e.result?.isError,
  );
  if (records.some(r => r.part?.state?.status === 'error')) {
    assert(
      errorResults.length > 0,
      `status:'error' tools flagged isError (${errorResults.length})`,
    );
  }

  // serviceCounts populated
  assert(
    Object.keys(parsed.serviceCounts).length > 0,
    `serviceCounts populated: ${JSON.stringify(parsed.serviceCounts)}`,
  );

  // MCP tools must classify by their real service, not all collapse to 'Tool':
  // opencode names them `mcp_<server>_<tool>`, so the parser canonicalizes to
  // the `mcp__server__tool` shape parseToolName understands.
  if (records.some(r => (r.part?.tool ?? '').startsWith('mcp_'))) {
    const nonToolServices = Object.keys(parsed.serviceCounts).filter(
      s => s !== 'Tool',
    );
    assert(
      nonToolServices.length > 0,
      `MCP tools classify by real service, not just 'Tool': ${JSON.stringify(
        parsed.serviceCounts,
      )}`,
    );
  }

  // numTurns == count of step_finish{reason:'stop'} (fallback 1)
  const stopCount = records.filter(
    r =>
      (r.type === 'step_finish' || r.part?.type === 'step-finish') &&
      r.part?.reason === 'stop',
  ).length;
  assert(
    parsed.numTurns === (stopCount || (parsed.events.length ? 1 : 0)),
    `numTurns (${parsed.numTurns}) === stop-boundary count (${stopCount})`,
  );

  // finalResponse == last non-empty text event
  const lastText = [...parsed.events]
    .reverse()
    .find(e => e.type === 'text' && e.text.trim());
  if (lastText && lastText.type === 'text') {
    assert(
      parsed.finalResponse === lastText.text,
      'finalResponse is the last non-empty assistant text',
    );
  }

  // reasoning part → thinking event (exercised by the inline fixture)
  if (records.some(r => r.part?.type === 'reasoning')) {
    assert(
      parsed.events.some(e => e.type === 'thinking'),
      'reasoning part becomes a thinking event',
    );
  }

  // duration from first/last record timestamp
  const stamps = records
    .map(r => r.timestamp)
    .filter((n): n is number => typeof n === 'number');
  const firstStamp = stamps[0];
  const lastStamp = stamps[stamps.length - 1];
  if (firstStamp !== undefined && lastStamp !== undefined) {
    assert(
      parsed.totalDurationMs === lastStamp - firstStamp,
      `totalDurationMs (${parsed.totalDurationMs}) === last-first timestamp`,
    );
  }

  // steps non-empty + field types sane
  assert(parsed.steps.length > 0, `steps non-empty (${parsed.steps.length})`);
  for (const e of parsed.events) {
    if (e.type === 'thinking' || e.type === 'text')
      assert(typeof e.text === 'string', `${e.type}.text is string`);
    if (e.type === 'tool_call') {
      assert(typeof e.name === 'string', 'tool_call.name is string');
      assert(
        e.input === null || typeof e.input === 'object',
        'tool_call.input is object',
      );
    }
  }

  console.log('\nAll assertions passed.');
}

main();
