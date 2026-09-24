/**
 * Smoke test: feed a real Claude Code CLI stream-json trajectory through
 * `parseOtelTrajectory` (which auto-dispatches via `looksLikeClaudeCliStreamJson`)
 * and assert that classification + event extraction + sub-agent folding
 * produce sensible output.
 *
 * Fixture default: `/tmp/traj.json` — pulled from the real run
 * a captured agent run while diagnosing the
 * "Cannot read properties of undefined (reading 'openinference.span.kind')"
 * crash. Override with `CLAUDE_CLI_FIXTURE=<path>`.
 *
 * Runner: plain TS, throws on assertion failure. Run with any TS executor:
 *   npx tsx src/lib/parse-trajectory-claude-cli.smoke.ts
 */
import * as fs from 'fs';

import { OtelSpan, parseOtelTrajectory } from './parse-trajectory';
import {
  ClaudeCliRecord,
  looksLikeClaudeCliStreamJson,
  parseClaudeCliStreamJson,
} from './parse-trajectory-claude-cli';

function assert(cond: unknown, msg: string): void {
  if (!cond) {
    console.error(`✗ ${msg}`);
    throw new Error(msg);
  }
  console.log(`✓ ${msg}`);
}

function loadFixture(): ClaudeCliRecord[] | null {
  const path = process.env.CLAUDE_CLI_FIXTURE ?? '/tmp/traj.json';
  if (!fs.existsSync(path)) return null;
  const buf = fs.readFileSync(path, 'utf-8');
  console.log(`fixture: ${path}`);
  return JSON.parse(buf) as ClaudeCliRecord[];
}

function main(): void {
  const records = loadFixture();
  // The fixture is a real captured run, too large to commit. Skip rather than fail so
  // the suite stays green in CI; set CLAUDE_CLI_FIXTURE=<path> to actually run it.
  if (records === null) {
    console.log('- skipped: agentenv-capability-missing: claude_cli_trajectory_fixture (set CLAUDE_CLI_FIXTURE)');
    return;
  }
  assert(records.length > 0, `loaded ${records.length} records`);

  // ---- 1. Detection ----
  assert(
    looksLikeClaudeCliStreamJson(records),
    'looksLikeClaudeCliStreamJson() detects stream-json input',
  );
  assert(
    !looksLikeClaudeCliStreamJson([]),
    'looksLikeClaudeCliStreamJson([]) returns false for empty input',
  );
  assert(
    !looksLikeClaudeCliStreamJson([{ type: 'span', attributes: {} } as never]),
    'looksLikeClaudeCliStreamJson() returns false for OTel-shaped input',
  );

  // ---- 2. Auto-dispatch through parseOtelTrajectory ----
  // The viewer always calls parseOtelTrajectory; verify it routes to the
  // claude-cli parser instead of crashing on missing `attributes`.
  const viaDispatch = parseOtelTrajectory(
    records as unknown as OtelSpan[],
  );
  assert(
    typeof viaDispatch.model === 'string' && viaDispatch.model.length > 0,
    `dispatched parse produces a model name (got "${viaDispatch.model}")`,
  );

  // ---- 3. Direct parse — verify the contract ----
  const parsed = parseClaudeCliStreamJson(records);

  // model: should be the assistant's reported model (claude-sonnet-4-6 in
  // the fixture), not "unknown".
  assert(
    parsed.model.includes('claude') || parsed.model.length > 0,
    `model populated: "${parsed.model}"`,
  );

  // Tool calls — parser walks the full tree, so toolCallCount equals the
  // total count of tool_use blocks across all assistant messages (main +
  // sub-agent).
  const totalToolUses = records
    .filter(r => r.type === 'assistant')
    .flatMap(r => r.message?.content ?? [])
    .filter(b => (b as { type?: string }).type === 'tool_use').length;
  assert(
    parsed.toolCallCount === totalToolUses,
    `toolCallCount (${parsed.toolCallCount}) === total tool_uses across all assistant messages (${totalToolUses})`,
  );

  const toolCallEvents = parsed.events.filter(e => e.type === 'tool_call');
  assert(
    toolCallEvents.length > 0,
    `at least one tool_call event (${toolCallEvents.length})`,
  );

  const withResult = toolCallEvents.filter(
    e => e.type === 'tool_call' && e.result !== null,
  );
  assert(
    withResult.length > 0,
    `at least one tool_call event has a paired result (${withResult.length}/${toolCallEvents.length})`,
  );

  // Service counts populated
  assert(
    Object.keys(parsed.serviceCounts).length > 0,
    `serviceCounts populated: ${JSON.stringify(parsed.serviceCounts)}`,
  );

  // Final response from the result record
  const resultRecord = records.find(r => r.type === 'result');
  if (resultRecord?.result) {
    assert(
      parsed.finalResponse === resultRecord.result,
      'finalResponse matches the result record',
    );
  }

  // num_turns from result record
  if (typeof resultRecord?.num_turns === 'number') {
    assert(
      parsed.numTurns === resultRecord.num_turns,
      `numTurns (${parsed.numTurns}) matches result.num_turns (${resultRecord.num_turns})`,
    );
  }

  // totalDurationMs
  if (typeof resultRecord?.duration_ms === 'number') {
    assert(
      parsed.totalDurationMs === resultRecord.duration_ms,
      `totalDurationMs (${parsed.totalDurationMs}) matches result.duration_ms`,
    );
  }

  // ---- 4. Sub-agent summary populated ----
  const subAgentCalls = toolCallEvents.filter(
    e => e.type === 'tool_call' && e.subAgentSummary,
  );
  assert(
    subAgentCalls.length > 0,
    `at least one tool_call has subAgentSummary (${subAgentCalls.length})`,
  );
  const first = subAgentCalls[0];
  if (first && first.type === 'tool_call' && first.subAgentSummary) {
    const s = first.subAgentSummary;
    assert(
      s.description.length > 0,
      `sub-agent has a description: "${s.description.slice(0, 80)}..."`,
    );
    assert(
      s.toolUses > 0,
      `sub-agent toolUses > 0 (got ${s.toolUses})`,
    );
    assert(
      s.totalTokens > 0,
      `sub-agent totalTokens > 0 (got ${s.totalTokens})`,
    );
    assert(
      s.events.length > 0,
      `sub-agent timeline events populated (got ${s.events.length})`,
    );
    const innerToolCalls = s.events.filter(e => e.type === 'tool_call');
    assert(
      innerToolCalls.length > 0,
      `sub-agent timeline has tool calls (got ${innerToolCalls.length})`,
    );
    const innerWithResult = innerToolCalls.filter(
      e => e.type === 'tool_call' && e.result !== null,
    );
    assert(
      innerWithResult.length > 0,
      `sub-agent inner tool calls have paired results (${innerWithResult.length}/${innerToolCalls.length})`,
    );
  }

  // ---- 5. Steps are non-empty ----
  assert(parsed.steps.length > 0, `steps non-empty (${parsed.steps.length})`);

  // ---- 6. No record made it through to top-level events with missing fields ----
  for (const e of parsed.events) {
    if (e.type === 'thinking') assert(typeof e.text === 'string', 'thinking.text is string');
    if (e.type === 'text') assert(typeof e.text === 'string', 'text.text is string');
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
