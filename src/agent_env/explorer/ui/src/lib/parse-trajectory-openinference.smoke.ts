/**
 * Smoke test: feed a real openai_agents_sdk OTEL trajectory through the
 * combined `normalizeOpenInferenceSpan` + `parseOtelTrajectory` pipeline
 * and assert that classification + event extraction produce sensible
 * output.
 *
 * The fixture is a 22-span trajectory captured from a 3-sub-agent kimi-k2p5
 * run via `/tmp/smoke_otel_bridge.py`. It exercises the OpenInference
 * adapter on every span kind (AGENT/CHAIN/TOOL/LLM), nested sub-agent
 * runs, and tool calls.
 *
 * Runner: this file is plain TS that throws on assertion failure. Run with
 * any TS executor — e.g. `npx tsx src/lib/parse-trajectory-openinference.smoke.ts`
 * — or compile + run via `tsc && node dist/.../smoke.js`. There's no test
 * framework wired into this package; the smoke is documentation as much
 * as automation.
 */
import * as fs from 'fs';
import * as path from 'path';

import { OtelSpan, parseOtelTrajectory } from './parse-trajectory';
import { normalizeOpenInferenceSpan } from './parse-trajectory-openinference';

function assert(cond: unknown, msg: string): void {
  if (!cond) {
    console.error(`✗ ${msg}`);
    throw new Error(msg);
  }
  console.log(`✓ ${msg}`);
}

function loadFixture(): OtelSpan[] | null {
  // OPENINFERENCE_FIXTURE only. This used to fall back to two paths in ~/Downloads,
  // which made "did this test run?" depend on what happened to be in one developer's
  // Downloads folder — the same machine-dependence the explorer test fixture was
  // fixed for.
  const candidates = [process.env.OPENINFERENCE_FIXTURE].filter(
    (p): p is string => Boolean(p),
  );
  for (const c of candidates) {
    try {
      const buf = fs.readFileSync(c, 'utf-8');
      console.log(`fixture: ${c}`);
      return JSON.parse(buf) as OtelSpan[];
    } catch {
      /* try next */
    }
  }
  return null;
}

function main(): void {
  const spans = loadFixture();
  // The fixture is a real captured span dump, too large to commit. Skip rather than fail
  // so the suite stays green in CI; set OPENINFERENCE_FIXTURE=<path> to actually run it.
  if (spans === null) {
    console.log('- skipped: agentenv-capability-missing: openinference_trajectory_fixture (set OPENINFERENCE_FIXTURE)');
    return;
  }
  assert(spans.length > 0, `loaded ${spans.length} spans`);

  // ---- 1. normalization adds gen_ai.* attributes in place ----
  const llmSpan = spans.find(
    s => s.attributes['openinference.span.kind'] === 'LLM',
  );
  const toolSpan = spans.find(
    s => s.attributes['openinference.span.kind'] === 'TOOL',
  );
  assert(!!llmSpan, 'fixture has at least one LLM span');
  assert(!!toolSpan, 'fixture has at least one TOOL span');

  if (llmSpan) {
    normalizeOpenInferenceSpan(llmSpan);
    assert(
      llmSpan.attributes['gen_ai.operation.name'] === 'chat',
      'LLM span gets gen_ai.operation.name=chat',
    );
    assert(
      !!llmSpan.attributes['gen_ai.request.model'],
      'LLM span gets gen_ai.request.model from llm.model_name',
    );
    assert(
      !!llmSpan.attributes['gen_ai.completion'],
      'LLM span gets gen_ai.completion JSON',
    );
    const completion = JSON.parse(
      llmSpan.attributes['gen_ai.completion'] as string,
    );
    assert(
      Array.isArray(completion.content),
      'gen_ai.completion has a content array',
    );
  }

  if (toolSpan) {
    normalizeOpenInferenceSpan(toolSpan);
    assert(
      toolSpan.attributes['gen_ai.operation.name'] === 'execute_tool',
      'TOOL span gets gen_ai.operation.name=execute_tool',
    );
    assert(
      !!toolSpan.attributes['gen_ai.tool.name'],
      'TOOL span gets gen_ai.tool.name from tool.name',
    );
  }

  // ---- 2. end-to-end parse produces a usable trajectory ----
  const result = parseOtelTrajectory(spans);
  console.log(
    `\nparsed: model=${result.model} turns=${result.numTurns} ` +
      `events=${result.events.length} tools=${result.toolCallCount}`,
  );
  console.log(
    `  finalResponse[:120]: ${(result.finalResponse || '').slice(0, 120)}`,
  );

  assert(result.model !== 'unknown', `model resolved (got: ${result.model})`);
  assert(
    result.numTurns > 0,
    `at least one turn parsed (got: ${result.numTurns})`,
  );
  assert(
    result.toolCallCount > 0,
    `at least one tool call parsed (got: ${result.toolCallCount})`,
  );
  assert(
    result.events.length > 0,
    `at least one event extracted (got: ${result.events.length})`,
  );

  // ---- 3. multi-tool-call extraction terminates (OOM regression guard) ----
  //
  // An earlier version built the inner-loop lookup key once outside the
  // while-condition with `${j}` baked in at j=0; the loop never advanced
  // and `content` grew unbounded → tab OOM. Synthesize a span with 4
  // tool_calls and assert we get exactly 4 tool_use blocks.
  const multiToolSpan: OtelSpan = {
    name: 'generation',
    context: { trace_id: '0xtrace', span_id: '0xspan' },
    kind: 'INTERNAL',
    parent_id: '0xparent',
    start_time: '2026-05-04T00:00:00.000Z',
    end_time: '2026-05-04T00:00:01.000Z',
    status: { status_code: 'OK' },
    attributes: {
      'openinference.span.kind': 'LLM',
      'llm.model_name': 'fireworks_ai/kimi-k2p5',
      'llm.output_messages.0.message.role': 'assistant',
      'llm.output_messages.0.message.content': '',
      'llm.output_messages.0.message.tool_calls.0.tool_call.id': 'call_a',
      'llm.output_messages.0.message.tool_calls.0.tool_call.function.name':
        'tool_a',
      'llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments':
        '{"x":1}',
      'llm.output_messages.0.message.tool_calls.1.tool_call.id': 'call_b',
      'llm.output_messages.0.message.tool_calls.1.tool_call.function.name':
        'tool_b',
      'llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments':
        '{"y":2}',
      'llm.output_messages.0.message.tool_calls.2.tool_call.id': 'call_c',
      'llm.output_messages.0.message.tool_calls.2.tool_call.function.name':
        'tool_c',
      'llm.output_messages.0.message.tool_calls.2.tool_call.function.arguments':
        '{"z":3}',
      'llm.output_messages.0.message.tool_calls.3.tool_call.id': 'call_d',
      'llm.output_messages.0.message.tool_calls.3.tool_call.function.name':
        'tool_d',
      'llm.output_messages.0.message.tool_calls.3.tool_call.function.arguments':
        '{"w":4}',
    },
    events: [],
    links: [],
    resource: { attributes: {} },
  };
  normalizeOpenInferenceSpan(multiToolSpan);
  const completion = JSON.parse(
    multiToolSpan.attributes['gen_ai.completion'] as string,
  );
  const toolUses = (
    completion.content as Array<{ type: string; name?: string }>
  ).filter(b => b.type === 'tool_use');
  assert(
    toolUses.length === 4,
    `4-tool-call span produces 4 tool_use blocks (got ${toolUses.length})`,
  );
  const expectedNames = ['tool_a', 'tool_b', 'tool_c', 'tool_d'];
  assert(
    JSON.stringify(toolUses.map(b => b.name)) === JSON.stringify(expectedNames),
    'tool_use blocks preserve order: ' + expectedNames.join(', '),
  );

  // ---- 4. preserves pure-OTEL spans (regression guard) ----
  const fakeOtel: OtelSpan = {
    name: 'turn',
    context: { trace_id: '0xabc', span_id: '0xdef' },
    kind: 'INTERNAL',
    parent_id: null,
    start_time: '2026-05-04T00:00:00.000Z',
    end_time: '2026-05-04T00:00:01.000Z',
    status: { status_code: 'OK' },
    attributes: {
      'gen_ai.operation.name': 'chat',
      'gen_ai.request.model': 'claude-opus-4-7',
      'gen_ai.completion': JSON.stringify({
        content: [{ type: 'text', text: 'hello' }],
      }),
    },
    events: [],
    links: [],
    resource: { attributes: {} },
  };
  const before = JSON.stringify(fakeOtel.attributes);
  normalizeOpenInferenceSpan(fakeOtel);
  assert(
    JSON.stringify(fakeOtel.attributes) === before,
    'pure-OTEL span passes through normalization unchanged',
  );

  console.log('\n✓ all OpenInference adapter assertions passed');
}

try {
  main();
  process.exit(0);
} catch {
  process.exit(1);
}
