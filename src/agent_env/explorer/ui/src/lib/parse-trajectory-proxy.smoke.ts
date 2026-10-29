/**
 * Smoke test for a ProxyAgent trajectory in `parseOtelTrajectory`: the spans the proxy writes, `chat` for each
 * recorded model call, `execute_tool` for each action and `chain` for the turn, with none of an exporter's span
 * context. A span with no status still renders.
 *
 * Runner: plain TS, exits non-zero on assertion failure. From this package:
 *   npx tsx src/lib/parse-trajectory-proxy.smoke.ts
 */
import { type OtelSpan, type ToolCallEvent, parseOtelTrajectory } from './parse-trajectory';

let failures = 0;
function assert(cond: unknown, msg: string): void {
  if (cond) {
    console.log(`✓ ${msg}`);
  } else {
    failures += 1;
    console.error(`✗ ${msg}`);
  }
}

function span(name: string, operation: string, at: string, attributes: Record<string, unknown>, status?: string) {
  const attrs: Record<string, string> = { 'gen_ai.operation.name': operation };
  for (const [key, value] of Object.entries(attributes)) {
    attrs[key] = typeof value === 'string' ? value : JSON.stringify(value);
  }
  return {
    name,
    start_time: at,
    end_time: at,
    ...(status ? { status: { status_code: status } } : {}),
    attributes: attrs,
  };
}

const spans = [
  span('chat', 'chat', '2026-10-23T10:00:01+00:00', {
    'gen_ai.request.model': 'qwen',
    'gen_ai.completion': { content: [
      { type: 'text', text: 'Copying it.' },
      { type: 'tool_use', id: 'c1', name: 'bash', input: { command: 'cp a b' } },
      { type: 'tool_use', id: 'c2', name: 'items_add_item', input: { item: 'crate' } },
    ] },
  }, 'OK'),
  span('bash', 'execute_tool', '2026-10-23T10:00:02+00:00', {
    'gen_ai.prompt': { input: { command: 'cp a b' } },
    'gen_ai.completion': { output: 'cp: a: No such file or directory\n', exit_code: 1 },
  }, 'ERROR'),
  span('items_add_item', 'execute_tool', '2026-10-23T10:00:03+00:00', {
    'gen_ai.prompt': { input: { item: 'crate' } },
    'gen_ai.completion': { output: '{"count": 1}' },
  }),
  span('turn', 'chain', '2026-10-23T10:00:00+00:00', {
    'gen_ai.request.model': 'qwen',
    'gen_ai.prompt': { prompt: 'Copy a to b and add a crate.', system_prompt: null },
    'gen_ai.completion': { content: [{ type: 'text', text: 'Added a crate; a was missing.' }] },
  }, 'OK'),
];

const parsed = parseOtelTrajectory(spans as unknown as OtelSpan[]);
const calls = parsed.events.filter((e): e is ToolCallEvent => e.type === 'tool_call');

assert(parsed.model === 'qwen', 'the model comes from the turn');
assert(parsed.userPrompt === 'Copy a to b and add a crate.', 'the prompt comes from the turn');
assert(parsed.finalResponse === 'Added a crate; a was missing.', 'the reply comes from the turn');
assert(calls.length === 2 && parsed.toolCallCount === 2, 'each recorded call is a tool call');
const [bash, add] = calls;
assert(bash?.result?.isError === true, 'an action the proxy marked ERROR is an error');
assert(add?.result?.output.includes('"count": 1') && add.result.isError === false,
  'an action with no status renders');

if (failures > 0) {
  console.error(`\n${failures} assertion(s) failed`);
  process.exit(1);
}
console.log('\nall proxy trajectory assertions passed');
