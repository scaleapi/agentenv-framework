/**
 * Smoke test for verifier classification (format-first dispatch + legacy fallback).
 *
 * Runner: plain TS, throws on assertion failure. From this package:
 *   npx tsx src/lib/verifier-visualizers.smoke.ts
 */
import { isJudgeOutputFormat } from './judge-output-format';
import { classifyVerifier } from './verifier-classification';

function assert(cond: unknown, msg: string): void {
  if (!cond) {
    console.error(`✗ ${msg}`);
    throw new Error(msg);
  }
  console.log(`✓ ${msg}`);
}

function assertKind(v: unknown, expected: string, label: string): void {
  assert(classifyVerifier(v) === expected, `${label} → ${expected}`);
}

function main(): void {
  assert(
    isJudgeOutputFormat('rubric_binary'),
    'isJudgeOutputFormat accepts rubric_binary',
  );
  assert(
    !isJudgeOutputFormat('not_a_format'),
    'isJudgeOutputFormat rejects unknown',
  );

  assertKind(
    {
      format: 'rubric_binary',
      score: 1,
      results: [{ id: 'c1', score: 1, result: true, justification: 'ok' }],
    },
    'rubric',
    'stamped rubric_binary',
  );

  assertKind(
    { score: 0.5, source_verifier_ids: ['a', 'b'] },
    'aggregate',
    'legacy aggregate (source_verifier_ids)',
  );

  assertKind(
    {
      score: 1,
      results: [{ type: 'probe_file_exists', score: 1, result: true }],
    },
    'sandbox',
    'legacy sandbox (probe_file_exists)',
  );

  assertKind(
    {
      score: 0,
      results: [{ id: 'c1', score: 0, result: false }],
    },
    'rubric',
    'legacy rubric-shaped (id/score/result rows)',
  );

  assertKind(
    { passed: true, score: 1, results: [] },
    'other',
    'MCP validation (passed)',
  );

  assertKind(
    { format: 'unknown_future_format', results: [] },
    'other',
    'unknown format + empty results',
  );

  assertKind(
    {
      format: 'unknown_future_format',
      results: [{ id: 'c1', score: 1, result: true }],
    },
    'rubric',
    'unknown format falls through to legacy rubric sniffing',
  );

  assertKind(null, 'other', 'null input');
  assertKind(undefined, 'other', 'undefined input');

  console.log('\nAll verifier-visualizers smoke tests passed.');
}

main();
