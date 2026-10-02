/**
 * Smoke test for liveSpecVersionRows — a live spec's overview shows both versions.
 *
 * Runner: plain TS, throws on assertion failure. From this package:
 *   npx tsx src/lib/live-spec-versions.smoke.ts
 */
import { liveSpecVersionRows } from './live-spec-versions';

function assert(cond: unknown, msg: string): void {
  if (!cond) {
    console.error(`✗ ${msg}`);
    throw new Error(msg);
  }
  console.log(`✓ ${msg}`);
}

function main(): void {
  // The shape openapi_docs.py writes for a live spec.
  const rows = liveSpecVersionRows({
    'agentenv-framework': '0.9.1260',
    'agentenv-protocol': '0.1.275',
  });
  assert(
    JSON.stringify(rows) ===
      JSON.stringify([
        ['agentenv-framework', '0.9.1260'],
        ['agentenv-framework-protocol', '0.1.275'],
      ]),
    'both versions read from the keys the backend writes',
  );

  // The key the overview read before the rename no longer fills a row.
  const legacy = liveSpecVersionRows({ 'agent-env': '0.9.1251' });
  assert(
    legacy.every(([, value]) => value === ''),
    'the retired agent-env key fills no row',
  );

  assert(
    liveSpecVersionRows(undefined).every(([, value]) => value === ''),
    'missing versions render blank, not undefined',
  );
}

main();
