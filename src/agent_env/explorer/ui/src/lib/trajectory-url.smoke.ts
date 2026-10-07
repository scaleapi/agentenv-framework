/**
 * Smoke test for reading a prompt response's trajectory URLs under either key.
 *
 * Runner: plain TS, exits non-zero on assertion failure. From this package:
 *   npx tsx src/lib/trajectory-url.smoke.ts
 */
import { materializeInstance } from './materialize';
import { perTurnTrajectoryUrls, trajectoryUrl } from './trajectory-url';

let failures = 0;
function assert(cond: unknown, msg: string): void {
  if (cond) {
    console.log(`✓ ${msg}`);
  } else {
    failures += 1;
    console.error(`✗ ${msg}`);
  }
}

const TURNS = ['obj://t/1.json', null, 'obj://t/2.json'];

function turnUrls(inst: Record<string, unknown>): unknown[] {
  const materialized = materializeInstance(inst, undefined);
  return (materialized.trajectory as Record<string, unknown>[])
    .filter(m => m.role === 'user')
    .map(m => m.trajectoryS3Uri);
}

function main(): void {
  {
    const legacy = {
      agent_trajectory_s3_uri: 'obj://t/2.json',
      target_agent_per_turn_trajectory_s3_uris: TURNS,
    };
    const neutral = {
      agent_trajectory_object_url: 'obj://t/2.json',
      target_agent_per_turn_trajectory_object_urls: TURNS,
    };
    for (const [name, pr] of [
      ['legacy', legacy],
      ['neutral', neutral],
    ] as const) {
      assert(
        trajectoryUrl(pr) === 'obj://t/2.json',
        `${name} keys: the trajectory URL`,
      );
      assert(
        JSON.stringify(perTurnTrajectoryUrls(pr)) === JSON.stringify(TURNS),
        `${name} keys: the per-turn URLs, a failed turn kept as null`,
      );
    }
  }

  // The S3-named key wins while both are written: only a raw-doc writer that knows nothing else makes them differ.
  {
    const pr = {
      agent_trajectory_s3_uri: 'obj://t/new.json',
      agent_trajectory_object_url: 'obj://t/stale.json',
      target_agent_per_turn_trajectory_s3_uris: [],
      target_agent_per_turn_trajectory_object_urls: ['obj://t/stale.json'],
    };
    assert(
      trajectoryUrl(pr) === 'obj://t/new.json',
      'the S3-named trajectory URL wins',
    );
    assert(
      perTurnTrajectoryUrls(pr)?.length === 0,
      'an empty S3-named per-turn list wins',
    );
  }

  {
    const pr = {
      agent_trajectory_s3_uri: null,
      agent_trajectory_object_url: 'obj://t/2.json',
    };
    assert(
      trajectoryUrl(pr) === 'obj://t/2.json',
      'a null S3-named key falls through',
    );
    assert(trajectoryUrl({}) === undefined, 'no key: no trajectory');
    assert(perTurnTrajectoryUrls({}) === undefined, 'no key: no per-turn list');
    assert(
      trajectoryUrl({ agent_trajectory_object_url: 42 }) === undefined,
      'a non-string URL is ignored',
    );
  }

  // materializeInstance reads them too: one user turn per per-turn URL, else one for the response's URL.
  {
    const response = { prompt_id: 'p1', response: 'done', prompt_text: 'go' };
    const multi = (keys: Record<string, unknown>) =>
      turnUrls({ context: { prompt_responses: [{ ...response, ...keys }] } });
    assert(
      JSON.stringify(
        multi({ target_agent_per_turn_trajectory_object_urls: TURNS }),
      ) === JSON.stringify(TURNS),
      'materialize: per-turn URLs under the object-store name',
    );
    assert(
      JSON.stringify(
        multi({ target_agent_per_turn_trajectory_s3_uris: TURNS }),
      ) === JSON.stringify(TURNS),
      'materialize: per-turn URLs under the S3-named key',
    );
    assert(
      JSON.stringify(
        multi({ agent_trajectory_object_url: 'obj://t/only.json' }),
      ) === JSON.stringify(['obj://t/only.json']),
      'materialize: a single-turn URL under the object-store name',
    );
  }

  if (failures > 0) {
    console.error(`\n${failures} assertion(s) failed`);
    process.exit(1);
  }
  console.log('\nall assertions passed');
}

main();
