/**
 * Smoke test for appendSeeds — paste/upload accumulate onto the grid.
 *
 * Runner: plain TS, throws on assertion failure. From this package:
 *   npx tsx src/lib/csv-seeds.smoke.ts
 */
import {
  MAX_PARALLEL_RUNS,
  appendSeeds,
  countRealSeeds,
  parseSeedsText,
  validateSeeds,
  type ParsedSeeds,
} from './csv-seeds';

function assert(cond: unknown, msg: string): void {
  if (!cond) {
    console.error(`✗ ${msg}`);
    throw new Error(msg);
  }
  console.log(`✓ ${msg}`);
}

const blank: ParsedSeeds = {
  columns: ['id', 'repo_link'],
  seeds: [{ id: 'row-1', repo_link: '' }],
  sourceLines: [-1],
};

function main(): void {
  // Appending onto a pristine blank grid drops the empty starter row.
  const first = appendSeeds(blank, parseSeedsText('repo_link\na\nb'));
  assert(first.seeds.length === 2, 'blank + 2 pasted → 2 rows (starter dropped)');
  assert(
    first.seeds.map(r => r.repo_link).join(',') === 'a,b',
    'pasted values preserved',
  );

  // A second paste APPENDS rather than replacing.
  const second = appendSeeds(first, parseSeedsText('repo_link\nc'));
  assert(
    second.seeds.map(r => r.repo_link).join(',') === 'a,b,c',
    'second paste appends (does not replace)',
  );

  // Ids stay unique across appends, and keys (sourceLines) stay unique.
  assert(new Set(second.seeds.map(r => r.id)).size === 3, 'ids unique after appends');
  assert(new Set(second.sourceLines).size === 3, 'row keys unique after appends');
  validateSeeds(second); // would throw on dup id / empty id

  // Typed values on existing rows survive an append.
  const typed = appendSeeds(blank, parseSeedsText('repo_link\nkept'));
  const afterAppend = appendSeeds(typed, parseSeedsText('repo_link\nnew'));
  assert(
    afterAppend.seeds[0]!.repo_link === 'kept',
    'existing typed row preserved on append',
  );

  // Cap: cumulative appends never exceed MAX_PARALLEL_RUNS (a single parse is
  // already capped, so this only bites across appends).
  const batch = (n: number) =>
    parseSeedsText('repo_link\n' + Array.from({ length: n }, (_, i) => `r${i}`).join('\n'));
  const capped = appendSeeds(appendSeeds(blank, batch(40)), batch(40));
  assert(capped.seeds.length === MAX_PARALLEL_RUNS, 'cumulative append caps at MAX_PARALLEL_RUNS');

  // countRealSeeds ignores blank rows.
  assert(countRealSeeds(blank) === 0, 'blank starter row is not a real seed');
  assert(countRealSeeds(first) === 2, 'countRealSeeds counts non-empty rows');

  console.log('\nAll appendSeeds smoke assertions passed.');
}

main();
