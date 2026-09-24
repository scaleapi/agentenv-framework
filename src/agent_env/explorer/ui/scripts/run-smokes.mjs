/**
 * Run every `src/lib/*.smoke.ts` and fail on the first one that does.
 *
 * Node rather than a shell one-liner so `npm run test:smoke` behaves the same on
 * Windows — `build:static`'s `cp` is already POSIX-only and that is a wart, not a
 * precedent to follow.
 *
 * A smoke may skip only by printing `agentenv-capability-missing: <name>` for a
 * capability in ALLOWED_CAPABILITIES — the same reason format the Python tiers use
 * (tst/util/capabilities.py) and CI enforces (.github/scripts/check_skip_policy.py).
 * Any other skip fails the run: a skip that merely prints a count is
 * indistinguishable from a pass.
 */
import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
// tsx's CLI run by this same node, so no shell and no PATH lookup: `npx`/`.cmd`
// shims are what forced `shell: true` on Windows.
const tsxCli = createRequire(import.meta.url).resolve('tsx/cli');
const smokes = readdirSync(path.join(root, 'src', 'lib'))
  .filter(f => f.endsWith('.smoke.ts'))
  .sort();

const CAPABILITY_PREFIX = 'agentenv-capability-missing: ';
// Capabilities a smoke may be missing. Both are a captured trajectory too large to
// commit, supplied by env var when someone wants that parser covered.
const ALLOWED_CAPABILITIES = new Set([
  'claude_cli_trajectory_fixture',
  'openinference_trajectory_fixture',
]);

let failed = 0;
let skipped = 0;
const badSkips = [];
for (const file of smokes) {
  const rel = path.join('src', 'lib', file);
  process.stdout.write(`\n— ${rel}\n`);
  const run = spawnSync(process.execPath, [tsxCli, rel], {
    cwd: root,
    stdio: 'pipe',
    encoding: 'utf8',
  });
  const out = (run.stdout || '') + (run.stderr || '');
  process.stdout.write(out);
  if (run.status !== 0) {
    failed += 1;
    continue;
  }
  // Only a line that STARTS with "- skipped:" is a skip. Assertion labels routinely
  // contain the word ("junk events are skipped"), and matching those would fail the
  // run on a passing test.
  const skipLine = out
    .split('\n')
    .find(line => line.trimStart().startsWith('- skipped:'));
  if (skipLine) {
    skipped += 1;
    const capability = skipLine.match(
      new RegExp(`${CAPABILITY_PREFIX}([\\w.-]+)`),
    )?.[1];
    if (!capability) {
      badSkips.push(`${file}: skipped without an '${CAPABILITY_PREFIX}' reason`);
    } else if (!ALLOWED_CAPABILITIES.has(capability)) {
      badSkips.push(`${file}: capability '${capability}' is not allowed here`);
    }
  }
}

const ran = smokes.length - skipped - failed;
console.log(
  `\n${ran} ran, ${skipped} skipped, ${failed} failed (of ${smokes.length})`,
);
for (const problem of badSkips) console.error(`disallowed skip — ${problem}`);
process.exit(failed === 0 && badSkips.length === 0 ? 0 : 1);
