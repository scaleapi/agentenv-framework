/**
 * Guard the containment on the artifact previews.
 *
 * The docx/xlsx/pptx previews render agent-produced bytes inside an unauthenticated
 * control-plane origin. docx-preview assigns a raw `w:sym w:char` into `innerHTML`,
 * which needs no click; measured against a crafted file, the payload fires in a plain
 * container and does not fire inside the sandboxed frame. The `<img>` is still created
 * either way — what the sandbox removes is execution, not the markup.
 *
 * This file guards ONE failure mode: someone adding `allow-scripts`, which together
 * with `allow-same-origin` lets framed content remove its own sandbox.
 *
 * It does NOT guard the containment as a whole, and should not be read as doing so.
 * docx-preview and pptx-preview both parse attacker-controlled markup in the host
 * document and only stay safe because they append into the frame synchronously — see
 * the `prepareSandboxDoc` docstring. A dependency bump can re-open the hole with this
 * file still green, which is why those two versions are pinned exactly in
 * package.json. Catching that needs a real render of a crafted file, not a grep.
 *
 * Runner: yarn test:smoke
 */
import * as fs from 'fs';
import * as path from 'path';

const VIEWER = path.join(
  __dirname,
  '..',
  'components',
  'task-instance-viewer.tsx',
);

let failures = 0;
function assert(cond: unknown, msg: string): void {
  if (cond) {
    console.log(`✓ ${msg}`);
  } else {
    failures += 1;
    console.error(`✗ ${msg}`);
  }
}

const src = fs.readFileSync(VIEWER, 'utf-8');

// Scan the <iframe ...> tags themselves rather than grepping for `sandbox=` anywhere
// in the file: the docstrings discuss sandbox values in prose, and matching those
// would both inflate the count and let a real unsandboxed frame hide behind them.
const frames = [...src.matchAll(/<iframe\b([\s\S]*?)\/>/g)].map(m => m[1] ?? '');
const sandboxOf = (tag: string): string | null => {
  const m = tag.match(/sandbox="([^"]*)"/);
  return m ? (m[1] ?? '') : null;
};

assert(frames.length >= 4, `found ${frames.length} preview frames (>= 4)`);

for (const tag of frames) {
  const value = sandboxOf(tag);
  const label = (tag.match(/title="([^"]*)"/)?.[1] ?? 'untitled').trim();
  assert(value !== null, `<iframe "${label}"> carries a sandbox attribute`);
  assert(
    value === null || !value.includes('allow-scripts'),
    `<iframe "${label}"> sandbox="${value}" does not grant allow-scripts`,
  );
  assert(
    value === null || ['', 'allow-same-origin', 'allow-popups'].includes(value),
    `<iframe "${label}"> sandbox="${value}" is an expected value`,
  );
}

// A preview must never be written straight into the host document again.
assert(
  !/dangerouslySetInnerHTML[\s\S]{0,400}sheets\[active\]/.test(src),
  'the spreadsheet preview does not use dangerouslySetInnerHTML',
);

if (failures > 0) {
  console.error(`\n${failures} preview-sandbox assertion(s) failed`);
  process.exit(1);
}
console.log('\nAll preview-sandbox smoke tests passed.');
