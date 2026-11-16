/**
 * Smoke test for the collected-artifacts "Download All" file list.
 *
 * Runner: plain TS, exits non-zero on assertion failure. From this package:
 *   npx tsx src/lib/collected-zip-files.smoke.ts
 */
import { collectedZipFiles } from './collected-zip-files';

let failures = 0;
function assert(cond: unknown, msg: string): void {
  if (cond) {
    console.log(`✓ ${msg}`);
  } else {
    failures += 1;
    console.error(`✗ ${msg}`);
  }
}

function main(): void {
  // Every backend's object URL is included; only an artifact without one is left out.
  {
    const files = collectedZipFiles(
      {
        'report.md': 's3://bucket/run/report.md',
        'out/data.csv': 'file:///state/objects/run/out/data.csv',
        'shot.png': 'gs://bucket/run/shot.png',
        'missing.txt': '',
        'broken.txt': null as unknown as string,
      },
      '/app',
    );
    const uris = files.map(f => f.objectUrl);
    assert(uris.length === 3, 'three artifacts have a url');
    assert(uris.includes('file:///state/objects/run/out/data.csv'), 'a local file:// url is included');
    assert(uris.includes('gs://bucket/run/shot.png'), 'another backend\'s url is included');
    assert(!files.some(f => f.path === 'missing.txt' || f.path === 'broken.txt'), 'empty and non-string urls are dropped');
  }

  // Relative keys stay as they are; absolute keys lose the base path, or their leading slash.
  {
    const files = collectedZipFiles(
      {
        'rel/a.txt': 'fake://home/a',
        '/app/abs/b.txt': 'fake://home/b',
        '/elsewhere/c.txt': 'fake://home/c',
      },
      '/app/',
    );
    const byUri = Object.fromEntries(files.map(f => [f.objectUrl, f.path]));
    assert(byUri['fake://home/a'] === 'rel/a.txt', 'a relative key is kept');
    assert(byUri['fake://home/b'] === 'abs/b.txt', 'an absolute key under base_path is made relative');
    assert(byUri['fake://home/c'] === 'elsewhere/c.txt', 'an absolute key elsewhere loses its leading slash');
  }

  // No member name climbs out of the archive or names a root or a drive, whatever the key holds.
  {
    const files = collectedZipFiles(
      {
        '../../x': 'fake://home/x',
        '/abs/../../y': 'fake://home/y',
        'a/../../../z': 'fake://home/z',
        '..\\..\\w.txt': 'fake://home/w',
        'C:\\Users\\op\\.ssh\\v': 'fake://home/v',
        'C:evil': 'fake://home/e',
        './out/./u.txt': 'fake://home/u',
        '..': 'fake://home/dots',
      },
      '/app',
    );
    const byUri = Object.fromEntries(files.map(f => [f.objectUrl, f.path]));
    assert(byUri['fake://home/x'] === 'x', 'leading .. segments are dropped');
    assert(byUri['fake://home/y'] === 'y', 'an absolute key outside base_path climbs no higher than the root');
    assert(byUri['fake://home/z'] === 'z', 'a .. past the start is dropped');
    assert(byUri['fake://home/w'] === 'w.txt', 'backslashes separate like slashes');
    assert(byUri['fake://home/v'] === 'Users/op/.ssh/v', 'a drive prefix is dropped');
    assert(byUri['fake://home/e'] === 'evil', 'a drive-relative prefix is dropped');
    assert(byUri['fake://home/u'] === 'out/u.txt', '. segments are dropped');
    assert(byUri['fake://home/dots'] === 'artifact', 'a key with nothing left is named artifact');
    assert(
      files.every(f => !f.path.split('/').some(part => part === '..' || part === '') && !/^[A-Za-z]:/.test(f.path)),
      'every member name is a clean relative path',
    );
  }

  if (failures > 0) {
    console.error(`\n${failures} assertion(s) failed`);
    process.exit(1);
  }
  console.log('\nall assertions passed');
}

main();
