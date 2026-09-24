/**
 * Smoke test for the task-instance bar download list.
 *
 * Runner: plain TS, exits non-zero on assertion failure. From this package:
 *   npx tsx src/lib/bar-downloads.smoke.ts
 *
 * `.smoke.ts`, not `.test.ts`, matching the sibling smoke files in this
 * directory. The package has no Jest config and `test` is a no-op, so there is
 * no runner a `.test.ts` file would be picked up by.
 */
import {
  MAX_INLINE_BAR_DOWNLOADS,
  buildBarDownloads,
  shouldCollapseBarDownloads,
} from './bar-downloads';

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
  // A single-step run keeps the bare "Trajectory" label it had before
  // multi-trajectory support.
  {
    const downloads = buildBarDownloads(
      [{ agent_trajectory_s3_uri: 'obj://t/one', step_id: 'run-solver' }],
      undefined,
    );
    assert(downloads.length === 1, 'single trajectory yields one download');
    assert(
      downloads[0]?.label === 'Trajectory',
      'single trajectory drops the step-name suffix',
    );
    assert(downloads[0]?.kind === 'trajectory', 'tagged as a trajectory');
    assert(downloads[0]?.s3Uri === 'obj://t/one', 'carries the trajectory URI');
    assert(
      !shouldCollapseBarDownloads(downloads.length),
      'one download stays inline',
    );
  }

  // Multi-step: label by step_id, fall back to prompt_id, then position.
  {
    const downloads = buildBarDownloads(
      [
        { agent_trajectory_s3_uri: 'obj://t/1', step_id: 'prepare-repo' },
        { agent_trajectory_s3_uri: 'obj://t/2', prompt_id: 'legacy-prompt' },
        { agent_trajectory_s3_uri: 'obj://t/3' },
      ],
      undefined,
    );
    assert(
      downloads.map(d => d.label).join(',') ===
        'prepare-repo,legacy-prompt,Trajectory 3',
      'labels prefer step_id, then prompt_id, then position',
    );
    assert(
      downloads.map(d => d.key).join(',') === 'obj://t/1,obj://t/2,obj://t/3',
      'keys are the trajectory URIs',
    );
  }

  // A null prompt_id (the shape TaskStepRef allows) must not become the label.
  {
    const [only] = buildBarDownloads(
      [
        { agent_trajectory_s3_uri: 'obj://t/a', prompt_id: null },
        { agent_trajectory_s3_uri: 'obj://t/b', prompt_id: null },
      ],
      undefined,
    );
    assert(only?.label === 'Trajectory 1', 'null prompt_id falls back to position');
  }

  // Prompt responses with no trajectory URI have nothing to download.
  {
    const downloads = buildBarDownloads(
      [
        { step_id: 'no-trajectory' },
        { agent_trajectory_s3_uri: 'obj://t/only', step_id: 'has-one' },
      ],
      undefined,
    );
    assert(downloads.length === 1, 'drops responses without a trajectory URI');
    assert(
      downloads[0]?.label === 'Trajectory',
      'the remaining single trajectory is treated as single',
    );
  }

  // The model lands in the tooltip, not the label — the bar is narrow.
  {
    const [entry] = buildBarDownloads(
      [
        { agent_trajectory_s3_uri: 'obj://t/1', step_id: 'a', model: 'opus-4.8' },
        { agent_trajectory_s3_uri: 'obj://t/2', step_id: 'b' },
      ],
      undefined,
    );
    assert(entry?.label === 'a', 'model is kept out of the label');
    assert(
      entry?.title === 'Download a (opus-4.8)',
      'model is shown in the tooltip',
    );
  }

  // Results are appended last, and count toward the collapse threshold.
  {
    const downloads = buildBarDownloads(
      [
        { agent_trajectory_s3_uri: 'obj://t/1', step_id: 'a' },
        { agent_trajectory_s3_uri: 'obj://t/2', step_id: 'b' },
      ],
      'obj://results/run.json',
    );
    assert(downloads.length === 3, 'results appended to the trajectories');
    const last = downloads[downloads.length - 1];
    assert(last?.kind === 'results', 'results entry comes last');
    assert(last?.label === 'Results', 'results entry is labelled "Results"');
    assert(
      last?.key === 'results:obj://results/run.json',
      'results key is namespaced so it cannot collide with a trajectory URI',
    );
    assert(
      shouldCollapseBarDownloads(downloads.length),
      'two trajectories plus results collapses',
    );
  }

  // No results URI → no results entry.
  {
    const downloads = buildBarDownloads(
      [{ agent_trajectory_s3_uri: 'obj://t/1', step_id: 'a' }],
      undefined,
    );
    assert(
      downloads.every(d => d.kind === 'trajectory'),
      'absent results URI adds no entry',
    );
  }

  // A run with no downloads at all renders an empty bar, not a dropdown.
  {
    const downloads = buildBarDownloads([], undefined);
    assert(downloads.length === 0, 'no responses yields no downloads');
    assert(!shouldCollapseBarDownloads(0), 'empty set does not collapse');
  }

  // The threshold itself: collapse strictly above MAX_INLINE_BAR_DOWNLOADS.
  {
    assert(
      !shouldCollapseBarDownloads(MAX_INLINE_BAR_DOWNLOADS),
      'exactly the inline limit stays inline',
    );
    assert(
      shouldCollapseBarDownloads(MAX_INLINE_BAR_DOWNLOADS + 1),
      'one over the inline limit collapses',
    );
  }

  // The reported case: demo-bench-proof-c2, many steps + results.
  {
    const many = Array.from({ length: 12 }, (_, i) => ({
      agent_trajectory_s3_uri: `obj://halt/${i}`,
      step_id: `step-${i}`,
    }));
    const downloads = buildBarDownloads(many, 'obj://halt/results.json');
    assert(downloads.length === 13, 'all 12 trajectories plus results kept');
    assert(
      shouldCollapseBarDownloads(downloads.length),
      'the reported task collapses into a dropdown',
    );
    assert(
      new Set(downloads.map(d => d.key)).size === downloads.length,
      'keys are unique across the collapsed list',
    );
  }

  if (failures > 0) {
    console.error(`\n${failures} assertion(s) failed`);
    process.exit(1);
  }
  console.log('\nall assertions passed');
}

main();
