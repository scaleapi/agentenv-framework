/** Download entries for the task-instance tab bar. Every prompt_agent step with `trajectory_output_prefix`
 *  uploads its own trajectory; the bar exposes one per trajectory (the last/solver step is usually most
 *  interesting). Past MAX_INLINE_BAR_DOWNLOADS the caller collapses them into a dropdown. Built here (not inline) so the row and dropdown share entries/order and the labels are testable. */

/** How many downloads may sit inline in the bar before they collapse. */
export const MAX_INLINE_BAR_DOWNLOADS = 2;

import { type TrajectoryUrlKeys, trajectoryUrl } from './trajectory-url';

/** The subset of a prompt-response the bar needs to label a trajectory. */
export interface BarDownloadTrajectory extends TrajectoryUrlKeys {
  step_id?: string;
  prompt_id?: string | null;
  model?: string;
}

export interface BarDownload {
  /** Stable React key. */
  key: string;
  label: string;
  /** Tooltip — spells out the action, and the model for trajectories. */
  title: string;
  objectUrl: string;
}

/** Build the bar's download list from a run's prompt responses.
 *  Trajectories are labelled by step_id, else prompt_id, else "Trajectory N"; a single trajectory gets the bare
 *  "Trajectory". Entries with no URI are dropped. */
export function buildBarDownloads(
  promptResponses: BarDownloadTrajectory[],
): BarDownload[] {
  const withTrajectory = promptResponses.filter(pr => trajectoryUrl(pr));
  const single = withTrajectory.length === 1;

  return withTrajectory.map((pr, i) => {
    const label = single
      ? 'Trajectory'
      : pr.step_id || pr.prompt_id || `Trajectory ${i + 1}`;
    const objectUrl = trajectoryUrl(pr) as string;
    return {
      key: objectUrl,
      label,
      title: pr.model ? `Download ${label} (${pr.model})` : `Download ${label}`,
      objectUrl,
    };
  });
}

/** Whether the bar should collapse its downloads into a dropdown. */
export function shouldCollapseBarDownloads(count: number): boolean {
  return count > MAX_INLINE_BAR_DOWNLOADS;
}
