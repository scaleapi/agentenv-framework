/** Download entries for the task-instance tab bar. Every prompt_agent step with `trajectory_output_prefix`
 *  uploads its own trajectory; the bar exposes one per trajectory (the last/solver step is usually most
 *  interesting), plus the verifier results file. Past MAX_INLINE_BAR_DOWNLOADS the caller collapses them into a
 *  dropdown. Built here (not inline) so the row and dropdown share entries/order and the labels are testable. */

/** How many downloads may sit inline in the bar before they collapse. */
export const MAX_INLINE_BAR_DOWNLOADS = 2;

/** The subset of a prompt-response the bar needs to label a trajectory. */
export interface BarDownloadTrajectory {
  agent_trajectory_s3_uri?: string;
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
  kind: 'trajectory' | 'results';
  s3Uri: string;
}

/** Build the bar's download list from a run's prompt responses and its verifier results URI, if any.
 *  Trajectories are labelled by step_id, else prompt_id, else "Trajectory N"; a single trajectory gets the bare
 *  "Trajectory". Entries with no URI are dropped. */
export function buildBarDownloads(
  promptResponses: BarDownloadTrajectory[],
  resultS3Uri?: string,
): BarDownload[] {
  const withTrajectory = promptResponses.filter(
    pr => pr.agent_trajectory_s3_uri,
  );
  const single = withTrajectory.length === 1;

  const downloads: BarDownload[] = withTrajectory.map((pr, i) => {
    const label = single
      ? 'Trajectory'
      : pr.step_id || pr.prompt_id || `Trajectory ${i + 1}`;
    const s3Uri = pr.agent_trajectory_s3_uri as string;
    return {
      key: s3Uri,
      label,
      title: pr.model ? `Download ${label} (${pr.model})` : `Download ${label}`,
      kind: 'trajectory',
      s3Uri,
    };
  });

  if (resultS3Uri) {
    downloads.push({
      key: `results:${resultS3Uri}`,
      label: 'Results',
      title: 'Download Results',
      kind: 'results',
      s3Uri: resultS3Uri,
    });
  }

  return downloads;
}

/** Whether the bar should collapse its downloads into a dropdown. */
export function shouldCollapseBarDownloads(count: number): boolean {
  return count > MAX_INLINE_BAR_DOWNLOADS;
}
