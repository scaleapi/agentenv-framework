/** A prompt response's trajectory URLs. Core writes each under an object-store name beside its S3-named key, and
 *  instances recorded before that carry only the S3-named one. Read the S3-named key first while both are written
 *  (a raw-doc writer that knows only it leaves the other stale), falling through to the other when it is null or
 *  missing — the order core's `read_dual_keyed` reads them in. */
export interface TrajectoryUrlKeys {
  agent_trajectory_s3_uri?: unknown;
  agent_trajectory_object_url?: unknown;
  target_agent_per_turn_trajectory_s3_uris?: unknown;
  target_agent_per_turn_trajectory_object_urls?: unknown;
}

/** The trajectory of the response's last turn, if it has one. */
export function trajectoryUrl(
  pr: TrajectoryUrlKeys | null | undefined,
): string | undefined {
  const url = pr?.agent_trajectory_s3_uri ?? pr?.agent_trajectory_object_url;
  return typeof url === 'string' && url ? url : undefined;
}

/** One trajectory per turn of a multi-turn response; null for a turn whose upload failed. */
export function perTurnTrajectoryUrls(
  pr: TrajectoryUrlKeys | null | undefined,
): (string | null)[] | undefined {
  const urls =
    pr?.target_agent_per_turn_trajectory_s3_uris ??
    pr?.target_agent_per_turn_trajectory_object_urls;
  return Array.isArray(urls)
    ? urls.map(url => (typeof url === 'string' && url ? url : null))
    : undefined;
}
