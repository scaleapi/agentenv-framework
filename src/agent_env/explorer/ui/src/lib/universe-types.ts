/**
 * Tolerant reader for the environment-universe artifact `type` discriminator.
 *
 * agent-env renamed `ServiceUniverseArtifact` to `EnvironmentUniverseArtifact`, so
 * `artifact.type` may be either `service_universe` (everything written to date) or
 * `environment_universe` (post-cutover writes). Reads accept both; writers only
 * ever emit `CANONICAL_ENVIRONMENT_UNIVERSE_TYPE`.
 *
 * No imports, so it stays portable across packages.
 */

/** Every wire spelling of the environment-universe artifact type. */
export const ENVIRONMENT_UNIVERSE_TYPES = [
  'service_universe',
  'environment_universe',
] as const;

/** The only spelling written today. Never flip this to the new name. */
export const CANONICAL_ENVIRONMENT_UNIVERSE_TYPE = 'service_universe';

/** Canonical union used for UI dispatch — one member per universe family. */
export type UniverseType =
  | 'service_universe'
  | 'file_artifact_universe'
  | 'coding_task_harbor';

export function isEnvironmentUniverseType(v: unknown): boolean {
  return (
    typeof v === 'string' &&
    (ENVIRONMENT_UNIVERSE_TYPES as readonly string[]).includes(v)
  );
}

/**
 * Collapse a raw `artifact.type` onto the canonical union. Unknown values fall
 * through to the environment universe, preserving the default branch of the
 * ternary this replaces.
 */
export function normalizeUniverseType(raw: unknown): UniverseType {
  const t = String(raw ?? '');
  if (t === 'file_artifact_universe') return 'file_artifact_universe';
  if (t === 'coding_task_harbor') return 'coding_task_harbor';
  return CANONICAL_ENVIRONMENT_UNIVERSE_TYPE;
}

/**
 * Wire values to send for one canonical type on the repeated `?type=` filter
 * of `GET /api/v1/artifacts` (the backend unions them with `$in`).
 */
export function universeTypeQueryValues(t: UniverseType): string[] {
  return t === CANONICAL_ENVIRONMENT_UNIVERSE_TYPE
    ? [...ENVIRONMENT_UNIVERSE_TYPES]
    : [t];
}

/** Wire values for the "All" tab. */
export const ALL_UNIVERSE_QUERY_TYPES: string[] = [
  ...ENVIRONMENT_UNIVERSE_TYPES,
  'file_artifact_universe',
  'coding_task_harbor',
];
