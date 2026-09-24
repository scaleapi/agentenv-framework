import {
  type JudgeOutputFormat,
  isJudgeOutputFormat,
  JudgeOutputFormat as JudgeOutputFormatEnum,
} from './judge-output-format';

// Verifier-output shapes persisted to `context.metadata.verifications[<verifier_id>]`:
//   `aggregate_verifiers` → { score, source_verifier_ids } (no results)
//   `verify_sandbox`      → { score, results: [{ type: probe_*|bash_cmd_succeeds }] }
//   `rubrics_verifier`    → { format: "rubric_binary", score, results }
// Prefer the stamped `format`; fall back to shape sniffing when it's absent.
export type VerifierKind = 'aggregate' | 'sandbox' | 'rubric' | 'other';

const SANDBOX_CRITERION_TYPES = new Set([
  'probe_file_exists',
  'probe_dir_exists',
  'probe_file_contains',
  'bash_cmd_succeeds',
]);

/** Maps stamped judge output formats to the viewer bucket used for layout. */
export const JUDGE_OUTPUT_FORMAT_TO_KIND: Record<
  JudgeOutputFormat,
  VerifierKind
> = {
  [JudgeOutputFormatEnum.RUBRIC_BINARY]: 'rubric',
};

export function classifyVerifier(v: unknown): VerifierKind {
  if (!v || typeof v !== 'object') return 'other';
  const obj = v as Record<string, unknown>;

  if (isJudgeOutputFormat(obj.format)) {
    return JUDGE_OUTPUT_FORMAT_TO_KIND[obj.format];
  }

  if (Array.isArray(obj.source_verifier_ids)) return 'aggregate';
  if (!Array.isArray(obj.results) || obj.results.length === 0) return 'other';
  const firstNonSkipped =
    (obj.results as Record<string, unknown>[]).find(r => !r?.skipped) ??
    (obj.results as Record<string, unknown>[])[0];
  const t = (firstNonSkipped as { type?: unknown } | undefined)?.type;
  if (typeof t === 'string' && SANDBOX_CRITERION_TYPES.has(t)) {
    return 'sandbox';
  }
  // 'passed' at top level = CUA/MCP validation entries (<MCPEnvValidationEntry>).
  if ('passed' in obj) return 'other';
  return 'rubric';
}

/**
 * Final rollout score for one instance's `metadata.verifications` map. Keys are in
 * completion order, not "aggregate last", so the aggregate is found structurally:
 * the entry `aggregate_verifiers` stamps with `source_verifier_ids`. One aggregate
 * → its score; else a lone verifier → its score; otherwise null (renders `--`).
 */
export function selectFinalScore(
  verifications: Record<string, unknown> | undefined | null,
): number | null {
  if (!verifications) return null;
  const scoreOf = (v: unknown): number | null => {
    const s = (v as { score?: unknown } | null | undefined)?.score;
    return typeof s === 'number' ? s : null;
  };
  const entries = Object.entries(verifications);
  const aggregates = entries.filter(
    ([, v]) => classifyVerifier(v) === 'aggregate',
  );
  const soleAggregate = aggregates.length === 1 ? aggregates[0] : undefined;
  if (soleAggregate) return scoreOf(soleAggregate[1]);
  const soleVerifier =
    aggregates.length === 0 && entries.length === 1 ? entries[0] : undefined;
  if (soleVerifier) return scoreOf(soleVerifier[1]);
  return null;
}
