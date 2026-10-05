'use client';

import { CheckCircle2, XCircle } from 'lucide-react';
import type { JudgeOutputFormat } from '../lib/judge-output-format';

export interface RubricCriterion {
  id: string;
  /** Short human-readable rubric label. Configs sometimes set only `criterion`; falls back to criterion then id. */
  title?: string;
  /** Full criterion text (the question being graded) — the field configs use in practice; the primary
   *  rubric-cell text when `title` is absent. */
  criterion?: string;
  /** Author-declared weight / score contribution. Load-bearing for `weighted_average`; informational for
   *  all_pass/any_pass. Negative weights are legitimate (penalty rubrics) and flagged distinctly. */
  weight?: number;
  annotations?: {
    rubric_category?: string;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface VerificationResult {
  id: string;
  score: number;
  result: boolean;
  justification?: string;
  message?: string;
  // AgentPromptResponseVerifier rows carry the original criterion fields; surface them so the row is
    // self-describing (pass-case messages omit the check values).
  type?: string;
  needles?: string[];
  pattern?: string;
  flags?: string[];
}

export interface VerifierOutput {
  /** Stamped by agent-env rubrics_verifier (judge output format registry). */
  format?: JudgeOutputFormat;
  results: VerificationResult[];
  score: number;
}

export type VerificationResults = Record<string, VerifierOutput>;

/** Extract the first verifier output from the verification results map. */
export function getFirstVerifierOutput(
  vr?: VerificationResults,
): VerifierOutput | null {
  if (!vr) return null;
  const firstKey = Object.keys(vr)[0];
  if (!firstKey) return null;
  return vr[firstKey] as VerifierOutput;
}

interface CategoryGroup {
  category: string;
  results: VerificationResult[];
}

function groupByCategory(
  results: VerificationResult[],
  rubrics: RubricCriterion[],
): CategoryGroup[] {
  const rubricMap = new Map(rubrics.map(r => [r.id, r]));
  const groupMap = new Map<string, VerificationResult[]>();

  for (const result of results) {
    const rubric = rubricMap.get(result.id);
    // Coerce so category holds even when rubric_category came through as a number/boolean; `?? 'Other'` only catches null/undefined.
    const rawCategory = rubric?.annotations?.rubric_category;
    const category =
      typeof rawCategory === 'string'
        ? rawCategory
        : rawCategory === null || rawCategory === undefined
        ? 'Other'
        : String(rawCategory);
    const group = groupMap.get(category);
    if (group) {
      group.push(result);
    } else {
      groupMap.set(category, [result]);
    }
  }

  // Preserve insertion order, but push "Other" to the end
  const groups: CategoryGroup[] = [];
  let otherGroup: CategoryGroup | null = null;
  for (const [category, results] of groupMap) {
    if (category === 'Other') {
      otherGroup = { category, results };
    } else {
      groups.push({ category, results });
    }
  }
  if (otherGroup) groups.push(otherGroup);

  return groups;
}

function capitalize(s: unknown): string {
  // annotations is typed `unknown`, so rubric_category may be a number/boolean/null from hand-edited or older
  // rubric JSON. Coerce to string defensively so a stray non-string doesn't break the panel.
  if (s === null || s === undefined) return '';
  const str = typeof s === 'string' ? s : String(s);
  return str
    .split(' ')
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/** Author-declared weight as a plain number. The tooltip carries the semantic context (default/penalty/zero). */
function WeightBadge({
  weight,
  isDefault = false,
}: {
  weight: number;
  isDefault?: boolean;
}) {
  if (isDefault) {
    return (
      <span
        className="font-mono text-sm italic text-[var(--muted-foreground)]"
        title="No weight configured — default 1.0"
      >
        1.0
      </span>
    );
  }
  // Display with a typographic minus for negative weights (rendered
  // text is `−5`, not `-5`).
  const displayed = weight < 0 ? `−${Math.abs(weight)}` : String(weight);
  return (
    <span
      className="font-mono text-sm text-[var(--foreground)]"
      title={
        weight < 0
          ? 'Penalty rubric: passing this reduces the score'
          : weight === 0
          ? "Zero-weight: doesn't contribute to score"
          : `Weight ${weight}`
      }
    >
      {displayed}
    </span>
  );
}

/** Inline display of an AgentPromptResponseVerifier check value: response_contains → needles,
 *  response_regex_present → pattern. Null for rubric verifiers with no such fields. */
function ResponseCheckSummary({ result }: { result: VerificationResult }) {
  if (result.type === 'response_contains' && result.needles?.length) {
    return (
      <div className="flex flex-wrap items-baseline gap-1">
        <span className="text-[10px] text-[var(--muted-foreground)]">
          needles:
        </span>
        {result.needles.map((n, i) => (
          <code
            key={i}
            className="text-[10px] font-mono px-1 py-0.5 rounded bg-[var(--secondary)] text-[var(--foreground)] max-w-[260px] truncate"
            title={n}
          >
            {n}
          </code>
        ))}
      </div>
    );
  }
  if (result.type === 'response_regex_present' && result.pattern) {
    const flagsLabel = result.flags?.length
      ? ` (${result.flags.join(', ')})`
      : '';
    return (
      <div className="flex flex-wrap items-baseline gap-1">
        <span className="text-[10px] text-[var(--muted-foreground)]">
          regex:
        </span>
        <code
          className="text-[10px] font-mono px-1 py-0.5 rounded bg-[var(--secondary)] text-[var(--foreground)] max-w-[420px] truncate"
          title={result.pattern + flagsLabel}
        >
          /{result.pattern}/{flagsLabel}
        </code>
      </div>
    );
  }
  return null;
}

export function RubricGradingResults({
  verificationResults,
  rubrics,
  aggregator,
  verifierId,
}: {
  verificationResults?: VerificationResults;
  rubrics: RubricCriterion[];
  /** `score_aggregator` from the rubrics_verifier step. Suppresses the "Score: X%" badge for all_pass (and
   *  the default), where the score is implied by the N/M passed count; kept for partial-credit aggregators. */
  aggregator?: string;
  /** Verifier id badge next to the header. Only useful when a task emits multiple rubric verifiers; single-verifier callers omit it. */
  verifierId?: string;
}) {
  const verifier = getFirstVerifierOutput(verificationResults);
  // Treat undefined / 'all_pass' as the binary case where score is
  // redundant with the pass count.
  const isBinaryAggregator =
    aggregator === undefined || aggregator === 'all_pass';

  if (!verifier?.results?.length) {
    return (
      <div className="flex items-center justify-center h-40">
        <span className="text-sm text-[var(--muted-foreground)]">
          No rubric grading results available
        </span>
      </div>
    );
  }

  const { results } = verifier;
  // Prefer `title`, fall back to `criterion` (full question) — configs frequently set `criterion` only.
  const rubricMap = new Map(
    rubrics.map(r => {
      const label =
        (typeof r.title === 'string' && r.title.trim() ? r.title : null) ??
        (typeof r.criterion === 'string' && r.criterion.trim()
          ? r.criterion
          : null);
      return [r.id, label] as const;
    }),
  );
  // Weight-chip map, built only from rubrics with a numeric weight, so the chip auto-hides when none are configured.
  const weightById = new Map<string, number>(
    rubrics
      .filter(
        (r): r is RubricCriterion & { weight: number } =>
          typeof r.weight === 'number',
      )
      .map(r => [r.id, r.weight]),
  );
  const anyWeights = weightById.size > 0;
  const passedCount = results.filter(r => r.result).length;
  const groups = groupByCategory(results, rubrics);

  return (
    <div className="p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-baseline gap-2">
          <h3 className="text-base font-semibold">Rubrics grading results</h3>
          {verifierId && (
            <code className="text-[10px] font-mono text-[var(--muted-foreground)]">
              {verifierId}
            </code>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span
            className={`inline-flex items-center gap-1 text-sm font-medium ${
              passedCount === results.length ? 'text-green-600' : 'text-red-500'
            }`}
          >
            {passedCount === results.length ? (
              <CheckCircle2 size={14} />
            ) : (
              <XCircle size={14} />
            )}
            {passedCount}/{results.length} passed
          </span>
          {!isBinaryAggregator && (
            <span
              className={`text-xs font-semibold px-1.5 py-0.5 rounded ${
                verifier.score >= 1
                  ? 'bg-green-500/10 text-green-500'
                  : 'bg-red-500/10 text-red-500'
              }`}
            >
              Score: {(verifier.score * 100).toFixed(0)}%
            </span>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-4">
        {groups.map(group => {
          const groupPassed = group.results.filter(r => r.result).length;
          // Suppress the per-group header when there's only one group (it'd duplicate the section header); keep it for multiple categories.
          const showGroupHeader = groups.length > 1;
          return (
            <div key={group.category}>
              {showGroupHeader && (
                <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-[var(--border)]">
                  <span className="text-sm font-semibold text-[var(--foreground)]">
                    {capitalize(group.category)}
                  </span>
                  <span
                    className={`text-xs font-medium ${
                      groupPassed === group.results.length
                        ? 'text-green-600'
                        : 'text-[var(--muted-foreground)]'
                    }`}
                  >
                    {groupPassed}/{group.results.length} passed
                  </span>
                </div>
              )}

              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-[var(--muted-foreground)]">
                    <th className="py-1.5 pr-4 font-medium">Rubric</th>
                    {anyWeights && (
                      <th className="py-1.5 pr-4 font-medium w-20">Weight</th>
                    )}
                    <th className="py-1.5 pr-4 font-medium w-24">Decision</th>
                    <th className="py-1.5 font-medium">Justification</th>
                  </tr>
                </thead>
                <tbody>
                  {group.results.map(result => (
                    <tr
                      key={result.id}
                      className="border-t border-[var(--border)]"
                    >
                      <td className="py-2.5 pr-4 font-medium align-top">
                        {(() => {
                          const label = rubricMap.get(result.id);
                          return (
                            <div className="flex flex-col gap-0.5">
                              {label && (
                                <span className="text-sm font-medium leading-snug">
                                  {label}
                                </span>
                              )}
                              <span className="text-[10px] font-mono text-[var(--muted-foreground)]">
                                {result.id}
                              </span>
                              <ResponseCheckSummary result={result} />
                            </div>
                          );
                        })()}
                      </td>
                      {anyWeights && (
                        <td className="py-2.5 pr-4 align-middle">
                          {(() => {
                            const w = weightById.get(result.id);
                            // 1.0 is the implicit default for a rubric with no weight (matches the weighted_average
                            // aggregator). Render it italic/muted so author-weighted rows are distinguishable.
                            const isExplicit = typeof w === 'number';
                            return (
                              <WeightBadge
                                weight={isExplicit ? (w as number) : 1}
                                isDefault={!isExplicit}
                              />
                            );
                          })()}
                        </td>
                      )}
                      <td className="py-2.5 pr-4">
                        {result.result ? (
                          <span className="inline-flex items-center gap-1 text-green-600">
                            <CheckCircle2 size={14} /> Pass
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-red-500">
                            <XCircle size={14} /> Fail
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 text-[var(--muted-foreground)]">
                        {result.justification || result.message}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>
    </div>
  );
}
