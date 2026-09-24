'use client';

import { useState, useMemo, useEffect } from 'react';
import { X, CheckCircle2, XCircle } from 'lucide-react';
import { cn } from '../lib/utils';
import {
  type RubricCriterion,
  type VerificationResult,
  type VerificationResults,
  type VerifierOutput,
} from './rubric-grading-results';
import { JudgeOutputFormat } from '../lib/judge-output-format';

/**
 * Resolve the user-facing label for a rubric criterion. Real task
 * configs frequently set only `criterion` (the full graded question),
 * not `title`. Fall back chain: title → criterion → id.
 */
function rubricLabel(rubric: RubricCriterion): string {
  if (typeof rubric.title === 'string' && rubric.title.trim()) {
    return rubric.title;
  }
  if (typeof rubric.criterion === 'string' && rubric.criterion.trim()) {
    return rubric.criterion;
  }
  return rubric.id;
}

/**
 * Pick the verifier output that actually grades the configured rubrics,
 * not just whichever happens to come first in dict iteration order.
 *
 * Tasks like `mm-avatar-full-urls` write multiple entries to
 * `metadata.verifications` — `verify-fs` (sandbox), `verify-response`
 * (rubric), `aggregate`. Prefer entries stamped with
 * `format: "rubric_binary"`; fall back to max rubric-id overlap
 * for legacy instances.
 */
function countRubricIdOverlap(
  output: VerifierOutput,
  rubricIds: Set<string>,
): number {
  if (!output?.results) return 0;
  let overlap = 0;
  for (const r of output.results) {
    if (r?.id && rubricIds.has(r.id)) overlap++;
  }
  return overlap;
}

function pickRubricVerifier(
  verificationResults: VerificationResults | undefined,
  rubrics: RubricCriterion[],
): VerifierOutput | null {
  if (!verificationResults) return null;
  const rubricIds = new Set(rubrics.map(r => r.id));

  const stamped = Object.values(verificationResults).filter(
    o =>
      o?.format === JudgeOutputFormat.RUBRIC_BINARY &&
      Array.isArray(o.results) &&
      o.results.length > 0,
  );
  if (stamped.length === 1) return stamped[0] ?? null;
  if (stamped.length > 1) {
    let bestStamped: { output: VerifierOutput; overlap: number } | null = null;
    for (const output of stamped) {
      const overlap = countRubricIdOverlap(output, rubricIds);
      if (overlap > 0 && (!bestStamped || overlap > bestStamped.overlap)) {
        bestStamped = { output, overlap };
      }
    }
    return bestStamped?.output ?? stamped[0] ?? null;
  }

  let best: { output: VerifierOutput; overlap: number } | null = null;
  for (const output of Object.values(verificationResults)) {
    const overlap = countRubricIdOverlap(output, rubricIds);
    if (overlap > 0 && (!best || overlap > best.overlap)) {
      best = { output, overlap };
    }
  }
  return best?.output ?? null;
}

export interface MatrixRun {
  id: string;
  name: string;
  status: string;
  verificationResults?: VerificationResults;
}

type DetailSelection =
  | { type: 'cell'; runId: string; criterionId: string }
  | { type: 'criterion'; criterionId: string }
  | null;

interface CriteriaColumn {
  id: string;
  rubric: RubricCriterion;
  passCount: number;
  totalCount: number;
  passRate: number;
}

interface MatrixRow {
  run: MatrixRun;
  cells: Map<string, VerificationResult>;
  passCount: number;
  totalCount: number;
  allPassed: boolean;
}

interface RubricMatrixProps {
  runs: MatrixRun[];
  rubrics: RubricCriterion[];
}

export function RubricMatrix({ runs, rubrics }: RubricMatrixProps) {
  const [selectedDetail, setSelectedDetail] = useState<DetailSelection>(null);

  // Escape key handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelectedDetail(null);
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const { columns, rows, passAt1 } = useMemo(() => {
    const completedRuns = runs.filter(
      r =>
        (r.status === 'completed' ||
          r.status === 'passed' ||
          r.status === 'failed') &&
        r.verificationResults,
    );

    const rubricMap = new Map(rubrics.map(r => [r.id, r]));

    // Build rows
    const matrixRows: MatrixRow[] = completedRuns.map(run => {
      const verifier = pickRubricVerifier(run.verificationResults, rubrics);
      const cells = new Map<string, VerificationResult>();
      let passCount = 0;
      let totalCount = 0;

      if (verifier?.results) {
        for (const result of verifier.results) {
          cells.set(result.id, result);
          totalCount++;
          if (result.result) passCount++;
        }
      }

      return {
        run,
        cells,
        passCount,
        totalCount,
        allPassed: totalCount > 0 && passCount === totalCount,
      };
    });

    // Build columns with pass rates
    const criteriaIds = rubrics.map(r => r.id);
    const cols: CriteriaColumn[] = criteriaIds.map(id => {
      let passCount = 0;
      let totalCount = 0;
      for (const row of matrixRows) {
        const cell = row.cells.get(id);
        if (cell) {
          totalCount++;
          if (cell.result) passCount++;
        }
      }
      return {
        id,
        rubric: rubricMap.get(id) ?? { id, title: id },
        passCount,
        totalCount,
        passRate: totalCount > 0 ? passCount / totalCount : 0,
      };
    });

    // Sort by pass rate ascending (hardest first), ties broken by ID
    cols.sort((a, b) => a.passRate - b.passRate || a.id.localeCompare(b.id));

    const totalCompleted = matrixRows.length;
    const totalPassed = matrixRows.filter(r => r.allPassed).length;
    const p1 = totalCompleted > 0 ? totalPassed / totalCompleted : null;

    return { columns: cols, rows: matrixRows, passAt1: p1 };
  }, [runs, rubrics]);

  if (rows.length === 0) return null;

  // Resolve detail data
  const detailData = (() => {
    if (!selectedDetail) return null;

    const col = columns.find(c => c.id === selectedDetail.criterionId);
    if (!col) return null;

    if (selectedDetail.type === 'criterion') {
      return { type: 'criterion' as const, column: col };
    }

    const row = rows.find(r => r.run.id === selectedDetail.runId);
    if (!row) return null;

    const cell = row.cells.get(selectedDetail.criterionId) ?? null;
    return { type: 'cell' as const, column: col, row, cell };
  })();

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--background)] w-fit max-w-full">
      {/* Stats bar */}
      <div className="flex items-center gap-3 px-4 pt-4 pb-2">
        <span className="inline-flex items-center text-xs font-medium px-2.5 py-1 rounded-full bg-[var(--secondary)] text-[var(--foreground)]">
          {rows.length} run{rows.length !== 1 ? 's' : ''}
        </span>
        <span className="inline-flex items-center text-xs font-medium px-2.5 py-1 rounded-full bg-[var(--secondary)] text-[var(--foreground)]">
          {columns.length} criteria
        </span>
        {passAt1 !== null && (
          <span
            className={cn(
              'inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-1 rounded-full',
              passAt1 === 1
                ? 'bg-green-50 text-green-600'
                : 'bg-red-50 text-red-500',
            )}
          >
            {passAt1 === 1 ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
            pass@1: {Math.round(passAt1 * 100)}%
          </span>
        )}
      </div>

      <div className="flex min-h-0 w-fit max-w-full">
        {/* Matrix table */}
        <div className="min-w-0 overflow-x-auto px-4 pb-4">
          <table className="border-separate" style={{ borderSpacing: '2px' }}>
            <thead>
              <tr>
                <th />
                {columns.map(col => (
                  <th
                    key={col.id}
                    className="p-0 cursor-pointer group"
                    onClick={() =>
                      setSelectedDetail({
                        type: 'criterion',
                        criterionId: col.id,
                      })
                    }
                  >
                    <div
                      className="text-[9px] font-normal text-[var(--muted-foreground)] group-hover:text-[var(--foreground)] px-0.5 py-1 max-h-[100px] overflow-hidden leading-tight"
                      style={{
                        writingMode: 'vertical-rl',
                        transform: 'rotate(180deg)',
                      }}
                      title={rubricLabel(col.rubric)}
                    >
                      {col.id.length > 12 ? `${col.id.slice(0, 10)}…` : col.id}
                    </div>
                  </th>
                ))}
                <th className="px-1.5 text-[9px] font-normal text-[var(--muted-foreground)]">
                  Score
                </th>
                <th className="px-1.5 text-[9px] font-normal text-[var(--muted-foreground)]">
                  Result
                </th>
              </tr>
            </thead>

            <tbody>
              {rows.map(row => (
                <tr key={row.run.id}>
                  <td className="pr-1.5 text-[9px] text-[var(--muted-foreground)] text-right whitespace-nowrap">
                    {row.run.name}
                  </td>
                  {columns.map(col => {
                    const cell = row.cells.get(col.id);
                    const isMissing = !cell;
                    const passed = cell?.result ?? false;

                    return (
                      <td key={col.id} className="p-0">
                        <button
                          className={cn(
                            'w-5 h-5 rounded-sm transition-shadow',
                            isMissing
                              ? 'bg-[var(--secondary)] cursor-default'
                              : passed
                              ? 'bg-green-600 cursor-pointer hover:ring-2 hover:ring-green-400'
                              : 'bg-red-500 cursor-pointer hover:ring-2 hover:ring-red-300',
                            selectedDetail?.type === 'cell' &&
                              selectedDetail.runId === row.run.id &&
                              selectedDetail.criterionId === col.id &&
                              'ring-2 ring-[var(--foreground)]',
                          )}
                          disabled={isMissing}
                          onClick={() =>
                            !isMissing &&
                            setSelectedDetail({
                              type: 'cell',
                              runId: row.run.id,
                              criterionId: col.id,
                            })
                          }
                          aria-label={
                            isMissing
                              ? `${row.run.name}, ${rubricLabel(
                                  col.rubric,
                                )}: No data`
                              : `${row.run.name}, ${rubricLabel(col.rubric)}: ${
                                  passed ? 'Pass' : 'Fail'
                                }`
                          }
                        />
                      </td>
                    );
                  })}
                  <td className="pl-1.5 text-[9px] text-[var(--muted-foreground)] whitespace-nowrap">
                    {row.passCount}/{row.totalCount}
                  </td>
                  <td className="pl-1.5">
                    <span
                      className={cn(
                        'text-[9px] font-semibold px-1.5 py-0.5 rounded',
                        row.allPassed
                          ? 'bg-green-100 text-green-700'
                          : 'bg-red-100 text-red-700',
                      )}
                    >
                      {row.allPassed ? 'PASS' : 'FAIL'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>

            <tfoot>
              <tr>
                <td className="pr-1.5 text-[9px] text-[var(--muted-foreground)] text-right">
                  Rate
                </td>
                {columns.map(col => (
                  <td
                    key={col.id}
                    className="text-[9px] text-center text-[var(--muted-foreground)] font-mono pt-1"
                  >
                    {col.totalCount > 0
                      ? `${Math.round(col.passRate * 100)}%`
                      : '–'}
                  </td>
                ))}
                <td />
                <td />
              </tr>
            </tfoot>
          </table>

          {/* Legend */}
          <div className="flex items-center gap-4 mt-3 text-[10px] text-[var(--muted-foreground)]">
            <span className="inline-flex items-center gap-1">
              <span className="w-3 h-3 rounded-sm bg-green-600" />
              Pass
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="w-3 h-3 rounded-sm bg-red-500" />
              Fail
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="w-3 h-3 rounded-sm bg-[var(--secondary)]" />
              N/A
            </span>
            <span className="text-[var(--muted-foreground)]">
              Click cell for detail · Click column header for criteria info
            </span>
          </div>
        </div>

        {/* Detail panel */}
        {detailData && (
          <div className="w-[360px] flex-shrink-0 border-l border-[var(--border)] overflow-y-auto p-4">
            <div className="flex items-start justify-between mb-3">
              <h3 className="text-sm font-semibold text-[var(--foreground)]">
                {detailData.type === 'cell' ? 'Run Detail' : 'Criteria Detail'}
              </h3>
              <button
                onClick={() => setSelectedDetail(null)}
                className="p-0.5 rounded hover:bg-[var(--secondary)] text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
              >
                <X size={14} />
              </button>
            </div>

            {/* Badges */}
            <div className="flex flex-wrap items-center gap-1.5 mb-3">
              {detailData.type === 'cell' && detailData.cell && (
                <span
                  className={cn(
                    'inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded',
                    detailData.cell.result
                      ? 'bg-green-100 text-green-700'
                      : 'bg-red-100 text-red-700',
                  )}
                >
                  {detailData.cell.result ? (
                    <CheckCircle2 size={10} />
                  ) : (
                    <XCircle size={10} />
                  )}
                  {detailData.cell.result ? 'PASS' : 'FAIL'}
                </span>
              )}
              <span
                className={cn(
                  'text-[10px] font-semibold px-2 py-0.5 rounded',
                  detailData.column.passRate >= 0.8
                    ? 'bg-green-50 text-green-600'
                    : detailData.column.passRate >= 0.5
                    ? 'bg-yellow-50 text-yellow-700'
                    : 'bg-red-50 text-red-500',
                )}
              >
                {Math.round(detailData.column.passRate * 100)}% pass rate
              </span>
              {detailData.column.rubric.annotations?.rubric_category && (
                <span className="text-[10px] px-2 py-0.5 rounded bg-[var(--secondary)] text-[var(--muted-foreground)]">
                  {detailData.column.rubric.annotations.rubric_category}
                </span>
              )}
            </div>

            {/* Criteria title — title field if set, else fall back to
                the full `criterion` text (real task configs typically
                ship `criterion` and leave `title` empty). */}
            <p className="text-xs font-medium text-[var(--foreground)] mb-3 leading-relaxed">
              {rubricLabel(detailData.column.rubric)}
            </p>

            {/* Fields */}
            <div className="flex flex-col gap-2.5 text-xs">
              {detailData.type === 'cell' && (
                <div>
                  <div className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wider mb-0.5">
                    Run
                  </div>
                  <div className="text-[var(--foreground)]">
                    {detailData.row.run.name}
                  </div>
                </div>
              )}

              <div>
                <div className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wider mb-0.5">
                  Criteria ID
                </div>
                <div className="text-[var(--foreground)] font-mono text-[11px]">
                  {detailData.column.id}
                </div>
              </div>

              {detailData.type === 'cell' &&
                (detailData.cell?.justification ||
                  detailData.cell?.message) && (
                  <div>
                    <div className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wider mb-0.5">
                      Judge Justification
                    </div>
                    <div className="text-[var(--foreground)] leading-relaxed whitespace-pre-wrap">
                      {detailData.cell.justification || detailData.cell.message}
                    </div>
                  </div>
                )}

              {detailData.column.rubric.annotations?.justification != null && (
                <div className="pt-1 border-t border-[var(--border)]">
                  <div className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wider mb-0.5">
                    Rubric Rationale
                  </div>
                  <div className="text-[var(--foreground)] leading-relaxed">
                    {String(detailData.column.rubric.annotations.justification)}
                  </div>
                </div>
              )}

              {detailData.column.rubric.annotations?.evidence != null && (
                <div>
                  <div className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wider mb-0.5">
                    Expected Evidence
                  </div>
                  <div className="text-[var(--foreground)] leading-relaxed">
                    {String(detailData.column.rubric.annotations.evidence)}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
