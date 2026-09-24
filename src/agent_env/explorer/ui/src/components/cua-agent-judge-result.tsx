'use client';

import { ChevronRight, CheckCircle2, XCircle, CircleDot } from 'lucide-react';
import { useMemo, useState } from 'react';

// `cua_evaluate` persists each agent_judge_* verifier as a flat result row whose
// `message` is a single rendered string, e.g.:
//   agent_judge_multi rubric_only score=0.60 (rubric_score=0.60): <narrative>
// Two-pass runs add a bracketed mode tag (which may itself contain a colon):
//   agent_judge_multi strictness=high score=0.34 (verdict_score=0.08, ...):
//     [must-pass] <title>: score=N weight=N contribution=N PASS
//     [must-pass] <title>: score=N weight=N FAIL (gates rubric to 0.00)
//     [regular]   <title>: score=N weight=N contribution=N (<reason>)
// must-pass gates the rubric to 0 on failure; "must-have" normalizes to "regular".
// When the message matches, render a structured layout; else fall back to raw <p>.

export interface CuaAgentJudgeCriterion {
  type: string; // "must-pass" | "regular" | etc. (legacy "must-have" is normalized to "regular" on parse)
  title: string;
  score: number;
  weight: number;
  contribution: number | null; // null when the row failed a must-pass gate
  passed: boolean;
  reason: string;
}

export interface CuaAgentJudgeParsed {
  func: string;
  variant: string; // e.g. "rubric_only"
  topScore: number;
  rubricScore: number | null;
  verdictScore: number | null;
  mode: string | null; // bracketed tag, e.g. "two-pass: golden-compare + golden-free rubric"
  narrative: string;
  criteria: CuaAgentJudgeCriterion[];
  footer: {
    gateState: string | null;
    gateCount: number | null;
    positive: number | null;
    totalWeight: number | null;
    penalty: number | null;
    rubricScore: number | null;
  };
}

// Groups: 1=func, 2=variant, 3=topScore, 4=score paren, 5=bracketed mode tag, 6=rest. The mode tag uses
// `[^\]]` (not lazy `.*?`) so an internal colon (e.g. "two-pass: …") doesn't trip the trailing ":".
const HEADER_RE =
  /^(\w+)\s+(\S+)\s+score=(-?\d+(?:\.\d+)?)\s*(?:\(([^)]*)\))?\s*(?:\[([^\]]*)\])?\s*:\s*([\s\S]*)$/;
const NAMED_SCORE_RE = (name: string) =>
  new RegExp(String.raw`\b${name}=(-?\d+(?:\.\d+)?)`);
// Per-criterion line — matches both "PASS contribution=N" and "FAIL (gates rubric to 0.00)". Strips the
// "(gates rubric to ...)" sentinel from the reason since the FAIL classification already carries it.
const CRITERION_RE = new RegExp(
  String.raw`\[([a-z][a-z-]*?)\]\s+` + // [must-pass] / [regular] (legacy [must-have] normalized to regular below)
    String.raw`(.+?):\s+` + // <title>:
    String.raw`score=(-?\d+(?:\.\d+)?)\s+` + // score=N.NN
    String.raw`weight=(-?\d+(?:\.\d+)?)` + // weight=N.N
    String.raw`(?:\s+contribution=(-?\d+(?:\.\d+)?))?` + // contribution=N.NN (only on PASS)
    String.raw`(\s+PASS)?` + // PASS marker (only on must-pass success)
    String.raw`(?:\s+FAIL\s*\([^)]*\))?` + // FAIL (gates rubric to 0.00) (only on must-pass failure)
    String.raw`\s+\((.*)\)`, // (<reason>) — the reason is the last thing on
  // Greedily capture to the final ")" on the line (no `s` flag, so it stays on the criterion line) to handle
  // reasons with nested parens.
  'g',
);
// Footer has two shapes: "must-pass gate PASSED (N gate(s)), positive=N/N, penalty=N, rubric_score=N" or
// "must-pass gate FAILED, rubric_score=0.00" (positive/penalty dropped since the gate forces 0). Match each piece independently.
const FOOTER_RE =
  /=>\s+must-pass\s+gate\s+(\w+)(?:\s*\((\d+)\s+gate\(s\)\))?(?:\s*,\s*positive=(-?\d+(?:\.\d+)?)\/(-?\d+(?:\.\d+)?))?(?:\s*,\s*penalty=(-?\d+(?:\.\d+)?))?(?:\s*,\s*rubric_score=(-?\d+(?:\.\d+)?))?/;

/** Parse the agent_judge_* rendered message; returns null on a shape mismatch so callers can fall back to plain text. */
export function parseCuaAgentJudgeMessage(
  message: string | null | undefined,
): CuaAgentJudgeParsed | null {
  if (typeof message !== 'string') return null;
  const trimmed = message.trim();
  if (!trimmed.toLowerCase().startsWith('agent_judge')) return null;

  const headerMatch = trimmed.match(HEADER_RE);
  if (!headerMatch) return null;
  // Capture groups type as `string | undefined`; pull each out with a fallback so the parser sees strings.
  const func = headerMatch[1] ?? 'agent_judge';
  const variant = headerMatch[2] ?? '';
  const topScoreStr = headerMatch[3] ?? '0';
  const scoreParen = headerMatch[4] ?? '';
  const rubricScoreStr = scoreParen.match(NAMED_SCORE_RE('rubric_score'))?.[1];
  const verdictScoreStr = scoreParen.match(
    NAMED_SCORE_RE('verdict_score'),
  )?.[1];
  const mode = headerMatch[5]?.trim() || null;
  const rest = headerMatch[6] ?? '';

  // Footer first so we can slice it off the body for the narrative.
  const footerMatch = trimmed.match(FOOTER_RE);
  let body = rest;
  let footer: CuaAgentJudgeParsed['footer'] = {
    gateState: null,
    gateCount: null,
    positive: null,
    totalWeight: null,
    penalty: null,
    rubricScore: null,
  };
  if (footerMatch) {
    body = body.slice(0, body.lastIndexOf(footerMatch[0]));
    footer = {
      gateState: footerMatch[1] ?? null,
      gateCount: footerMatch[2] ? Number(footerMatch[2]) : null,
      positive: footerMatch[3] !== undefined ? Number(footerMatch[3]) : null,
      totalWeight: footerMatch[4] !== undefined ? Number(footerMatch[4]) : null,
      penalty: footerMatch[5] !== undefined ? Number(footerMatch[5]) : null,
      rubricScore: footerMatch[6] !== undefined ? Number(footerMatch[6]) : null,
    };
  }

  // Narrative is everything up to "Rubric breakdown:" — the agent_judge_*
  // family always emits this marker even when the rubric is empty.
  const breakdownIdx = body.indexOf('Rubric breakdown:');
  const narrative = (breakdownIdx >= 0 ? body.slice(0, breakdownIdx) : body)
    .replace(/\s+/g, ' ')
    .trim();

  const criteria: CuaAgentJudgeCriterion[] = [];
  if (breakdownIdx >= 0) {
    const breakdown = body.slice(breakdownIdx + 'Rubric breakdown:'.length);
    let m: RegExpExecArray | null;
    CRITERION_RE.lastIndex = 0;
    while ((m = CRITERION_RE.exec(breakdown)) !== null) {
      const rawType = (m[1] ?? '').toLowerCase();
      // "must-have" is the legacy label for "regular" — normalize so old outputs render identically to new.
      const type = rawType === 'must-have' ? 'regular' : rawType;
      const title = (m[2] ?? '').trim();
      const score = Number(m[3] ?? '0');
      const weight = Number(m[4] ?? '0');
      const contribution = m[5] !== undefined ? Number(m[5]) : null;
      const passMarker = m[6];
      const reason = (m[7] ?? '').trim();
      const passed = passMarker !== undefined || score >= 1;
      criteria.push({
        type,
        title,
        score,
        weight,
        contribution,
        passed,
        reason,
      });
    }
  }

  return {
    func,
    variant,
    topScore: Number(topScoreStr),
    rubricScore: rubricScoreStr !== undefined ? Number(rubricScoreStr) : null,
    verdictScore:
      verdictScoreStr !== undefined ? Number(verdictScoreStr) : null,
    mode,
    narrative,
    criteria,
    footer,
  };
}

type FilterMode = 'failures' | 'all';
type SortMode = 'failures-first' | 'weight-desc' | 'contribution-desc';

function statusIcon(c: CuaAgentJudgeCriterion) {
  if (c.passed && c.score >= 1) {
    return (
      <CheckCircle2 size={14} className="text-emerald-500 flex-shrink-0" />
    );
  }
  if (!c.passed && c.score === 0) {
    return <XCircle size={14} className="text-red-500 flex-shrink-0" />;
  }
  return <CircleDot size={14} className="text-amber-500 flex-shrink-0" />;
}

function typeBadge(type: string) {
  const isMustPass = type === 'must-pass';
  return (
    <span
      className={`text-[10px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded ${
        isMustPass
          ? 'bg-red-500/10 text-red-600 dark:text-red-400'
          : 'bg-[var(--secondary)] text-[var(--muted-foreground)]'
      }`}
    >
      {type}
    </span>
  );
}

function CriterionRow({ c }: { c: CuaAgentJudgeCriterion }) {
  const [open, setOpen] = useState(false);
  // The reason field carries the LLM's per-criterion justification. It's
  // always a single line in practice; allow expanding if it wraps.
  return (
    <div className="border-t border-[var(--border)]">
      <div
        role="button"
        tabIndex={0}
        // A div (not <button>) so the title stays selectable; suppress the toggle when the click ends a text selection drag.
        onClick={() => {
          const selection = window.getSelection();
          if (selection && selection.toString().length > 0) return;
          setOpen(v => !v);
        }}
        onKeyDown={e => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            setOpen(v => !v);
          }
        }}
        className="w-full flex items-start gap-2 py-2 px-3 text-left cursor-pointer select-text hover:bg-[var(--secondary)]/40 transition-colors"
      >
        <ChevronRight
          size={14}
          className={`text-[var(--muted-foreground)] flex-shrink-0 mt-0.5 transition-transform ${
            open ? 'rotate-90' : ''
          }`}
        />
        <div className="flex-shrink-0 mt-0.5">{statusIcon(c)}</div>
        <div className="flex-shrink-0 mt-0.5">{typeBadge(c.type)}</div>
        <span className="text-xs text-[var(--muted-foreground)] flex-shrink-0 mt-0.5 font-mono w-16">
          w {c.weight}
        </span>
        <span className="text-xs text-[var(--muted-foreground)] flex-shrink-0 mt-0.5 font-mono w-20">
          {c.contribution !== null
            ? `+${c.contribution.toFixed(2)}`
            : c.passed
            ? '+0.00'
            : '—'}
        </span>
        {/* flex-1 + min-w-0 lets the title consume the remaining row width
            and wrap; without min-w-0 the flex item refuses to shrink below
            its content and the card's overflow-hidden clips long titles.
            break-words handles long unbroken filenames. */}
        <span className="text-sm leading-snug flex-1 min-w-0 break-words">
          {c.title}
        </span>
      </div>
      {open && (
        <div className="pb-3 px-3 pl-[5.5rem] text-xs text-[var(--muted-foreground)] leading-relaxed">
          {c.reason}
        </div>
      )}
    </div>
  );
}

export function CuaAgentJudgeResult({
  message,
  topLineScore,
}: {
  message: string;
  /** Score from the surrounding verifier card; when provided, skip the duplicate score chip in our header. */
  topLineScore?: number;
}) {
  const parsed = useMemo(() => parseCuaAgentJudgeMessage(message), [message]);
  const [filter, setFilter] = useState<FilterMode>('all');
  const [sort, setSort] = useState<SortMode>('failures-first');

  // Fall back to plain text when parsing fails — keeps the renderer drop-in safe for any non-agent_judge message.
  if (!parsed) {
    return (
      <p className="text-xs text-[var(--muted-foreground)] mb-3 whitespace-pre-wrap">
        {message}
      </p>
    );
  }

  const failingCount = parsed.criteria.filter(c => !c.passed).length;
  const passingCount = parsed.criteria.length - failingCount;
  const filtered = parsed.criteria.filter(c =>
    filter === 'all' ? true : !c.passed,
  );
  const sorted = [...filtered].sort((a, b) => {
    if (sort === 'weight-desc') return b.weight - a.weight;
    if (sort === 'contribution-desc') {
      const aC = a.contribution ?? -1;
      const bC = b.contribution ?? -1;
      return bC - aC;
    }
    // failures-first then by weight descending
    if (a.passed !== b.passed) return a.passed ? 1 : -1;
    return b.weight - a.weight;
  });

  const showScoreChip = topLineScore === undefined;
  // The FAILED-gate footer omits positive=N/N (the gate forces 0), so compute the weighted bar from the
  // criteria: sum of weights as denominator, sum of contributions (weight × score) as numerator, excluding the gated must-pass row.
  const derivedTotalWeight = parsed.criteria.reduce((s, c) => s + c.weight, 0);
  const derivedPositive = parsed.criteria.reduce(
    (s, c) => s + (c.contribution ?? c.score * c.weight),
    0,
  );
  const positive = parsed.footer.positive ?? derivedPositive;
  const totalWeight = parsed.footer.totalWeight ?? derivedTotalWeight;
  const hasWeighted = totalWeight > 0;
  const ratio = hasWeighted ? positive / Math.max(totalWeight, 1) : 0;
  const barColor =
    (parsed.footer.rubricScore ?? parsed.topScore) >= 1
      ? 'bg-emerald-500'
      : (parsed.footer.rubricScore ?? parsed.topScore) >= 0.5
      ? 'bg-amber-500'
      : 'bg-red-500';
  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--background)] overflow-hidden">
      {/* Header — `agent_judge_multi` / `rubric_only` are already shown by
          the surrounding verifier card (check header + Function detail
          row), so we don't repeat them here. The footer's
          positive/total + penalty (the "how was the percentage actually
          computed" data that used to live below) is now inline here
          along with the must-pass gate and the criterion pass count. */}
      <div className="px-3 py-2 border-b border-[var(--border)] bg-[var(--secondary)]/40 flex items-center gap-2 flex-wrap text-xs">
        {/* must-pass gate state is omitted here — the surrounding verifier
            card already shows the overall pass/fail icon next to the func
            name; repeating "MUST-PASS PASSED" inline made the row noisy
            without adding information. The gate count is still surfaced via
            tooltip on the criterion list if needed. */}
        {hasWeighted && (
          <>
            <span
              className="font-mono font-semibold text-[var(--foreground)]"
              title={
                parsed.footer.positive !== null
                  ? `${positive.toFixed(2)} of ${totalWeight} weighted points`
                  : `${positive.toFixed(
                      2,
                    )} of ${totalWeight} weighted points (derived from criteria; must-pass gate failure forced the published score to 0)`
              }
            >
              {positive.toFixed(2)} / {totalWeight}
            </span>
            <div
              className="relative h-2 w-40 rounded bg-[var(--secondary)] overflow-hidden border border-[var(--border)]"
              title={`${(ratio * 100).toFixed(1)}% weighted contribution`}
            >
              <div
                className={`absolute inset-y-0 left-0 ${barColor}`}
                style={{
                  width: `${Math.max(0, Math.min(100, ratio * 100))}%`,
                }}
              />
            </div>
            {parsed.footer.penalty !== null && parsed.footer.penalty !== 0 && (
              <span
                className="text-[var(--muted-foreground)] font-mono"
                title="Penalty subtracted from the positive contribution"
              >
                −{Math.abs(parsed.footer.penalty).toFixed(2)} penalty
              </span>
            )}
          </>
        )}
        <span className="text-[var(--muted-foreground)]">
          {passingCount}/{parsed.criteria.length} criteria pass
        </span>
        {parsed.mode && (
          <span
            className="px-1.5 py-0.5 rounded bg-[var(--secondary)] border border-[var(--border)] text-[var(--muted-foreground)]"
            title="Scoring mode reported by the verifier"
          >
            {parsed.mode}
          </span>
        )}
        {showScoreChip && (
          <span
            className={`ml-auto font-semibold px-1.5 py-0.5 rounded ${
              parsed.topScore >= 1
                ? 'bg-emerald-500/10 text-emerald-500'
                : parsed.topScore > 0
                ? 'bg-amber-500/10 text-amber-500'
                : 'bg-red-500/10 text-red-500'
            }`}
          >
            {(parsed.topScore * 100).toFixed(0)}%
          </span>
        )}
      </div>

      {/* Narrative */}
      {parsed.narrative && (
        <div className="px-3 py-2 text-xs text-[var(--muted-foreground)] leading-relaxed border-b border-[var(--border)]">
          {parsed.narrative}
        </div>
      )}

      {/* Filter / sort controls */}
      {parsed.criteria.length > 0 && (
        <>
          <div className="px-3 py-1.5 flex items-center gap-3 text-[11px] text-[var(--muted-foreground)] bg-[var(--secondary)]/20 border-b border-[var(--border)]">
            <span>
              {failingCount} fail · {passingCount} pass
            </span>
            <div className="flex items-center gap-1">
              <span>Show:</span>
              <button
                type="button"
                onClick={() => setFilter('failures')}
                className={`px-1.5 py-0.5 rounded ${
                  filter === 'failures'
                    ? 'bg-[var(--foreground)] text-[var(--background)]'
                    : 'hover:bg-[var(--secondary)]'
                }`}
              >
                failures
              </button>
              <button
                type="button"
                onClick={() => setFilter('all')}
                className={`px-1.5 py-0.5 rounded ${
                  filter === 'all'
                    ? 'bg-[var(--foreground)] text-[var(--background)]'
                    : 'hover:bg-[var(--secondary)]'
                }`}
              >
                all
              </button>
            </div>
            <div className="flex items-center gap-1">
              <span>Sort:</span>
              <select
                value={sort}
                onChange={e => setSort(e.target.value as SortMode)}
                className="bg-transparent border border-[var(--border)] rounded px-1 py-0.5 text-[11px]"
              >
                <option value="failures-first">failures first</option>
                <option value="weight-desc">weight ↓</option>
                <option value="contribution-desc">contribution ↓</option>
              </select>
            </div>
          </div>

          <div className="flex flex-col">
            {sorted.map((c, i) => (
              <CriterionRow key={`${c.title}-${i}`} c={c} />
            ))}
            {sorted.length === 0 && (
              <div className="px-3 py-3 text-xs text-[var(--muted-foreground)] italic">
                No criteria matched the current filter.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
