'use client';

import type React from 'react';
import {
  RubricGradingResults,
  type RubricCriterion,
  type VerificationResults,
} from '../components/rubric-grading-results';
import {
  type JudgeOutputFormat,
  isJudgeOutputFormat,
  JudgeOutputFormat as JudgeOutputFormatEnum,
} from './judge-output-format';

export {
  classifyVerifier,
  JUDGE_OUTPUT_FORMAT_TO_KIND,
  type VerifierKind,
} from './verifier-classification';

export interface VerifierRenderContext {
  verifierId: string;
  verifier: Record<string, unknown>;
  rubricsCriteria?: Record<string, unknown>[];
  rubricsAggregator?: string;
}

type VerifierRenderer = (ctx: VerifierRenderContext) => React.ReactNode;

export const JUDGE_OUTPUT_FORMAT_RENDERERS: Partial<
  Record<JudgeOutputFormat, VerifierRenderer>
> = {
  [JudgeOutputFormatEnum.RUBRIC_BINARY]: ctx => {
    if (!ctx.rubricsCriteria?.length) return null;
    return (
      <RubricGradingResults
        verifierId={ctx.verifierId}
        verificationResults={
          { [ctx.verifierId]: ctx.verifier } as unknown as VerificationResults
        }
        rubrics={ctx.rubricsCriteria as unknown as RubricCriterion[]}
        aggregator={ctx.rubricsAggregator}
      />
    );
  },
};

/** Render a verifier via the format registry, or null if format is absent/unknown. */
export function renderJudgeOutputFormatVerifier(
  ctx: VerifierRenderContext,
): React.ReactNode | null {
  const format = ctx.verifier.format;
  if (!isJudgeOutputFormat(format)) return null;
  const render = JUDGE_OUTPUT_FORMAT_RENDERERS[format];
  return render?.(ctx) ?? null;
}
