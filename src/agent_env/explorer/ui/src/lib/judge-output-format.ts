/** Mirror of agent-env's `JudgeOutputFormat` enum. String values must stay in sync — renderers dispatch on the stamped `format` in context.metadata.verifications[<id>]. */
export const JudgeOutputFormat = {
  RUBRIC_BINARY: 'rubric_binary',
} as const;

export type JudgeOutputFormat =
  (typeof JudgeOutputFormat)[keyof typeof JudgeOutputFormat];

/** All registered judge output formats (extend when agent-env adds members). */
export const JUDGE_OUTPUT_FORMATS: readonly JudgeOutputFormat[] = [
  JudgeOutputFormat.RUBRIC_BINARY,
];

export function isJudgeOutputFormat(v: unknown): v is JudgeOutputFormat {
  return (
    typeof v === 'string' &&
    (JUDGE_OUTPUT_FORMATS as readonly string[]).includes(v)
  );
}
