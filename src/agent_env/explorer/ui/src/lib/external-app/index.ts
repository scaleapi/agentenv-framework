/**
 * No-op stand-in for a host `use-external-app` hook. The standalone app is the
 * top-level page (never embedded in a host frame), so this keeps the hook's shape
 * and always reports "not embedded"; every embedded-only branch is guarded on
 * `isReady` / `receivedInputs` and simply never runs.
 */

export interface DataUrlItemContent {
  id: string;
  type: 'dataUrl';
  data: string;
}

export interface JsonItemContent {
  id: string;
  type: 'json';
  data: Record<string, unknown>;
}

export type SubmissionItemContent = DataUrlItemContent | JsonItemContent;

export type SubmissionItem = {
  content: SubmissionItemContent;
  metadata?: Record<string, unknown>;
};

export interface SubmissionPayload {
  items: SubmissionItem[];
}

export type BeforeNextHandler = () => unknown;

export interface ExternalApp {
  /** True only when running embedded in a host frame; always false standalone. */
  isReady: boolean;
  /** Inputs posted in by the host frame. */
  receivedInputs: Record<string, unknown> | null;
  receivedOutput: SubmissionPayload | null;
  skipRegeneration: boolean;
  /** Wraps the items in a SubmissionPayload. */
  sendSubmission: (items: SubmissionItem[]) => void;
  registerBeforeNextHandler: (handler: BeforeNextHandler | null) => void;
  assignmentId: string | null;
  taskId: string | null;
}

/** No-op external-app client: the standalone hub is never embedded. */
export function useExternalApp(_options?: {
  onBeforeNext?: BeforeNextHandler;
}): ExternalApp {
  return {
    isReady: false,
    receivedInputs: null,
    receivedOutput: null,
    skipRegeneration: false,
    sendSubmission: () => {},
    registerBeforeNextHandler: () => {},
    assignmentId: null,
    taskId: null,
  };
}
