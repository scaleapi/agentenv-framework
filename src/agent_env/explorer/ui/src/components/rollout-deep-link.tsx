import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Check, Copy } from 'lucide-react';

/**
 * Deep-linking for one rollout on the task detail page: URL contract,
 * parse/serialize, apply-on-load, and the copy-link button. Hash format
 * `#<instance_id>__<task_version>` (the version selects the right filter before
 * the instance is fetched); split on the LAST `__` so an id containing `__` still
 * round-trips. Reads/writes the hash only.
 */

const SEPARATOR = '__';

export interface RolloutDeepLink {
  instanceId: string;
  /** null when the link omits a version (the version filter is left as-is). */
  version: number | null;
}

/** The hash fragment, including the leading '#', for a rollout. */
export function rolloutHash({ instanceId, version }: RolloutDeepLink): string {
  const suffix = version == null ? '' : String(version);
  return `#${instanceId}${SEPARATOR}${suffix}`;
}

// Full shareable URL: current origin/path/query + rollout hash. SSR-safe.
export function rolloutDeepLinkUrl(link: RolloutDeepLink): string {
  if (typeof window === 'undefined') return rolloutHash(link);
  const { origin, pathname, search } = window.location;
  return `${origin}${pathname}${search}${rolloutHash(link)}`;
}

// Parse a hash (with or without leading '#') into a link, or null. Splits on
// the LAST separator so ids containing `__` survive the round trip.
export function parseRolloutHash(hash: string): RolloutDeepLink | null {
  const raw = hash.startsWith('#') ? hash.slice(1) : hash;
  if (!raw) return null;
  let decoded = raw;
  try {
    decoded = decodeURIComponent(raw);
  } catch {
    // Malformed percent-escape — fall back to the raw value.
  }
  const idx = decoded.lastIndexOf(SEPARATOR);
  if (idx <= 0) return null; // no separator, or empty instance id
  const instanceId = decoded.slice(0, idx);
  if (!instanceId) return null;
  const versionStr = decoded.slice(idx + SEPARATOR.length);
  const versionNum = Number(versionStr);
  const version =
    versionStr !== '' && Number.isFinite(versionNum) ? versionNum : null;
  return { instanceId, version };
}

// Update the hash via replaceState — no history entry, no scroll, no Next.js
// route change. Pure client-side bookmark of the focused rollout.
export function writeRolloutHash(link: RolloutDeepLink): void {
  if (typeof window === 'undefined') return;
  const { pathname, search } = window.location;
  window.history.replaceState(
    window.history.state,
    '',
    `${pathname}${search}${rolloutHash(link)}`,
  );
}

// Drop the hash (e.g. when the focused instance is collapsed).
export function clearRolloutHash(): void {
  if (typeof window === 'undefined') return;
  const { pathname, search } = window.location;
  window.history.replaceState(window.history.state, '', `${pathname}${search}`);
}

// The link to apply for the given task, re-read on taskId change. TaskDetailPage
// stays mounted across tasks, so re-reading (not capturing once) lets a switched-to
// task see the current hash (null, since navigation drops the stale one).
export function useRolloutDeepLinkForTask(
  taskId: string,
): RolloutDeepLink | null {
  const [link, setLink] = useState<RolloutDeepLink | null>(() =>
    typeof window === 'undefined'
      ? null
      : parseRolloutHash(window.location.hash),
  );
  const lastTaskId = useRef(taskId);
  useEffect(() => {
    if (lastTaskId.current === taskId) return;
    lastTaskId.current = taskId;
    setLink(
      typeof window === 'undefined'
        ? null
        : parseRolloutHash(window.location.hash),
    );
  }, [taskId]);
  return link;
}

// Copy-icon button: copies the rollout's full deep-link URL. Stops propagation
// so it doesn't also toggle/deep-link its row.
export function CopyRolloutLinkButton({
  link,
  className,
}: {
  link: RolloutDeepLink;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );

  const handleCopy = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      const writePromise = navigator.clipboard?.writeText(
        rolloutDeepLinkUrl(link),
      );
      if (!writePromise) return;
      void writePromise
        .then(() => {
          setCopied(true);
          if (timer.current) clearTimeout(timer.current);
          timer.current = setTimeout(() => setCopied(false), 1500);
        })
        .catch(() => {
          // Clipboard blocked (permissions / insecure context) — no-op.
        });
    },
    [link],
  );

  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label={`Copy deep link to rollout ${link.instanceId}`}
      title={copied ? 'Copied' : 'Copy deep link'}
      className={`inline-flex shrink-0 items-center justify-center rounded p-0.5 text-[var(--muted-foreground)] opacity-0 transition-opacity hover:bg-[var(--accent)] hover:text-[var(--foreground)] focus:opacity-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)] group-hover:opacity-100 ${
        copied ? 'opacity-100' : ''
      } ${className ?? ''}`}
    >
      {copied ? (
        <Check size={12} aria-hidden className="text-green-500" />
      ) : (
        <Copy size={12} aria-hidden />
      )}
    </button>
  );
}
