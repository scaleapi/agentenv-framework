/**
 * Scheme allowlist for any href built from data an agent or a task author controls.
 *
 * React escapes text but does not strip `javascript:` from an href, and the app CSP
 * keeps `script-src 'unsafe-inline'` for the static export's bootstrap script, so it is
 * not a backstop either. A task whose `repo_url` or agent card `documentationUrl` is
 * `javascript:...` would otherwise run script in an unauthenticated same-origin plane
 * the moment a reviewer clicks it.
 *
 * Returns undefined for anything not http(s)/mailto, so the caller renders plain text
 * rather than a live link.
 */
const ALLOWED_PROTOCOLS = new Set(['http:', 'https:', 'mailto:']);

export function safeHref(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  if (!trimmed) return undefined;

  // Relative and root-relative urls never carry a scheme, so they cannot be
  // `javascript:`; `//host` is protocol-relative and inherits ours.
  if (trimmed.startsWith('/') || trimmed.startsWith('#') || trimmed.startsWith('?')) {
    return trimmed;
  }

  try {
    // A base makes a scheme-less value parse as relative rather than throwing, so the
    // check below sees the resolved protocol either way.
    const url = new URL(trimmed, 'https://invalid.local/');
    return ALLOWED_PROTOCOLS.has(url.protocol) ? trimmed : undefined;
  } catch {
    return undefined;
  }
}
