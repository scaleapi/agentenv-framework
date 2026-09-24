/** Extract <key> placeholder tokens from a task's prompts. Mirrors the `prompt_agent` substitution
 *  prompt.replace(f"<{key}>", str(value)), so `<name>` expects a seed key `name`. Paired XML-style markup (e.g.
 *  <IMPORTANT>…</IMPORTANT>) has only its delimiters stripped so tag names don't masquerade as placeholders. */

const PLACEHOLDER_RE = /<([A-Za-z0-9_]+)>/g;
const TAG_TOKEN_RE = /<\/?([A-Za-z0-9_]+)>/g;

function stripPairedTagDelimiters(text: string): string {
  const opened = new Set<string>();
  const closed = new Set<string>();
  for (const match of text.matchAll(TAG_TOKEN_RE)) {
    const token = match[0];
    const tagName = match[1];
    if (!tagName) continue;
    if (token.startsWith('</')) {
      closed.add(tagName);
    } else {
      opened.add(tagName);
    }
  }
  const pairedTagNames = new Set<string>();
  for (const tagName of opened) {
    if (closed.has(tagName)) pairedTagNames.add(tagName);
  }
  if (pairedTagNames.size === 0) return text;
  return text.replace(TAG_TOKEN_RE, (_token, tagName: string) =>
    pairedTagNames.has(tagName) ? '' : _token,
  );
}

export function extractPlaceholders(
  steps: ReadonlyArray<Record<string, unknown>>,
): string[] {
  const seen = new Set<string>();
  const ordered: string[] = [];
  for (const step of steps) {
    if (step.type !== 'prompt_agent') continue;
    for (const field of ['prompt', 'system_prompt']) {
      const text = step[field];
      if (typeof text !== 'string' || !text) continue;
      const stripped = stripPairedTagDelimiters(text);
      for (const match of stripped.matchAll(PLACEHOLDER_RE)) {
        const key = match[1];
        if (key && !seen.has(key)) {
          seen.add(key);
          ordered.push(key);
        }
      }
    }
  }
  return ordered;
}
