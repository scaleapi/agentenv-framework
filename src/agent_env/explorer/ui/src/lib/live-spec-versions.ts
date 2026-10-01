/**
 * The package-version rows of a live spec's overview. The keys are the ones
 * `explorer/openapi_docs.py` writes into the metadata's `versions`, which keeps the
 * protocol under its import-package key, `agentenv-protocol`, whatever its
 * distribution is called.
 */
export function liveSpecVersionRows(
  versions: Record<string, string | null | undefined> | undefined,
): [string, string][] {
  return [
    ['agentenv-framework', versions?.['agentenv-framework'] ?? ''],
    ['agentenv-framework-protocol', versions?.['agentenv-protocol'] ?? ''],
  ];
}
