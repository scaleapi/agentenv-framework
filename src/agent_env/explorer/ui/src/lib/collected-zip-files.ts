/** Files for a collect step's "Download All" zip, keyed relative to the step's base_path so the archive
 *  tree stays clean. Every artifact with a URL is included: the explorer decides what it can serve, and a
 *  refused one is listed in the zip's _failed.txt. */

export interface CollectedZipFile {
  path: string;
  objectUrl: string;
}

/** Artifact keys are absolute source paths or relative ones (joined under base_path). */
export const isAbsoluteArtifactPath = (p: string): boolean => /^(\/|[A-Za-z]:[\\/])/.test(p);

export function collectedZipFiles(
  artifacts: Record<string, string>,
  basePath: string,
): CollectedZipFile[] {
  const base = basePath.replace(/\/+$/, '');
  return Object.entries(artifacts)
    .filter(([, uri]) => typeof uri === 'string' && uri !== '')
    .map(([key, uri]) => ({
      path: archiveName(
        !isAbsoluteArtifactPath(key) ? key : key.startsWith(`${base}/`) ? key.slice(base.length + 1) : key,
      ),
      objectUrl: uri,
    }));
}

/** A key as a member name that stays inside the archive when extracted: an agent names its manifest's entries. */
export function archiveName(key: string): string {
  const parts: string[] = [];
  for (const part of key.replace(/^[A-Za-z]:/, '').split(/[\\/]+/)) {
    if (part === '..') parts.pop();
    else if (part !== '' && part !== '.') parts.push(part);
  }
  return parts.join('/') || 'artifact';
}
