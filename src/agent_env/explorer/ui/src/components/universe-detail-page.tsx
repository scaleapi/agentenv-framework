import { useEffect, useState } from 'react';
import {
  ArrowLeft,
  FlaskConical,
  Download,
  ChevronDown,
} from 'lucide-react';
import {
  BACKEND_URL,
  apiFetch,
  MetadataTable,
  formatCellValue,
  objectContentUrl,
} from './shared';
import { isEnvironmentUniverseType } from '../lib/universe-types';
import { safeHref } from '../lib/safe-url';

type VersionEntry = {
  version: number;
  type?: string;
  created_at_utc?: string;
};

export function UniverseDetailPage({
  universeId,
  initialVersion,
  onBack,
}: {
  universeId: string;
  initialVersion?: number;
  onBack: () => void;
}) {
  const [artifact, setArtifact] = useState<Record<string, unknown> | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [selectedVersion, setSelectedVersion] = useState<number | undefined>(
    initialVersion,
  );
  const [versions, setVersions] = useState<VersionEntry[]>([]);
  const [showVersionMenu, setShowVersionMenu] = useState(false);
  const [editing, setEditing] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    apiFetch(`${BACKEND_URL}/api/v1/artifacts/${encodeURIComponent(universeId)}/versions`)
      .then(res => (res.ok ? res.json() : []))
      .then((list: VersionEntry[]) => {
        setVersions(Array.isArray(list) ? list : []);
      })
      .catch(() => setVersions([]));
  }, [universeId, reloadKey]);

  useEffect(() => {
    setLoading(true);
    const url =
      selectedVersion != null
        ? `${BACKEND_URL}/api/v1/artifacts/${encodeURIComponent(universeId)}?version=${selectedVersion}`
        : `${BACKEND_URL}/api/v1/artifacts/${encodeURIComponent(universeId)}`;
    apiFetch(url)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to fetch (${res.status})`);
        return res.json();
      })
      .then(data => {
        setArtifact(data);
        setLoading(false);
      })
      .catch(e => {
        setFetchError(e instanceof Error ? e.message : 'Failed to load');
        setLoading(false);
      });
  }, [universeId, selectedVersion, reloadKey]);

  const services = (artifact?.services ?? []) as {
    artifact_id: string;
    service_name: string;
    version?: number;
  }[];
  const files = (artifact?.files ?? []) as {
    filename: string;
    artifact_id: string;
    version?: number;
    content_type?: string;
    object_url?: string;
  }[];
  const metadataRefs = (artifact?.metadata_refs ?? null) as Record<
    string,
    { id: string; version: number }
  > | null;
  const metadataDisplay = metadataRefs
    ? Object.fromEntries(
        Object.entries(metadataRefs).map(([k, ref]) => [
          k,
          `${ref.id} @ v${ref.version}`,
        ]),
      )
    : ((artifact?.metadata ?? null) as Record<string, string> | null);
  const artifactType = String(artifact?.type ?? '');
  const isEnvironmentUniverse = isEnvironmentUniverseType(artifactType);
  const currentVersion = artifact?.version as number | undefined;
  const isLatest =
    versions.length > 0 && currentVersion === versions[0]?.version;

  return (
    <div className="p-8 flex flex-col h-full">
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 text-sm text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors mb-6"
      >
        <ArrowLeft size={14} />
        Back
      </button>

      {loading && (
        <p className="text-sm text-[var(--muted-foreground)]">Loading...</p>
      )}
      {fetchError && <p className="text-sm text-red-500">{fetchError}</p>}

      {artifact && !loading && (
        <>
          <div className="mb-6">
            <div className="flex items-baseline gap-3 flex-wrap">
              <h1 className="text-2xl font-semibold font-mono">{universeId}</h1>

              {versions.length > 1 ? (
                <div className="relative">
                  <button
                    onClick={() => setShowVersionMenu(v => !v)}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded border border-[var(--border)] text-sm text-[var(--muted-foreground)] hover:bg-[var(--accent)] transition-colors"
                  >
                    v{String(currentVersion)}
                    {!isLatest && (
                      <span className="ml-1 text-[10px] font-semibold uppercase tracking-wider text-amber-600">
                        older
                      </span>
                    )}
                    <ChevronDown size={12} />
                  </button>
                  {showVersionMenu && (
                    <div className="absolute left-0 top-full mt-1 z-10 min-w-[180px] rounded-md border border-[var(--border)] bg-[var(--background)] shadow-md py-1">
                      {versions.map((v, idx) => {
                        const active = v.version === currentVersion;
                        return (
                          <button
                            key={v.version}
                            onClick={() => {
                              setSelectedVersion(v.version);
                              setShowVersionMenu(false);
                            }}
                            className={`block w-full text-left px-3 py-1.5 text-sm transition-colors ${
                              active
                                ? 'bg-[var(--secondary)] text-[var(--foreground)] font-medium'
                                : 'text-[var(--muted-foreground)] hover:bg-[var(--accent)] hover:text-[var(--foreground)]'
                            }`}
                          >
                            v{v.version}
                            {idx === 0 && (
                              <span className="ml-2 text-[10px] uppercase tracking-wider text-[var(--muted-foreground)]">
                                latest
                              </span>
                            )}
                            {v.created_at_utc && (
                              <span className="block text-[10px] text-[var(--muted-foreground)]">
                                {formatCellValue(
                                  'created_at_utc',
                                  v.created_at_utc,
                                )}
                              </span>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              ) : (
                <span className="text-sm text-[var(--muted-foreground)]">
                  v{String(currentVersion)}
                </span>
              )}

              <span className="px-2 py-0.5 rounded text-xs font-medium bg-[var(--secondary)] text-[var(--foreground)]">
                {artifactType}
              </span>
            </div>
            {!!artifact.created_at_utc && (
              <p className="text-sm text-[var(--muted-foreground)] mt-1">
                Last Modified:{' '}
                {formatCellValue('created_at_utc', artifact.created_at_utc)}
              </p>
            )}

          </div>

          {services.length > 0 ? (
            <div className="mb-6">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
                Services ({services.length})
              </h3>
              <div className="flex flex-wrap gap-2">
                {services.map(s => (
                  <span
                    key={s.artifact_id}
                    className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm bg-[var(--secondary)]"
                  >
                    <span className="font-medium text-[var(--foreground)]">
                      {s.service_name}
                    </span>
                    <span className="text-[var(--muted-foreground)] font-mono text-xs">
                      {s.artifact_id}
                      {s.version != null && ` v${s.version}`}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          ) : null}

          {files.length > 0 && (
            <div className="mb-6">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
                Files ({files.length})
              </h3>
              <div className="flex flex-col gap-1.5 max-w-3xl">
                {files.map(f => (
                  <div
                    key={f.filename}
                    className="flex items-center justify-between gap-3 px-3.5 py-2 rounded-lg bg-[var(--secondary)]"
                  >
                    <div className="flex flex-col min-w-0">
                      <span className="font-mono text-sm text-[var(--foreground)] truncate">
                        {f.filename}
                      </span>
                      <span className="text-[var(--muted-foreground)] font-mono text-xs truncate">
                        {f.artifact_id}
                        {f.version != null ? ` v${f.version}` : ''}
                        {f.content_type ? ` · ${f.content_type}` : ''}
                      </span>
                    </div>
                    {f.object_url && (
                      <a
                        href={objectContentUrl(f.object_url)}
                        download={f.filename}
                        className="shrink-0 inline-flex items-center gap-1 px-2 py-1 rounded text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors"
                        title={`Download ${f.filename}`}
                      >
                        <Download size={12} />
                        Download
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {artifactType === 'coding_task_harbor' && (
            <CodingTaskHarborDetails
              artifact={artifact}
              universeId={universeId}
              selectedVersion={selectedVersion}
            />
          )}

          {metadataDisplay && <MetadataTable metadata={metadataDisplay} />}
        </>
      )}

    </div>
  );
}

function CodingTaskHarborDetails({
  artifact,
  universeId,
  selectedVersion,
}: {
  artifact: Record<string, unknown>;
  universeId: string;
  selectedVersion?: number;
}) {
  const bundle = (artifact.bundle ?? {}) as Record<string, string | undefined>;
  const mirror = (artifact.mirror_metadata ?? {}) as Record<string, unknown>;
  const harbor = (artifact.harbor ?? {}) as Record<string, unknown>;
  const milestones = (artifact.milestones ?? {}) as Record<string, string>;

  const owner = String(mirror.owner ?? artifact.owner ?? '');
  const repo = String(mirror.repo ?? artifact.repo ?? '');
  const baseCommit = String(mirror.base_commit ?? artifact.base_commit ?? '');
  const repoUrl = String(mirror.repo_url ?? artifact.repo_url ?? '');
  const harvestId = String(mirror.harvest_id ?? '');
  const taskName = String(harbor.name ?? '');
  const taskDescription = String(harbor.description ?? '');

  const bundlePieces = (
    [
      ['Dockerfile', 'docker_file'],
      ['Image', 'docker_image'],
      ['Interface', 'interface_md'],
      ['Run script', 'run_script'],
      ['Golden patch', 'golden_patch'],
      ['Test patch', 'test_patch'],
    ] as const
  ).filter(([, key]) => !!bundle[key]);

  return (
    <>
      {(owner || repo || baseCommit || repoUrl) && (
        <div className="mb-6">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
            Source
          </h3>
          <div className="flex flex-col gap-1 text-sm font-mono">
            {(owner || repo) && (
              <div>
                <span className="text-[var(--muted-foreground)]">repo: </span>
                <span className="text-[var(--foreground)]">
                  {owner && repo ? `${owner}/${repo}` : owner || repo}
                </span>
              </div>
            )}
            {baseCommit && (
              <div>
                <span className="text-[var(--muted-foreground)]">
                  base_commit:{' '}
                </span>
                <span className="text-[var(--foreground)]">{baseCommit}</span>
              </div>
            )}
            {repoUrl && (
              <div>
                <span className="text-[var(--muted-foreground)]">url: </span>
                <a
                  href={safeHref(repoUrl)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[var(--foreground)] underline hover:text-violet-600"
                >
                  {repoUrl}
                </a>
              </div>
            )}
            {harvestId && (
              <div>
                <span className="text-[var(--muted-foreground)]">
                  harvest_id:{' '}
                </span>
                <span className="text-[var(--foreground)]">{harvestId}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {(taskName || taskDescription) && (
        <div className="mb-6">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
            Task
          </h3>
          {taskName && (
            <div className="text-sm font-mono text-[var(--foreground)]">
              {taskName}
            </div>
          )}
          {taskDescription && (
            <p className="text-sm text-[var(--muted-foreground)] mt-1 whitespace-pre-wrap">
              {taskDescription}
            </p>
          )}
        </div>
      )}

      {bundlePieces.length > 0 && (
        <div className="mb-6">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
            Bundle ({bundlePieces.length})
          </h3>
          <div className="flex flex-col gap-1.5 max-w-3xl">
            {bundlePieces.map(([label, key]) => (
              <div
                key={key}
                className="flex items-center justify-between gap-3 px-3.5 py-2 rounded-lg bg-[var(--secondary)]"
              >
                <div className="flex flex-col min-w-0">
                  <span className="text-sm text-[var(--foreground)]">
                    {label}
                  </span>
                  <a
                    href={safeHref(bundle[key])}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[var(--muted-foreground)] font-mono text-xs truncate hover:underline"
                  >
                    {bundle[key]}
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {Object.keys(milestones).length > 0 && (
        <div className="mb-6">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
            Milestones ({Object.keys(milestones).length})
          </h3>
          <div className="flex flex-col gap-1.5 max-w-3xl">
            {Object.entries(milestones).map(([k, v]) => (
              <div
                key={k}
                className="flex items-center justify-between gap-3 px-3.5 py-2 rounded-lg bg-[var(--secondary)]"
              >
                <span className="font-mono text-sm text-[var(--foreground)] truncate">
                  {k}
                </span>
                <a
                  href={safeHref(v)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[var(--muted-foreground)] font-mono text-xs truncate hover:underline"
                >
                  {v}
                </a>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
