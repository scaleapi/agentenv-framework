import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  ChevronDown,
  Clipboard,
  Loader2,
  Play,
  Upload,
} from 'lucide-react';
import { MODEL_OPTIONS, SECTION_HEADER_CLASS } from './shared';
import {
  useRunGroupPolling,
  type RunOverrides,
} from '../hooks/useRunGroupPolling';
import {
  MAX_CSV_SIZE,
  MAX_PARALLEL_RUNS,
  appendSeeds,
  countRealSeeds,
  parseSeedsText,
  validateSeeds,
  type ParsedSeeds,
} from '../lib/csv-seeds';
import { extractPlaceholders } from '../lib/placeholders';

const RUN_CONFIG_STORAGE_KEY = 'agent-env-explorer:run-config';

interface PersistedConfig {
  agent_model?: string;
  agent_artifact_id?: string;
  a2a_agent_id?: string;
  extra_overrides_json?: string;
  count?: number;
  concurrency?: number;
}

function loadPersistedConfig(): PersistedConfig {
  try {
    const raw = sessionStorage.getItem(RUN_CONFIG_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as PersistedConfig) : {};
  } catch {
    return {};
  }
}

function savePersistedConfig(config: PersistedConfig) {
  try {
    sessionStorage.setItem(RUN_CONFIG_STORAGE_KEY, JSON.stringify(config));
  } catch (err) {
    console.warn('sessionStorage write for run config failed', err);
  }
}

interface StartRunsPanelProps {
  taskId: string;
  taskVersion: number;
  taskSteps?: ReadonlyArray<Record<string, unknown>>;
  taskProjectId?: string;
  onStarted?: () => void;
  onCompleted?: () => void;
}

// One empty seed row — the grid's starting state for a placeholder task.
function blankSeedGrid(seedValuePlaceholders: string[]): ParsedSeeds {
  const columns = ['id', ...seedValuePlaceholders];
  const row = Object.fromEntries(
    columns.map(c => [c, c === 'id' ? 'row-1' : '']),
  ) as Record<string, string>;
  return { columns, seeds: [row], sourceLines: [-1] };
}

const formatTagList = (placeholders: string[]): string =>
  placeholders.map(ph => `<${ph}>`).join(', ');

export function StartRunsPanel({
  taskId,
  taskVersion,
  taskSteps,
  taskProjectId,
  onStarted,
  onCompleted,
}: StartRunsPanelProps) {
  const [persisted] = useState<PersistedConfig>(loadPersistedConfig);

  const [projectId, setProjectId] = useState<string>(
    taskProjectId?.trim() ?? '',
  );
  const [agentModel, setAgentModel] = useState<string>(
    persisted.agent_model ?? '',
  );
  const [agentArtifactId, setAgentArtifactId] = useState<string>(
    persisted.agent_artifact_id ?? '',
  );
  const [a2aAgentId, setA2aAgentId] = useState<string>(
    persisted.a2a_agent_id ?? '',
  );
  const [extraOverridesJson, setExtraOverridesJson] = useState<string>(
    persisted.extra_overrides_json ?? '',
  );
  // Raw text (not number) so backspacing a digit leaves the field
  // momentarily invalid rather than silent-clamping.
  const [countText, setCountText] = useState<string>(
    String(persisted.count ?? 1),
  );
  const [concurrencyText, setConcurrencyText] = useState<string>(
    String(persisted.concurrency ?? 5),
  );
  const placeholders = useMemo(
    () => extractPlaceholders(taskSteps ?? []),
    [taskSteps],
  );
  const seedValuePlaceholders = useMemo(
    () => placeholders.filter(ph => ph !== 'id'),
    [placeholders],
  );
  // Auto-expand so the count input (which drives the inline form) is visible.
  const [advancedOpen, setAdvancedOpen] = useState<boolean>(
    () => placeholders.length > 0,
  );

  const [parsedSeeds, setParsedSeeds] = useState<ParsedSeeds | null>(null);
  const seeds = parsedSeeds?.seeds ?? null;
  const seedColumns = parsedSeeds?.columns ?? [];
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [pasteText, setPasteText] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { status, runGroup, startResponse, error, startRuns } =
    useRunGroupPolling(taskId);

  const notifiedStartRef = useRef<string | null>(null);
  useEffect(() => {
    if (!startResponse) {
      notifiedStartRef.current = null;
      return;
    }
    if (notifiedStartRef.current === startResponse.run_group_id) return;
    notifiedStartRef.current = startResponse.run_group_id;
    onStarted?.();
  }, [startResponse, onStarted]);

  const parsedCount = parsePositiveIntInRange(countText, 1, MAX_PARALLEL_RUNS);
  const parsedConcurrency = parsePositiveIntInRange(
    concurrencyText,
    1,
    MAX_PARALLEL_RUNS,
  );

  // Placeholder tasks show the seed grid; seed one blank row so the user can start typing without pasting first. Only when empty.
  useEffect(() => {
    if (placeholders.length === 0) return;
    setParsedSeeds(prev => prev ?? blankSeedGrid(seedValuePlaceholders));
  }, [placeholders.length, seedValuePlaceholders]);

  useEffect(() => {
    savePersistedConfig({
      agent_model: agentModel || undefined,
      agent_artifact_id: agentArtifactId || undefined,
      a2a_agent_id: a2aAgentId || undefined,
      extra_overrides_json: extraOverridesJson || undefined,
      count: parsedCount ?? undefined,
      concurrency: parsedConcurrency ?? undefined,
    });
  }, [
    agentModel,
    agentArtifactId,
    a2aAgentId,
    extraOverridesJson,
    parsedCount,
    parsedConcurrency,
  ]);

  // Fire onCompleted only on polling → done/error. A starting → error means POST failed before a workflow existed, so there's nothing to refetch.
  const prevStatus = useRef(status);
  useEffect(() => {
    if (
      prevStatus.current === 'polling' &&
      (status === 'done' || status === 'error')
    ) {
      onCompleted?.();
    }
    prevStatus.current = status;
  }, [status, onCompleted]);

  // Paste / upload append onto the grid rather than replacing it (Clear resets).
  const loadSeeds = useCallback(
    (text: string) => {
      const added = parseSeedsText(text);
      setUploadError(
        countRealSeeds(parsedSeeds) + added.seeds.length > MAX_PARALLEL_RUNS
          ? `Capped at ${MAX_PARALLEL_RUNS} runs; extra rows were dropped.`
          : null,
      );
      setParsedSeeds(prev => appendSeeds(prev, added));
    },
    [parsedSeeds],
  );

  const handleFileUpload = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      if (!file.name.endsWith('.csv')) {
        setUploadError('File must be a .csv');
        return;
      }
      if (file.size > MAX_CSV_SIZE) {
        setUploadError(`CSV exceeds ${MAX_CSV_SIZE / (1024 * 1024)} MB limit`);
        return;
      }
      setUploading(true);

      const reader = new FileReader();
      reader.onload = () => {
        try {
          loadSeeds(reader.result as string);
        } catch (err) {
          setUploadError(
            err instanceof Error ? err.message : 'Failed to parse CSV',
          );
        }
      };
      reader.onerror = () => setUploadError('Failed to read file');
      // Reset the input so re-selecting the same file fires `change` again.
      reader.onloadend = () => {
        setUploading(false);
        if (fileInputRef.current) fileInputRef.current.value = '';
      };
      reader.readAsText(file);
    },
    [loadSeeds],
  );

  const handlePasteLoad = useCallback(() => {
    if (!pasteText.trim()) return;
    try {
      loadSeeds(pasteText);
      setPasteText('');
    } catch (err) {
      setUploadError(
        err instanceof Error ? err.message : 'Failed to parse pasted rows',
      );
    }
  }, [pasteText, loadSeeds]);

  const clearSeeds = useCallback(() => {
    // Placeholder tasks reset to a fresh blank row (grid stays); count-only
    // tasks have no grid, so clear to null.
    setParsedSeeds(
      placeholders.length > 0 ? blankSeedGrid(seedValuePlaceholders) : null,
    );
    setUploadError(null);
  }, [placeholders.length, seedValuePlaceholders]);

  const updateSeedCell = useCallback(
    (rowIdx: number, col: string, value: string) => {
      setParsedSeeds(prev =>
        prev
          ? {
              ...prev,
              seeds: prev.seeds.map((r, i) =>
                i === rowIdx ? { ...r, [col]: value } : r,
              ),
            }
          : prev,
      );
    },
    [],
  );

  const addSeedRow = useCallback(() => {
    setParsedSeeds(prev => {
      if (!prev || prev.seeds.length >= MAX_PARALLEL_RUNS) return prev;
      const blank = Object.fromEntries(
        prev.columns.map(c => [c, '']),
      ) as Record<string, string>;
      if (prev.columns.includes('id')) {
        const existing = new Set(prev.seeds.map(r => r.id));
        let n = prev.seeds.length + 1;
        while (existing.has(`row-${n}`)) n++;
        blank.id = `row-${n}`;
      }
      // Unique non-positive sentinel: distinguishes synthetic rows from parsed
      // ones (positive line numbers) and stays unique for stable React keys.
      const nextLine = Math.min(0, ...prev.sourceLines) - 1;
      return {
        ...prev,
        seeds: [...prev.seeds, blank],
        sourceLines: [...prev.sourceLines, nextLine],
      };
    });
  }, []);

  const removeSeedRow = useCallback((rowIdx: number) => {
    setParsedSeeds(prev =>
      prev
        ? {
            ...prev,
            seeds: prev.seeds.filter((_, i) => i !== rowIdx),
            sourceLines: prev.sourceLines.filter((_, i) => i !== rowIdx),
          }
        : prev,
    );
  }, []);

  const requiresSeeds = placeholders.length > 0;
  const effectiveCount = requiresSeeds ? seeds?.length ?? 0 : parsedCount ?? 0;

  // Surface seed columns that don't satisfy the task's placeholders — the worker leaves unmatched `<key>` tokens literal, silently breaking runs.
  const csvMissingPlaceholders = useMemo(() => {
    if (!seeds || placeholders.length === 0) return [];
    const cols = new Set(seedColumns);
    return seedValuePlaceholders.filter(ph => !cols.has(ph));
  }, [seeds, seedColumns, placeholders, seedValuePlaceholders]);

  // Re-validate on every edit (seeds are editable): beyond id validity, every placeholder column must be filled or a literal `<token>` reaches the prompt.
  const seedsError = useMemo(() => {
    if (!parsedSeeds) return null;
    try {
      validateSeeds(parsedSeeds);
    } catch (err) {
      return err instanceof Error ? err.message : 'Invalid seeds';
    }
    for (let i = 0; i < parsedSeeds.seeds.length; i++) {
      for (const ph of seedValuePlaceholders) {
        if (!parsedSeeds.seeds[i]![ph]?.trim()) {
          return `Row ${i + 1} is missing a value for <${ph}>`;
        }
      }
    }
    return null;
  }, [parsedSeeds, seedValuePlaceholders]);

  // Parse the freeform JSON; backend accepts any shape via `extra="allow"`
  // so we just forward whatever parses, and surface a UI error otherwise.
  const { extraOverrides, extraOverridesError } = useMemo(() => {
    const text = extraOverridesJson.trim();
    if (!text) return { extraOverrides: {}, extraOverridesError: null };
    try {
      const parsed: unknown = JSON.parse(text);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        return {
          extraOverrides: {},
          extraOverridesError: 'Must be a JSON object',
        };
      }
      return {
        extraOverrides: parsed as Record<string, unknown>,
        extraOverridesError: null,
      };
    } catch (err) {
      return {
        extraOverrides: {},
        extraOverridesError:
          err instanceof Error ? err.message : 'Invalid JSON',
      };
    }
  }, [extraOverridesJson]);

  const buildOverrides = useCallback((): RunOverrides => {
    // priority=0 (interactive): a human watches the live stream. Freeform extras can override (e.g. {"priority":1}); dropdowns win last.
    const o = { priority: 0, ...extraOverrides } as RunOverrides;
    if (agentModel.trim()) o.agent_model = agentModel.trim();
    if (agentArtifactId.trim()) o.agent_artifact_id = agentArtifactId.trim();
    if (a2aAgentId.trim()) o.a2a_agent_id = a2aAgentId.trim();
    return o;
  }, [extraOverrides, agentModel, agentArtifactId, a2aAgentId]);

  const canStart =
    parsedConcurrency !== null &&
    extraOverridesError === null &&
    (requiresSeeds
      ? effectiveCount >= 1 &&
        seedsError === null &&
        csvMissingPlaceholders.length === 0
      : parsedCount !== null);

  // Clamp concurrency to the visible count after a seeds upload shrinks it.
  useEffect(() => {
    if (
      parsedConcurrency !== null &&
      parsedConcurrency > effectiveCount &&
      effectiveCount >= 1
    ) {
      setConcurrencyText(String(effectiveCount));
    }
  }, [parsedConcurrency, effectiveCount]);

  const handleStart = useCallback(async () => {
    if (!canStart) return;
    // Placeholder tasks submit the grid rows; count-only tasks submit a count.
    const submitSeeds = requiresSeeds ? seeds ?? undefined : undefined;
    await startRuns({
      version: taskVersion,
      projectId: projectId.trim() || undefined,
      overrides: buildOverrides(),
      count: submitSeeds ? undefined : parsedCount ?? undefined,
      seeds: submitSeeds,
      concurrency: parsedConcurrency ?? undefined,
    });
  }, [
    canStart,
    startRuns,
    taskVersion,
    projectId,
    buildOverrides,
    requiresSeeds,
    seeds,
    parsedCount,
    parsedConcurrency,
  ]);

  const n = requiresSeeds ? seeds?.length ?? 0 : parsedCount;
  const countLabel =
    n !== null && n !== undefined ? `${n} Run${n === 1 ? '' : 's'}` : 'Run';
  const startTitle = startDisabledReason(
    parsedConcurrency,
    extraOverridesError,
    requiresSeeds,
    effectiveCount,
    csvMissingPlaceholders,
    seedsError,
    parsedCount,
  );

  // Disable Start between submit and first confirmed `running` so a double-click can't kick off two groups. `provisioning` is excluded (running/completed are both 0 then).
  const isWaitingToStart =
    status === 'starting' ||
    (status === 'polling' &&
      (runGroup?.running ?? 0) + (runGroup?.completed ?? 0) === 0);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        <h3 className={SECTION_HEADER_CLASS}>Start Runs</h3>

        <button
          onClick={handleStart}
          disabled={!canStart || isWaitingToStart}
          title={startTitle}
          className="flex items-center gap-1 px-2.5 py-0.5 rounded-md border border-[var(--border)] text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors disabled:opacity-50"
        >
          {isWaitingToStart ? (
            <>
              <Loader2 size={12} className="animate-spin" />
              Waiting to start…
            </>
          ) : (
            <>
              <Play size={12} />
              Start {countLabel}
            </>
          )}
        </button>

        <button
          onClick={() => setAdvancedOpen(!advancedOpen)}
          className="flex items-center gap-1 text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors"
          title="Advanced configuration"
        >
          Advanced config
          <ChevronDown
            size={12}
            className={`transition-transform ${
              advancedOpen ? 'rotate-180' : ''
            }`}
          />
        </button>
      </div>

      {requiresSeeds && parsedSeeds && (
        <SeedGrid
          parsedSeeds={parsedSeeds}
          placeholders={placeholders}
          seedsError={seedsError}
          csvMissingPlaceholders={csvMissingPlaceholders}
          onUpdateCell={updateSeedCell}
          onAddRow={addSeedRow}
          onRemoveRow={removeSeedRow}
          onClear={clearSeeds}
          footer={
            <SeedsSection
              uploading={uploading}
              uploadError={uploadError}
              fileInputRef={fileInputRef}
              onFileUpload={handleFileUpload}
              pasteText={pasteText}
              setPasteText={setPasteText}
              onPasteLoad={handlePasteLoad}
            />
          }
        />
      )}

      {advancedOpen && (
        <div className="space-y-3">
          <AdvancedConfig
            requiresSeeds={requiresSeeds}
            countText={countText}
            setCountText={setCountText}
            parsedCount={parsedCount}
            concurrencyText={concurrencyText}
            setConcurrencyText={setConcurrencyText}
            parsedConcurrency={parsedConcurrency}
            effectiveCount={effectiveCount}
            agentModel={agentModel}
            setAgentModel={setAgentModel}
            agentArtifactId={agentArtifactId}
            setAgentArtifactId={setAgentArtifactId}
            a2aAgentId={a2aAgentId}
            setA2aAgentId={setA2aAgentId}
            extraOverridesJson={extraOverridesJson}
            setExtraOverridesJson={setExtraOverridesJson}
            extraOverridesError={extraOverridesError}
          />
        </div>
      )}

      {status === 'error' && error && (
        <div
          role="alert"
          className="text-xs text-red-500 bg-red-500/10 border border-red-500/30 rounded-md px-2 py-1"
        >
          {error}
        </div>
      )}
    </div>
  );
}

function AdvancedConfig({
  requiresSeeds,
  countText,
  setCountText,
  parsedCount,
  concurrencyText,
  setConcurrencyText,
  parsedConcurrency,
  effectiveCount,
  agentModel,
  setAgentModel,
  agentArtifactId,
  setAgentArtifactId,
  a2aAgentId,
  setA2aAgentId,
  extraOverridesJson,
  setExtraOverridesJson,
  extraOverridesError,
}: {
  requiresSeeds: boolean;
  countText: string;
  setCountText: (v: string) => void;
  parsedCount: number | null;
  concurrencyText: string;
  setConcurrencyText: (v: string) => void;
  parsedConcurrency: number | null;
  effectiveCount: number;
  agentModel: string;
  setAgentModel: (v: string) => void;
  agentArtifactId: string;
  setAgentArtifactId: (v: string) => void;
  a2aAgentId: string;
  setA2aAgentId: (v: string) => void;
  extraOverridesJson: string;
  setExtraOverridesJson: (v: string) => void;
  extraOverridesError: string | null;
}) {
  const fieldInput =
    'w-full px-1.5 py-0.5 rounded border border-[var(--border)] bg-[var(--background)] text-xs';
  return (
    <div className="space-y-2 max-w-xl p-3 rounded-md border border-[var(--border)] bg-[var(--secondary)]/40">
      <div className="grid grid-cols-2 gap-2">
        {!requiresSeeds && (
          <FieldLabel label={`Total runs (1–${MAX_PARALLEL_RUNS})`}>
            <input
              type="text"
              inputMode="numeric"
              value={countText}
              onChange={e => setCountText(onlyDigits(e.target.value))}
              aria-invalid={parsedCount === null}
              className={`${fieldInput} ${
                parsedCount === null ? 'border-red-500' : ''
              }`}
            />
          </FieldLabel>
        )}
        <FieldLabel
          label={`Max concurrency (1–${MAX_PARALLEL_RUNS}) — ignored locally`}
          title="Inherited from the hosted hub's run API and currently ignored by the local runner: runs are started sequentially and execution is bounded by [runner.config] workers in .agentenv/config.toml, not by this field."
        >
          <input
            type="text"
            inputMode="numeric"
            value={concurrencyText}
            onChange={e => setConcurrencyText(onlyDigits(e.target.value))}
            aria-invalid={parsedConcurrency === null}
            className={`${fieldInput} ${
              parsedConcurrency === null ? 'border-red-500' : ''
            }`}
          />
        </FieldLabel>
        <FieldLabel label="Agent model">
          <select
            value={agentModel}
            onChange={e => setAgentModel(e.target.value)}
            className={fieldInput}
          >
            <option value="">(task default)</option>
            {MODEL_OPTIONS.map(m => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </FieldLabel>
        <FieldLabel label="Artifact ID">
          <input
            type="text"
            value={agentArtifactId}
            onChange={e => setAgentArtifactId(e.target.value)}
            placeholder="(task default)"
            className={fieldInput}
          />
        </FieldLabel>
      </div>
      <FieldLabel label="A2A agent ID">
        <input
          type="text"
          value={a2aAgentId}
          onChange={e => setA2aAgentId(e.target.value)}
          placeholder="(none)"
          className={fieldInput}
        />
      </FieldLabel>
      <FieldLabel
        label="Extra overrides (JSON)"
        title="Free-form JSON object merged into the request overrides. Dropdown fields above take precedence on key collisions. Fields the worker recognizes are used; the rest are ignored server-side."
      >
        <textarea
          value={extraOverridesJson}
          onChange={e => setExtraOverridesJson(e.target.value)}
          placeholder='e.g. {"maxTokens": 4000}'
          rows={3}
          className={`${fieldInput} font-mono resize-y`}
        />
        {extraOverridesError && (
          <span className="text-red-500 text-xs">
            Invalid JSON: {extraOverridesError}
          </span>
        )}
      </FieldLabel>
    </div>
  );
}

function FieldLabel({
  label,
  title,
  children,
}: {
  label: string;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <label
      className="flex flex-col gap-1 text-xs text-[var(--muted-foreground)]"
      title={title}
    >
      {label}
      {children}
    </label>
  );
}

function SeedsSection({
  uploading,
  uploadError,
  fileInputRef,
  onFileUpload,
  pasteText,
  setPasteText,
  onPasteLoad,
}: {
  uploading: boolean;
  uploadError: string | null;
  fileInputRef: React.RefObject<HTMLInputElement>;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  pasteText: string;
  setPasteText: (v: string) => void;
  onPasteLoad: () => void;
}) {
  const [pasteOpen, setPasteOpen] = useState(false);
  const buttonCls =
    'flex items-center gap-1 px-2 py-0.5 rounded-md border border-[var(--border)] text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors disabled:opacity-40';
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 flex-wrap">
        <label className={`${buttonCls} cursor-pointer`}>
          <Upload size={12} />
          {uploading ? 'Parsing…' : 'Upload CSV'}
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            disabled={uploading}
            onChange={onFileUpload}
          />
        </label>
        <button
          type="button"
          onClick={() => setPasteOpen(o => !o)}
          className={buttonCls}
        >
          <Clipboard size={12} />
          Paste rows
        </button>
        {uploadError && (
          <span className="text-xs text-red-500">{uploadError}</span>
        )}
      </div>
      {pasteOpen && (
        <div className="space-y-2">
          <div className="text-xs text-[var(--muted-foreground)]">
            CSV or tab-separated rows from a sheet; first row is the header. An{' '}
            <code>id</code> column is added automatically if absent. Rows are
            appended to the grid — use Clear to reset.
          </div>
          <textarea
            value={pasteText}
            onChange={e => setPasteText(e.target.value)}
            placeholder={'repo_url\nhttps://github.com/pytorch/pytorch'}
            rows={4}
            className="w-full px-1.5 py-1 rounded border border-[var(--border)] bg-[var(--background)] text-xs font-mono resize-y"
          />
          <button
            type="button"
            onClick={onPasteLoad}
            disabled={!pasteText.trim()}
            className={buttonCls}
          >
            Add pasted rows
          </button>
        </div>
      )}
    </div>
  );
}

function MissingPlaceholdersAlert({ missing }: { missing: string[] }) {
  if (missing.length === 0) return null;
  const plural = missing.length === 1 ? '' : 's';
  return (
    <div
      role="alert"
      className="text-xs text-red-500 bg-red-500/10 border border-red-500/30 rounded px-2 py-1"
    >
      Missing column{plural} for placeholder{plural}: {formatTagList(missing)}.
      Add the column{plural} — otherwise the{' '}
      {plural ? 'tokens stay' : 'token stays'} literal in the prompt.
    </div>
  );
}

// The single seed surface: hand-add rows, or prefill via paste/CSV. Edits write
// back to the same `parsedSeeds` the run submits.
function SeedGrid({
  parsedSeeds,
  placeholders,
  seedsError,
  csvMissingPlaceholders,
  onUpdateCell,
  onAddRow,
  onRemoveRow,
  onClear,
  footer,
}: {
  parsedSeeds: ParsedSeeds;
  placeholders: string[];
  seedsError: string | null;
  csvMissingPlaceholders: string[];
  onUpdateCell: (rowIdx: number, col: string, value: string) => void;
  onAddRow: () => void;
  onRemoveRow: (rowIdx: number) => void;
  onClear: () => void;
  footer: React.ReactNode;
}) {
  const { columns, seeds, sourceLines } = parsedSeeds;
  const inputCls =
    'w-full px-1.5 py-0.5 rounded border border-[var(--border)] bg-[var(--background)] text-xs';
  return (
    <div className="space-y-2 max-w-4xl p-3 rounded-md border border-[var(--border)] bg-[var(--secondary)]/40">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="text-xs text-[var(--muted-foreground)]">
          {seeds.length} run{seeds.length === 1 ? '' : 's'} ·{' '}
          {formatTagList(placeholders)}
        </div>
        <button
          type="button"
          onClick={onClear}
          className="text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
        >
          Clear
        </button>
      </div>
      <MissingPlaceholdersAlert missing={csvMissingPlaceholders} />
      {seedsError && (
        <div
          role="alert"
          className="text-xs text-red-500 bg-red-500/10 border border-red-500/30 rounded px-2 py-1"
        >
          {seedsError}
        </div>
      )}
      <div className="overflow-auto max-h-80">
        <table className="text-xs w-full">
          <thead>
            <tr className="text-left text-[var(--muted-foreground)]">
              {columns.map(col => (
                <th key={col} className="px-2 pb-1 font-medium">
                  {col}
                </th>
              ))}
              <th className="pb-1" aria-label="Remove row" />
            </tr>
          </thead>
          <tbody>
            {seeds.map((row, i) => (
              <tr key={sourceLines[i] ?? i}>
                {columns.map(col => (
                  <td key={col} className="px-2 pb-1 align-top">
                    <input
                      type="text"
                      value={row[col] ?? ''}
                      onChange={e => onUpdateCell(i, col, e.target.value)}
                      aria-label={`Row ${i + 1} ${col}`}
                      className={inputCls}
                    />
                  </td>
                ))}
                <td className="pb-1 align-top">
                  <button
                    type="button"
                    onClick={() => onRemoveRow(i)}
                    aria-label={`Remove row ${i + 1}`}
                    className="px-1 text-[var(--muted-foreground)] hover:text-red-500"
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button
        type="button"
        onClick={onAddRow}
        disabled={seeds.length >= MAX_PARALLEL_RUNS}
        className="text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] disabled:opacity-40"
      >
        + Add row
      </button>
      <div className="pt-1 border-t border-[var(--border)]">{footer}</div>
    </div>
  );
}

function startDisabledReason(
  parsedConcurrency: number | null,
  extraOverridesError: string | null,
  requiresSeeds: boolean,
  effectiveCount: number,
  csvMissingPlaceholders: string[],
  seedsError: string | null,
  parsedCount: number | null,
): string | undefined {
  if (parsedConcurrency === null) {
    return `Max concurrency must be a whole number between 1 and ${MAX_PARALLEL_RUNS}`;
  }
  if (extraOverridesError)
    return `Fix extra overrides JSON: ${extraOverridesError}`;
  if (requiresSeeds) {
    if (seedsError) return seedsError;
    if (csvMissingPlaceholders.length > 0) {
      return `Missing column${
        csvMissingPlaceholders.length === 1 ? '' : 's'
      } for placeholder${
        csvMissingPlaceholders.length === 1 ? '' : 's'
      }: ${csvMissingPlaceholders.map(ph => `<${ph}>`).join(', ')}`;
    }
    if (effectiveCount < 1) return 'Add at least one seed row before starting';
  } else if (parsedCount === null) {
    return `Total runs must be a whole number between 1 and ${MAX_PARALLEL_RUNS}`;
  }
  return undefined;
}

/** Strip everything that isn't a 0-9 digit — handles both typing and paste. */
function onlyDigits(value: string): string {
  return value.replace(/\D/g, '');
}

/** Integer value of `text` if a clean positive integer in [min, max], else null. Strict (rejects empty,
 *  decimals, negatives, spaces, hex) so the call site can block submit. */
function parsePositiveIntInRange(
  text: string,
  min: number,
  max: number,
): number | null {
  if (!/^\d+$/.test(text)) return null;
  const n = Number(text);
  if (!Number.isInteger(n) || n < min || n > max) return null;
  return n;
}
