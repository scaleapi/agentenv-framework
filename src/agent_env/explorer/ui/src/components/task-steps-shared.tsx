import { useState, useEffect, useCallback, useId } from 'react';
import {
  X,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  Plus,
  ArrowUp,
  ArrowDown,
  GripVertical,
} from 'lucide-react';
import { Select } from '@radix-ui/themes';
import { MODEL_OPTIONS, BACKEND_URL, apiFetch } from './shared';
import { EnvironmentsPage, ENV_SECTIONS } from './environments-page';
import { UniversesPage } from './universes-page';

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

export const STEP_COLORS: Record<string, string> = {
  deploy_env: '#3b82f6',
  load_artifact: '#8b5cf6',
  deploy_agent: '#06b6d4',
  prompt_agent: '#f59e0b',
  collect_artifacts: '#14b8a6',
  rubrics_verifier: '#10b981',
  env_outcome_verifier: '#10b981',
  cua_initialize: '#ec4899',
  cua_evaluate: '#ef4444',
  prompt_usersim: '#d946ef',
  build_interactive_solver_image: '#0ea5e9',
  interactive_usersim: '#a855f7',
  interactive_usersim_deploy: '#a855f7',
  human_interaction: '#f97316',
  run_code: '#84cc16',
  publish: '#6366f1',
  register_env_triggers: '#0d9488',
  register_agent_triggers: '#c026d3',
  sync_env_clock: '#0891b2',
};

export const STEP_TYPES = [
  'deploy_env',
  'deploy_agent',
  'load_artifact',
  'prompt_agent',
  'collect_artifacts',
  'rubrics_verifier',
  'env_outcome_verifier',
  'cua_initialize',
  'cua_evaluate',
  'prompt_usersim',
  'build_interactive_solver_image',
  'interactive_usersim',
  'interactive_usersim_deploy',
  'human_interaction',
  'run_code',
  'publish',
  'register_env_triggers',
  'register_agent_triggers',
  'sync_env_clock',
] as const;

export type StepType = (typeof STEP_TYPES)[number];

export const STEP_TYPE_LABELS: Record<StepType, string> = {
  deploy_env: 'Deploy Env',
  deploy_agent: 'Deploy Agent',
  load_artifact: 'Load Artifact',
  prompt_agent: 'Prompt Agent',
  collect_artifacts: 'Collect Artifacts',
  rubrics_verifier: 'Rubrics Verifier',
  env_outcome_verifier: 'Env Outcome Verifier',
  cua_initialize: 'CUA Initialize',
  cua_evaluate: 'CUA Evaluate',
  prompt_usersim: 'Prompt UserSim',
  build_interactive_solver_image: 'Build Interactive Solver Image',
  interactive_usersim: 'Interactive UserSim',
  interactive_usersim_deploy: 'Interactive UserSim Deploy',
  human_interaction: 'Human Interaction',
  run_code: 'Run Code',
  publish: 'Publish',
  register_env_triggers: 'Register Env Triggers',
  register_agent_triggers: 'Register Agent Triggers',
  sync_env_clock: 'Sync Env Clock',
};

// Trigger-registration and clock steps have no StepFields editor yet
// (rendered, not addable); keep them out of the Add Step menus
// so users can't create unconfigurable empty steps.
export const ADDABLE_STEP_TYPES = STEP_TYPES.filter(
  t =>
    t !== 'register_env_triggers' &&
    t !== 'register_agent_triggers' &&
    t !== 'sync_env_clock',
);

export const SCORE_AGGREGATOR_OPTIONS = [
  'all_pass',
  'any_pass',
  'weighted_average',
];

export const AUTO_ID_STEP_TYPES = new Set<StepType>([
  'deploy_env',
  'load_artifact',
  'prompt_agent',
]);

export const JSON_FIELDS = new Set([
  'criteria',
  'init_config',
  'evaluator',
  'args',
  'options',
  'output_format',
]);

export const EVALUATOR_STEP_TYPES = new Set<StepType>([
  'rubrics_verifier',
  'env_outcome_verifier',
  'cua_evaluate',
]);

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export interface StepState {
  key: number;
  type: StepType;
  id: string;
  fields: Record<string, unknown>;
  advancedJson: string;
  showAdvanced: boolean;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

export function getJsonError(json: string): string | null {
  const trimmed = json.trim();
  if (!trimmed || trimmed === '{}') return null;
  try {
    const parsed = JSON.parse(trimmed);
    if (
      typeof parsed !== 'object' ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      return 'Must be a JSON object';
    }
    return null;
  } catch {
    return 'Invalid JSON';
  }
}

export function randomSuffix(): string {
  return Math.random().toString(36).slice(2, 8);
}

export function makeDefaultStep(type: StepType, key: number): StepState {
  const id = `${type.replace(/_/g, '-')}-${key + 1}`;
  const fields: Record<string, unknown> = {};
  switch (type) {
    case 'deploy_env':
      fields.env_id = '';
      fields.ttl_seconds = 1209600;
      fields.disk_size_gb = 10;
      break;
    case 'deploy_agent':
      fields.agent_name = 'default-agent';
      fields.env_ids = '';
      fields.enable_docker = false;
      break;
    case 'load_artifact':
      fields.env_id = '';
      fields.artifact_id = '';
      fields.artifact_version = null;
      fields.destination_path = '';
      break;
    case 'prompt_agent': {
      const suffix = randomSuffix();
      fields.prompt = '';
      fields.system_prompt = '';
      fields.model = MODEL_OPTIONS[0];
      fields.agent_name = 'default-agent';
      fields.timeout_seconds = 3600;
      fields.agentenv_tools = [];
      fields.max_turns = '';
      fields.effort = '';
      fields.max_thinking_tokens = '';
      fields.output_format = '';
      fields._suffix = suffix;
      return {
        key,
        type,
        id: `prompt-default-agent-${suffix}`,
        fields,
        advancedJson: '{}',
        showAdvanced: false,
      };
    }
    case 'collect_artifacts':
      fields.agent_name = 'default-agent';
      fields.base_path = '/app/artifact';
      fields.manifest_step_id = '';
      fields.exclude_basenames = '';
      fields.artifacts_key = 'expected_artifacts';
      fields.artifact_paths = '';
      break;
    case 'rubrics_verifier':
      fields.prompt_id = '';
      fields.default_model = MODEL_OPTIONS[0];
      fields.score_aggregator = 'all_pass';
      fields.criteria = '[]';
      break;
    case 'env_outcome_verifier':
      fields.env_id = '';
      fields.file_artifact_id = '';
      fields.score_aggregator = 'all_pass';
      break;
    case 'cua_initialize':
      fields.env_id = '';
      fields.init_config = '[]';
      break;
    case 'cua_evaluate':
      fields.env_id = '';
      fields.evaluator = '{}';
      break;
    case 'prompt_usersim':
      fields.usersim_image_id = '';
      fields.usersim_model = MODEL_OPTIONS[2];
      fields.solver_image_id = '';
      fields.task_artifact_id = '';
      fields.timeout_seconds = 28800;
      fields.poll_interval = 30;
      break;
    case 'build_interactive_solver_image':
      fields.task_artifact_id = '';
      fields.output_artifact_id = '';
      fields.language_fallback = '';
      fields.solver_base_image_id = '';
      fields.build_script_path = '';
      break;
    case 'interactive_usersim':
    case 'interactive_usersim_deploy':
      fields.task_artifact_id = '';
      fields.solver_image_id = '';
      fields.userassistant_image_id = 'acc-userassistant';
      fields.proxy_image_id = 'acc-proxy';
      fields.advisor_model = MODEL_OPTIONS[0];
      fields.agent_model = 'anthropic/claude-sonnet-4-6';
      fields.timeout_seconds = 10800;
      fields.model_label = 'a';
      fields.advisor_api_key_secret = '';
      fields.advisor_base_url = '';
      break;
    case 'human_interaction':
      fields.timeout_seconds = 10800;
      break;
    case 'run_code':
      fields.script_artifact_id = '';
      fields.entrypoint = 'run';
      fields.args = '{}';
      fields.env_id = '';
      fields.agent_name = 'default-agent';
      fields.timeout_seconds = 600;
      break;
    case 'publish':
      fields.target = 'harbor';
      fields.collected_artifacts_step_id = '';
      fields.options = '{}';
      break;
  }
  return { key, type, id, fields, advancedJson: '{}', showAdvanced: false };
}

export function getRecommendedStep(steps: StepState[]): StepType | null {
  if (steps.length === 0) return 'deploy_env';

  const types = new Set(steps.map(s => s.type));
  const hasCua = steps.some(
    s => s.type === 'deploy_env' && String(s.fields._env_type) === 'cua',
  );
  const last = steps[steps.length - 1];
  if (!last) return 'deploy_env';
  const lastType = last.type;

  if (lastType === 'deploy_env' && !types.has('load_artifact') && !hasCua)
    return 'load_artifact';
  if (lastType === 'deploy_env' && hasCua && !types.has('cua_initialize'))
    return 'cua_initialize';
  if (
    (lastType === 'deploy_env' ||
      lastType === 'load_artifact' ||
      lastType === 'cua_initialize') &&
    !types.has('deploy_agent')
  )
    return 'deploy_agent';
  if (lastType === 'deploy_agent' && !types.has('prompt_agent'))
    return 'prompt_agent';
  if (lastType === 'prompt_agent' && !hasCua && !types.has('rubrics_verifier'))
    return 'rubrics_verifier';
  if (lastType === 'prompt_agent' && hasCua && !types.has('cua_evaluate'))
    return 'cua_evaluate';
  if (
    lastType === 'build_interactive_solver_image' &&
    !types.has('interactive_usersim') &&
    !types.has('interactive_usersim_deploy')
  )
    return 'interactive_usersim_deploy';
  if (
    lastType === 'interactive_usersim_deploy' &&
    !types.has('human_interaction')
  )
    return 'human_interaction';

  return null;
}

export function stepToDict(step: StepState): Record<string, unknown> {
  const base: Record<string, unknown> = { type: step.type, id: step.id };

  for (const [key, value] of Object.entries(step.fields)) {
    if (key.startsWith('_')) continue;
    if (value === '' || value === null || value === undefined) continue;
    if (Array.isArray(value) && value.length === 0) continue;
    if (
      step.type === 'deploy_agent' &&
      key === 'env_ids' &&
      typeof value === 'string'
    ) {
      base[key] = value
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
      continue;
    }
    if (
      step.type === 'collect_artifacts' &&
      (key === 'artifact_paths' || key === 'exclude_basenames') &&
      typeof value === 'string'
    ) {
      base[key] = value
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
      continue;
    }
    if (JSON_FIELDS.has(key) && typeof value === 'string') {
      try {
        base[key] = JSON.parse(value);
      } catch {
        // object-typed JSON fields fall back to {}, array-typed to []
        base[key] =
          key === 'evaluator' ||
          key === 'args' ||
          key === 'options' ||
          key === 'output_format'
            ? {}
            : [];
      }
      continue;
    }
    base[key] = value;
  }

  try {
    const advanced = JSON.parse(step.advancedJson);
    if (typeof advanced === 'object' && advanced !== null) {
      Object.assign(base, advanced);
    }
  } catch {
    /* ignore invalid advanced JSON */
  }

  base.type = step.type;
  base.id = step.id;
  return base;
}

export function stepFromDict(
  dict: Record<string, unknown>,
  key: number,
): StepState {
  const type = dict.type as StepType;
  const id = String(dict.id ?? '');
  const knownKeys = new Set(['type', 'id', 'version']);
  const fields: Record<string, unknown> = {};
  const advanced: Record<string, unknown> = {};

  const structured = new Set(Object.keys(makeDefaultStep(type, 0).fields));
  for (const [k, v] of Object.entries(dict)) {
    if (knownKeys.has(k)) continue;
    if (structured.has(k)) {
      if (k === 'env_ids' && Array.isArray(v)) {
        fields[k] = v.join(', ');
      } else if (
        (k === 'artifact_paths' || k === 'exclude_basenames') &&
        Array.isArray(v)
      ) {
        fields[k] = v.join(', ');
      } else if (JSON_FIELDS.has(k) && typeof v !== 'string') {
        // null/undefined = unset; load as empty, not the literal "null".
        fields[k] = v == null ? '' : JSON.stringify(v, null, 2);
      } else {
        fields[k] = v;
      }
    } else {
      advanced[k] = v;
    }
  }

  const advancedJson =
    Object.keys(advanced).length > 0 ? JSON.stringify(advanced, null, 2) : '{}';

  return {
    key,
    type,
    id,
    fields,
    advancedJson,
    showAdvanced: Object.keys(advanced).length > 0,
  };
}

/* ------------------------------------------------------------------ */
/*  UI Components                                                      */
/* ------------------------------------------------------------------ */

export function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <label className="text-xs text-[var(--muted-foreground)]">{label}</label>
      {children}
    </div>
  );
}

function EnvIdPicker({
  value,
  onChange,
  selectedType,
}: {
  value: string;
  onChange: (val: string, type?: string) => void;
  /** Set only by steps that persist the env type; shows the restricted-host selector. */
  selectedType?: string;
}) {
  const [browsing, setBrowsing] = useState(!value);
  // Hosted deployments can restrict environment-catalog browsing, so restricted hosts get
  // id entry instead. The type comes from the catalog otherwise, so steps that
  // keep it have to declare it here.
  const hostRestricted = false;  // no host gating in the standalone hub

  return (
    <div>
      {value && !browsing ? (
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm text-[var(--foreground)]">
            {value}
          </span>
          <button
            type="button"
            onClick={() => setBrowsing(true)}
            className="text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors underline"
          >
            Change
          </button>
        </div>
      ) : hostRestricted ? (
        <div className="flex flex-col gap-2">
          <input
            type="text"
            value={value}
            onChange={e => onChange(e.target.value, selectedType)}
            placeholder="Environment id"
            className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-[var(--ring)]"
          />
          {selectedType !== undefined && (
            <Select.Root
              value={selectedType}
              onValueChange={type => onChange(value, type)}
              size="2"
            >
              <Select.Trigger
                variant="surface"
                placeholder="Environment type"
                className="w-full max-w-xs"
              />
              <Select.Content position="popper" sideOffset={4}>
                {ENV_SECTIONS.map(section => (
                  <Select.Item key={section.type} value={section.type}>
                    {section.label}
                  </Select.Item>
                ))}
              </Select.Content>
            </Select.Root>
          )}
        </div>
      ) : (
        <div className="rounded-lg border border-[var(--border)] overflow-hidden max-h-[420px] overflow-y-auto">
          <EnvironmentsPage
            selectionMode
            selectedId={value}
            onSelect={(id, type) => {
              onChange(id, type);
              setBrowsing(false);
            }}
          />
        </div>
      )}
    </div>
  );
}

function ArtifactIdPicker({
  value,
  onChange,
  universeType,
  selectedType,
}: {
  value: string;
  onChange: (
    val: string,
    type?: 'service_universe' | 'file_artifact_universe',
  ) => void;
  universeType?: 'service_universe' | 'file_artifact_universe';
  /** Already-selected universe type, for the restricted-host type toggle. */
  selectedType?: string;
}) {
  const [browsing, setBrowsing] = useState(!value);
  const typeGroupName = useId();
  // Hosted deployments can restrict universe-catalog browsing, so restricted hosts get
  // id entry instead. The type comes from the catalog otherwise, so it has to
  // be declared here — `destination_path` below keys off it.
  const hostRestricted = false;  // no host gating in the standalone hub

  return (
    <div>
      {value && !browsing ? (
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm text-[var(--foreground)]">
            {value}
          </span>
          <button
            type="button"
            onClick={() => setBrowsing(true)}
            className="text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors underline"
          >
            Change
          </button>
        </div>
      ) : hostRestricted ? (
        <div className="flex flex-col gap-2">
          <input
            type="text"
            value={value}
            onChange={e => onChange(e.target.value, universeType)}
            placeholder="Universe id"
            className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-[var(--ring)]"
          />
          {!universeType && (
            <div className="flex items-center gap-4">
              {(
                [
                  ['service_universe', 'Service'],
                  ['file_artifact_universe', 'File artifact'],
                ] as const
              ).map(([type, label]) => (
                <label
                  key={type}
                  className="flex items-center gap-1.5 cursor-pointer text-xs text-[var(--muted-foreground)]"
                >
                  <input
                    type="radio"
                    name={typeGroupName}
                    checked={selectedType === type}
                    onChange={() => onChange(value, type)}
                    className="border-[var(--border)]"
                  />
                  {label}
                </label>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="rounded-lg border border-[var(--border)] overflow-hidden max-h-[420px] overflow-y-auto">
          <UniversesPage
            selectionMode
            universeType={universeType}
            selectedId={value}
            onSelect={(id, _v, type) => {
              if (type === 'coding_task_harbor') return;
              onChange(id, type);
              setBrowsing(false);
            }}
          />
        </div>
      )}
    </div>
  );
}

function ArtifactVersionSelect({
  artifactId,
  value,
  onChange,
}: {
  artifactId: string;
  value: number | null;
  onChange: (v: number | null) => void;
}) {
  const [versions, setVersions] = useState<number[]>([]);
  useEffect(() => {
    if (!artifactId) return;
    const controller = new AbortController();
    apiFetch(
      `${BACKEND_URL}/api/v1/artifacts/${encodeURIComponent(
        artifactId,
      )}/versions`,
      { signal: controller.signal },
    )
      .then(r => (r.ok ? r.json() : []))
      .then((list: { version: number }[]) =>
        setVersions(Array.isArray(list) ? list.map(v => v.version) : []),
      )
      .catch(() => {});
    return () => controller.abort();
  }, [artifactId]);

  const latestLabel = versions.length
    ? `Latest (v${Math.max(...versions)})`
    : 'Latest';
  return (
    <Select.Root
      value={value == null ? 'latest' : String(value)}
      onValueChange={v => onChange(v === 'latest' ? null : Number(v))}
      size="2"
    >
      <Select.Trigger variant="surface" className="w-full max-w-xs" />
      <Select.Content position="popper" sideOffset={4}>
        <Select.Item value="latest">{latestLabel}</Select.Item>
        {versions.map(v => (
          <Select.Item key={v} value={String(v)}>
            v{v}
          </Select.Item>
        ))}
      </Select.Content>
    </Select.Root>
  );
}

function CriteriaEditor({
  value,
  onChange,
}: {
  value: string;
  onChange: (val: string) => void;
}) {
  const [mode, setMode] = useState<'visual' | 'json'>('visual');
  const [jsonDraft, setJsonDraft] = useState(value);
  const [jsonError, setJsonError] = useState<string | null>(null);

  const criteria: Record<string, unknown>[] = (() => {
    try {
      const parsed = JSON.parse(value || '[]');
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  })();

  const updateCriteria = useCallback(
    (next: Record<string, unknown>[]) => {
      const json = JSON.stringify(next, null, 2);
      onChange(json);
      setJsonDraft(json);
    },
    [onChange],
  );

  const addCriterion = useCallback(() => {
    updateCriteria([...criteria, { id: randomSuffix() + randomSuffix() }]);
  }, [criteria, updateCriteria]);

  const removeCriterion = useCallback(
    (index: number) => {
      updateCriteria(criteria.filter((_, i) => i !== index));
    },
    [criteria, updateCriteria],
  );

  const updateCriterionField = useCallback(
    (index: number, key: string, val: string) => {
      updateCriteria(
        criteria.map((c, i) => (i === index ? { ...c, [key]: val } : c)),
      );
    },
    [criteria, updateCriteria],
  );

  const removeCriterionField = useCallback(
    (index: number, key: string) => {
      updateCriteria(
        criteria.map((c, i) => {
          if (i !== index) return c;
          const next = { ...c };
          delete next[key];
          return next;
        }),
      );
    },
    [criteria, updateCriteria],
  );

  const addCriterionField = useCallback(
    (index: number) => {
      const key = `field_${Object.keys(criteria[index] ?? {}).length}`;
      updateCriterionField(index, key, '');
    },
    [criteria, updateCriterionField],
  );

  const renameCriterionField = useCallback(
    (index: number, oldKey: string, newKey: string) => {
      if (!newKey || newKey === oldKey) return;
      updateCriteria(
        criteria.map((c, i) => {
          if (i !== index) return c;
          const entries = Object.entries(c).map(([k, v]) =>
            k === oldKey ? [newKey, v] : [k, v],
          );
          return Object.fromEntries(entries);
        }),
      );
    },
    [criteria, updateCriteria],
  );

  const switchToJson = useCallback(() => {
    setJsonDraft(JSON.stringify(criteria, null, 2));
    setJsonError(null);
    setMode('json');
  }, [criteria]);

  const switchToVisual = useCallback(() => {
    try {
      const parsed = JSON.parse(jsonDraft);
      if (!Array.isArray(parsed)) {
        setJsonError('Must be a JSON array');
        return;
      }
      onChange(jsonDraft);
      setJsonError(null);
      setMode('visual');
    } catch {
      setJsonError('Invalid JSON');
    }
  }, [jsonDraft, onChange]);

  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <button
          type="button"
          onClick={() =>
            mode === 'visual' ? switchToJson() : switchToVisual()
          }
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-[var(--border)] text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors"
        >
          {mode === 'visual' ? '{ } Switch to JSON' : 'Switch to Visual'}
        </button>
      </div>

      {mode === 'json' ? (
        <div className="flex flex-col gap-1">
          <textarea
            value={jsonDraft}
            onChange={e => {
              setJsonDraft(e.target.value);
              setJsonError(null);
              try {
                const parsed = JSON.parse(e.target.value);
                if (Array.isArray(parsed)) onChange(e.target.value);
              } catch {
                /* defer validation to mode switch */
              }
            }}
            rows={10}
            className={`w-full rounded-md border bg-[var(--background)] px-3 py-2 text-xs font-mono focus:outline-none focus:ring-1 focus:ring-[var(--ring)] resize-y ${
              jsonError ? 'border-red-500' : 'border-[var(--border)]'
            }`}
          />
          {jsonError && (
            <span className="text-xs text-red-500">{jsonError}</span>
          )}
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {criteria.length === 0 && (
            <p className="text-xs text-[var(--muted-foreground)] italic">
              No criteria yet
            </p>
          )}
          {criteria.map((criterion, ci) => (
            <div
              key={ci}
              className="rounded-md border border-[var(--border)] p-3"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
                  Criterion #{ci + 1}
                </span>
                <button
                  type="button"
                  onClick={() => removeCriterion(ci)}
                  className="text-[var(--muted-foreground)] hover:text-red-500 transition-colors"
                >
                  <X size={14} />
                </button>
              </div>
              <div className="flex flex-col gap-1.5">
                {Object.entries(criterion).map(([key, val]) => (
                  <div key={key} className="flex items-start gap-1.5">
                    <input
                      type="text"
                      value={key}
                      onChange={e =>
                        renameCriterionField(ci, key, e.target.value)
                      }
                      className="w-28 flex-shrink-0 rounded border border-[var(--border)] bg-[var(--secondary)] px-1.5 py-0.5 text-[11px] font-mono focus:outline-none focus:ring-1 focus:ring-[var(--ring)]"
                    />
                    {typeof val === 'object' && val !== null ? (
                      <textarea
                        value={JSON.stringify(val, null, 2)}
                        onChange={e => {
                          try {
                            updateCriterionField(
                              ci,
                              key,
                              JSON.parse(e.target.value),
                            );
                          } catch {
                            updateCriterionField(
                              ci,
                              key,
                              e.target.value as unknown as string,
                            );
                          }
                        }}
                        rows={3}
                        className="flex-1 rounded border border-[var(--border)] bg-[var(--background)] px-1.5 py-0.5 text-[11px] font-mono focus:outline-none focus:ring-1 focus:ring-[var(--ring)] resize-y"
                      />
                    ) : (
                      <input
                        type="text"
                        value={String(val ?? '')}
                        onChange={e =>
                          updateCriterionField(ci, key, e.target.value)
                        }
                        className="flex-1 rounded border border-[var(--border)] bg-[var(--background)] px-1.5 py-0.5 text-[11px] font-mono focus:outline-none focus:ring-1 focus:ring-[var(--ring)]"
                      />
                    )}
                    {key !== 'id' && (
                      <button
                        type="button"
                        onClick={() => removeCriterionField(ci, key)}
                        className="p-0.5 text-[var(--muted-foreground)] hover:text-red-500 transition-colors flex-shrink-0"
                      >
                        <X size={12} />
                      </button>
                    )}
                  </div>
                ))}
                <button
                  type="button"
                  onClick={() => addCriterionField(ci)}
                  className="text-[11px] text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors self-start"
                >
                  + Add field
                </button>
              </div>
            </div>
          ))}
          <button
            type="button"
            onClick={addCriterion}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-[var(--border)] text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors self-start"
          >
            <Plus size={12} />
            Add Criterion
          </button>
        </div>
      )}
    </div>
  );
}

export function StepFields({
  step,
  index,
  allSteps,
  onUpdateField,
  onUpdateStep,
}: {
  step: StepState;
  index: number;
  allSteps: StepState[];
  onUpdateField: (index: number, key: string, value: unknown) => void;
  onUpdateStep: (index: number, updates: Partial<StepState>) => void;
}) {
  // Auto-select first available option for fields that depend on other steps
  useEffect(() => {
    const deployedAgentNames = allSteps
      .filter(s => s.type === 'deploy_agent' && s.fields.agent_name)
      .map(s => String(s.fields.agent_name));
    const uniqueAgents = [...new Set(deployedAgentNames)];

    const deployedEnvIds = allSteps
      .filter(s => s.type === 'deploy_env' && s.fields.env_id)
      .map(s => String(s.fields.env_id));

    const promptStepIds = allSteps
      .filter(s => s.type === 'prompt_agent' && s.id)
      .map(s => s.id);

    // Agent name auto-select for prompt_agent, collect_artifacts, rubrics_verifier, cua_initialize, cua_evaluate
    if (
      [
        'prompt_agent',
        'collect_artifacts',
        'rubrics_verifier',
        'cua_initialize',
        'cua_evaluate',
      ].includes(step.type) &&
      !step.fields.agent_name &&
      uniqueAgents.length > 0
    ) {
      onUpdateField(index, 'agent_name', uniqueAgents[0]!);
    }

    // deploy_agent: auto-add all deployed env_ids
    if (step.type === 'deploy_agent') {
      const currentIds = String(step.fields.env_ids ?? '')
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
      const selectedIds = new Set(currentIds);
      const missing = deployedEnvIds.filter(id => !selectedIds.has(id));
      if (missing.length > 0) {
        missing.forEach(id => selectedIds.add(id));
        onUpdateField(index, 'env_ids', Array.from(selectedIds).join(', '));
      }
    }

    // load_artifact, cua_initialize, cua_evaluate: auto-select first env_id
    if (
      ['load_artifact', 'cua_initialize', 'cua_evaluate'].includes(step.type) &&
      !step.fields.env_id &&
      deployedEnvIds.length > 0
    ) {
      onUpdateField(index, 'env_id', deployedEnvIds[0]);
    }

    // rubrics_verifier: auto-select first prompt_id
    if (
      step.type === 'rubrics_verifier' &&
      !step.fields.prompt_id &&
      promptStepIds.length > 0
    ) {
      onUpdateField(index, 'prompt_id', promptStepIds[0]);
    }

    // run_code: default to host when there's no deploy_agent step (matches the agent radio).
    if (
      step.type === 'run_code' &&
      !step.fields.env_id &&
      !allSteps.some(s => s.type === 'deploy_agent') &&
      deployedEnvIds.length > 0
    ) {
      onUpdateField(index, 'env_id', deployedEnvIds[0]);
    }

    // publish: auto-resolve the collect source — synced to the sole collect step, or the first
    // of several (the user only sees a picker when there's an actual choice).
    if (step.type === 'publish') {
      const collectIds = allSteps
        .filter(s => s.type === 'collect_artifacts')
        .map(s => s.id);
      const current = step.fields.collected_artifacts_step_id;
      if (collectIds.length === 1 && current !== collectIds[0]) {
        onUpdateField(index, 'collected_artifacts_step_id', collectIds[0]);
      } else if (collectIds.length > 1 && !current) {
        onUpdateField(index, 'collected_artifacts_step_id', collectIds[0]);
      }
    }
  }, [step.type, step.fields, index, allSteps, onUpdateField]);

  const text = (
    key: string,
    label: string,
    opts?: { placeholder?: string; mono?: boolean },
  ) => (
    <Field label={label} key={key}>
      <input
        type="text"
        value={String(step.fields[key] ?? '')}
        onChange={e => onUpdateField(index, key, e.target.value)}
        placeholder={opts?.placeholder}
        className={`w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-[var(--ring)] ${
          opts?.mono ? 'font-mono' : ''
        }`}
      />
    </Field>
  );

  // A placeholder shows the default as a hint while the field stays empty (unset),
  // so hide the number spinner — its arrows would increment from 0, not the hint.
  const number = (
    key: string,
    label: string,
    opts?: { placeholder?: string; step?: number },
  ) => (
    <Field label={label} key={key}>
      <input
        type="number"
        value={String(step.fields[key] ?? '')}
        placeholder={opts?.placeholder}
        step={opts?.step}
        onChange={e =>
          onUpdateField(
            index,
            key,
            e.target.value === '' ? '' : Number(e.target.value),
          )
        }
        className={`w-32 rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-[var(--ring)] ${
          opts?.placeholder
            ? '[appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none'
            : ''
        }`}
      />
    </Field>
  );

  const textarea = (
    key: string,
    label: string,
    rows = 3,
    opts?: { mono?: boolean },
  ) => (
    <Field label={label} key={key}>
      <textarea
        value={String(step.fields[key] ?? '')}
        onChange={e => onUpdateField(index, key, e.target.value)}
        rows={rows}
        className={`w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-[var(--ring)] resize-y ${
          opts?.mono ? 'font-mono' : ''
        }`}
      />
    </Field>
  );

  const select = (key: string, label: string, options: string[]) => (
    <Field label={label} key={key}>
      <Select.Root
        value={String(step.fields[key] ?? '')}
        onValueChange={val => onUpdateField(index, key, val)}
        size="2"
      >
        <Select.Trigger variant="surface" className="w-full max-w-xs" />
        <Select.Content position="popper" sideOffset={4}>
          {options.map(o => (
            <Select.Item key={o} value={o}>
              {o}
            </Select.Item>
          ))}
        </Select.Content>
      </Select.Root>
    </Field>
  );

  const deployedAgentNames = allSteps
    .filter(s => s.type === 'deploy_agent' && s.fields.agent_name)
    .map(s => String(s.fields.agent_name));
  const uniqueAgentNames = [...new Set(deployedAgentNames)];

  const agentNamePicker = (key: string) => {
    const currentAgent = String(step.fields[key] ?? '');
    const suffix = String(step.fields._suffix ?? '');
    const setAgent = (name: string) => {
      onUpdateField(index, key, name);
      if (step.type === 'prompt_agent' && suffix) {
        onUpdateStep(index, { id: `prompt-${name}-${suffix}` });
      }
    };
    // Auto-selection handled by useEffect above
    return (
      <Field label="Agent Name" key={key}>
        {uniqueAgentNames.length === 0 ? (
          <p className="text-xs text-[var(--muted-foreground)] italic">
            Add a Deploy Agent step first
          </p>
        ) : (
          <div className="flex flex-col gap-1">
            {uniqueAgentNames.map(name => (
              <label
                key={name}
                className="flex items-center gap-2 cursor-pointer py-0.5"
              >
                <input
                  type="radio"
                  name={`${key}-${step.key}`}
                  checked={currentAgent === name}
                  onChange={() => setAgent(name)}
                  className="border-[var(--border)]"
                />
                <span className="text-sm font-mono">{name}</span>
              </label>
            ))}
          </div>
        )}
      </Field>
    );
  };

  switch (step.type) {
    case 'deploy_env':
      return (
        <>
          <Field label="Environment ID">
            <EnvIdPicker
              value={String(step.fields.env_id ?? '')}
              selectedType={String(step.fields._env_type ?? '')}
              onChange={(val, envType) => {
                onUpdateField(index, 'env_id', val);
                if (envType) onUpdateField(index, '_env_type', envType);
                onUpdateStep(index, { id: `deploy-${val || 'env'}` });
              }}
            />
          </Field>
          {number('ttl_seconds', 'TTL (seconds)')}
          {number('disk_size_gb', 'Disk Size (GB)')}
        </>
      );
    case 'deploy_agent': {
      const deployedEnvIds = allSteps
        .filter(s => s.type === 'deploy_env' && s.fields.env_id)
        .map(s => String(s.fields.env_id));
      const currentIds = String(step.fields.env_ids ?? '')
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
      const selectedIds = new Set(currentIds);
      const missing = deployedEnvIds.filter(id => !selectedIds.has(id));
      // Auto-add missing env_ids handled by useEffect above
      if (missing.length > 0) {
        missing.forEach(id => selectedIds.add(id));
      }
      const toggleEnvId = (envId: string) => {
        const next = new Set(selectedIds);
        if (next.has(envId)) next.delete(envId);
        else next.add(envId);
        onUpdateField(index, 'env_ids', Array.from(next).join(', '));
      };
      return (
        <>
          <Field label="Environment IDs">
            {deployedEnvIds.length === 0 ? (
              <p className="text-xs text-[var(--muted-foreground)] italic">
                Add a Deploy Env step first
              </p>
            ) : (
              <div className="flex flex-col gap-1">
                {deployedEnvIds.map(envId => (
                  <label
                    key={envId}
                    className="flex items-center gap-2 cursor-pointer py-0.5"
                  >
                    <input
                      type="checkbox"
                      checked={selectedIds.has(envId)}
                      onChange={() => toggleEnvId(envId)}
                      className="rounded border-[var(--border)]"
                    />
                    <span className="text-sm font-mono">{envId}</span>
                  </label>
                ))}
              </div>
            )}
          </Field>
          {text('agent_name', 'Agent Name')}
          <Field label="Docker">
            <label className="flex items-center gap-2 cursor-pointer py-0.5">
              <input
                type="checkbox"
                checked={Boolean(step.fields.enable_docker)}
                onChange={() =>
                  onUpdateField(
                    index,
                    'enable_docker',
                    !step.fields.enable_docker,
                  )
                }
                className="rounded border-[var(--border)]"
              />
              <span className="text-sm">
                Enable Docker (isolated rootless daemon, VM only)
              </span>
            </label>
          </Field>
        </>
      );
    }
    case 'load_artifact': {
      const deployedEnvIdsForArtifact = allSteps
        .filter(s => s.type === 'deploy_env' && s.fields.env_id)
        .map(s => String(s.fields.env_id));
      const currentEnvId = String(step.fields.env_id ?? '');
      // Auto-selection handled by useEffect above
      return (
        <>
          <Field label="Environment ID">
            {deployedEnvIdsForArtifact.length === 0 ? (
              <p className="text-xs text-[var(--muted-foreground)] italic">
                Add a Deploy Env step first
              </p>
            ) : (
              <div className="flex flex-col gap-1">
                {deployedEnvIdsForArtifact.map(envId => (
                  <label
                    key={envId}
                    className="flex items-center gap-2 cursor-pointer py-0.5"
                  >
                    <input
                      type="radio"
                      name={`load-artifact-env-${step.key}`}
                      checked={currentEnvId === envId}
                      onChange={() => onUpdateField(index, 'env_id', envId)}
                      className="border-[var(--border)]"
                    />
                    <span className="text-sm font-mono">{envId}</span>
                  </label>
                ))}
              </div>
            )}
          </Field>
          <Field label="Artifact ID">
            <ArtifactIdPicker
              value={String(step.fields.artifact_id ?? '')}
              selectedType={String(step.fields._artifact_universe_type ?? '')}
              onChange={(val, type) => {
                onUpdateField(index, 'artifact_id', val);
                // Reset the pin: a version from the previous universe won't exist on the new one.
                onUpdateField(index, 'artifact_version', null);
                if (type) {
                  onUpdateField(index, '_artifact_universe_type', type);
                }
                onUpdateStep(index, {
                  id: `load-artifact-${val || 'artifact'}`,
                });
              }}
            />
          </Field>
          {step.fields.artifact_id ? (
            <Field label="Version">
              <ArtifactVersionSelect
                artifactId={String(step.fields.artifact_id)}
                value={
                  step.fields.artifact_version == null ||
                  step.fields.artifact_version === ''
                    ? null
                    : Number(step.fields.artifact_version)
                }
                onChange={v => onUpdateField(index, 'artifact_version', v)}
              />
              <p className="mt-1 text-xs text-[var(--muted-foreground)]">
                Latest follows edits to this universe; pin a version for a
                fixed, reproducible mount.
              </p>
            </Field>
          ) : null}
          {/* `destination_path` is meaningful only for FileArtifactUniverse;
              ServiceUniverse loads have no filesystem path. Show only when
              the picker has confirmed a file_artifact_universe selection. */}
          {String(step.fields._artifact_universe_type ?? '') ===
            'file_artifact_universe' &&
            text('destination_path', 'Destination Path', {
              mono: true,
              placeholder: 'Optional — defaults to /tmp/file_artifacts',
            })}
        </>
      );
    }
    case 'prompt_agent': {
      const outputFormatErr = getJsonError(
        String(step.fields.output_format ?? ''),
      );
      // Default the section open when any advanced field already has a value (e.g. a
      // loaded task), so configured values aren't hidden. The explicit toggle wins once set.
      const isSet = (v: unknown) => v !== '' && v !== undefined && v !== null;
      const hasAdvancedValue =
        isSet(step.fields.max_turns) ||
        isSet(step.fields.effort) ||
        isSet(step.fields.max_thinking_tokens) ||
        isSet(step.fields.output_format);
      const advOpen =
        step.fields._advancedAgentOpen !== undefined
          ? Boolean(step.fields._advancedAgentOpen)
          : hasAdvancedValue;
      return (
        <>
          {textarea('system_prompt', 'System Prompt', 4)}
          {textarea('prompt', 'Prompt', 4)}
          {select('model', 'Model', MODEL_OPTIONS)}
          {agentNamePicker('agent_name')}
          {number('timeout_seconds', 'Timeout (seconds)')}
          <div className="mt-1">
            <button
              type="button"
              onClick={() =>
                onUpdateField(index, '_advancedAgentOpen', !advOpen)
              }
              className="flex items-center gap-1.5 text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] focus:outline-none"
            >
              {advOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              <span>Advanced agent settings</span>
            </button>
            {advOpen && (
              <div className="mt-2 flex flex-col gap-3 border-l-2 border-[var(--border)] pl-3">
                <div className="text-[10px] font-medium uppercase tracking-wide text-[var(--muted-foreground)]">
                  Agent behavior
                </div>
                {number('max_turns', 'Max Turns', {
                  placeholder: '120',
                  step: 10,
                })}
                <Field label="Effort" key="effort">
                  <Select.Root
                    value={String(step.fields.effort || '__default__')}
                    onValueChange={v =>
                      onUpdateField(
                        index,
                        'effort',
                        v === '__default__' ? '' : v,
                      )
                    }
                    size="2"
                  >
                    <Select.Trigger
                      variant="surface"
                      className="w-full max-w-xs"
                    />
                    <Select.Content position="popper" sideOffset={4}>
                      <Select.Item value="__default__">
                        (model default)
                      </Select.Item>
                      {['low', 'medium', 'high', 'xhigh', 'max'].map(o => (
                        <Select.Item key={o} value={o}>
                          {o}
                        </Select.Item>
                      ))}
                    </Select.Content>
                  </Select.Root>
                </Field>
                {number('max_thinking_tokens', 'Max Thinking Tokens', {
                  placeholder: 'model default',
                  step: 1024,
                })}
                <div className="text-[10px] font-medium uppercase tracking-wide text-[var(--muted-foreground)]">
                  Output contract
                </div>
                {textarea(
                  'output_format',
                  'Structured Output Schema (JSON)',
                  4,
                  {
                    mono: true,
                  },
                )}
                {outputFormatErr && (
                  <span className="text-xs text-red-500">
                    {outputFormatErr}
                  </span>
                )}
              </div>
            )}
          </div>
        </>
      );
    }
    case 'run_code': {
      const deployedEnvIdsForRun = allSteps
        .filter(s => s.type === 'deploy_env' && s.fields.env_id)
        .map(s => String(s.fields.env_id));
      const hasDeployAgent = allSteps.some(s => s.type === 'deploy_agent');
      const currentRunEnvId = String(step.fields.env_id ?? '');
      return (
        <>
          {text('script_artifact_id', 'Script Artifact ID', {
            mono: true,
            placeholder:
              'e.g. run-code-demo-filter (a single-file FileArtifact)',
          })}
          {text('entrypoint', 'Entrypoint', {
            mono: true,
            placeholder: 'run',
          })}
          {textarea('args', 'Args (JSON)', 4)}
          <Field label="Run target">
            <div className="flex flex-col gap-1">
              {hasDeployAgent && (
                <label className="flex items-center gap-2 cursor-pointer py-0.5">
                  <input
                    type="radio"
                    name={`run-code-target-${step.key}`}
                    checked={currentRunEnvId === ''}
                    onChange={() => onUpdateField(index, 'env_id', '')}
                    className="border-[var(--border)]"
                  />
                  <span className="text-sm text-[var(--muted-foreground)]">
                    Agent container
                  </span>
                </label>
              )}
              {deployedEnvIdsForRun.length === 0 && !hasDeployAgent && (
                <p className="text-xs text-[var(--muted-foreground)] italic">
                  Add a Deploy Env (host) or Deploy Agent step first
                </p>
              )}
              {deployedEnvIdsForRun.map(envId => (
                <label
                  key={envId}
                  className="flex items-center gap-2 cursor-pointer py-0.5"
                >
                  <input
                    type="radio"
                    name={`run-code-target-${step.key}`}
                    checked={currentRunEnvId === envId}
                    onChange={() => onUpdateField(index, 'env_id', envId)}
                    className="border-[var(--border)]"
                  />
                  <span className="text-sm font-mono">
                    {envId}{' '}
                    <span className="text-[var(--muted-foreground)]">
                      (VM host — no agent)
                    </span>
                  </span>
                </label>
              ))}
            </div>
          </Field>
          {currentRunEnvId === '' && agentNamePicker('agent_name')}
          {number('timeout_seconds', 'Timeout (seconds)')}
        </>
      );
    }
    case 'rubrics_verifier': {
      const promptStepIds = allSteps
        .filter(s => s.type === 'prompt_agent' && s.id)
        .map(s => s.id);
      const currentPromptId = String(step.fields.prompt_id ?? '');
      // Auto-selection handled by useEffect above
      return (
        <>
          <Field label="Prompt ID">
            {promptStepIds.length === 0 ? (
              <p className="text-xs text-[var(--muted-foreground)] italic">
                Add a Prompt Agent step first
              </p>
            ) : (
              <div className="flex flex-col gap-1">
                {promptStepIds.map(id => (
                  <label
                    key={id}
                    className="flex items-center gap-2 cursor-pointer py-0.5"
                  >
                    <input
                      type="radio"
                      name={`prompt-id-${step.key}`}
                      checked={currentPromptId === id}
                      onChange={() => onUpdateField(index, 'prompt_id', id)}
                      className="border-[var(--border)]"
                    />
                    <span className="text-sm font-mono">{id}</span>
                  </label>
                ))}
              </div>
            )}
          </Field>
          {select('default_model', 'Default Model', MODEL_OPTIONS)}
          {select(
            'score_aggregator',
            'Score Aggregator',
            SCORE_AGGREGATOR_OPTIONS,
          )}
          <Field label="Criteria">
            <CriteriaEditor
              value={String(step.fields.criteria ?? '[]')}
              onChange={val => onUpdateField(index, 'criteria', val)}
            />
          </Field>
        </>
      );
    }
    case 'env_outcome_verifier':
      return (
        <>
          <Field label="Environment ID">
            <EnvIdPicker
              value={String(step.fields.env_id ?? '')}
              onChange={val => onUpdateField(index, 'env_id', val)}
            />
          </Field>
          {text('file_artifact_id', 'File Artifact ID', { mono: true })}
          {select(
            'score_aggregator',
            'Score Aggregator',
            SCORE_AGGREGATOR_OPTIONS,
          )}
        </>
      );
    case 'cua_initialize': {
      const cuaInitEnvIds = allSteps
        .filter(s => s.type === 'deploy_env' && s.fields.env_id)
        .map(s => String(s.fields.env_id));
      const cuaInitCurrent = String(step.fields.env_id ?? '');
      // Auto-selection handled by useEffect above
      return (
        <>
          <Field label="Environment ID">
            {cuaInitEnvIds.length === 0 ? (
              <p className="text-xs text-[var(--muted-foreground)] italic">
                Add a Deploy Env step first
              </p>
            ) : (
              <div className="flex flex-col gap-1">
                {cuaInitEnvIds.map(envId => (
                  <label
                    key={envId}
                    className="flex items-center gap-2 cursor-pointer py-0.5"
                  >
                    <input
                      type="radio"
                      name={`cua-init-env-${step.key}`}
                      checked={cuaInitCurrent === envId}
                      onChange={() => onUpdateField(index, 'env_id', envId)}
                      className="border-[var(--border)]"
                    />
                    <span className="text-sm font-mono">{envId}</span>
                  </label>
                ))}
              </div>
            )}
          </Field>
          {textarea('init_config', 'Init Config (JSON array)', 4)}
        </>
      );
    }
    case 'cua_evaluate': {
      const cuaEvalEnvIds = allSteps
        .filter(s => s.type === 'deploy_env' && s.fields.env_id)
        .map(s => String(s.fields.env_id));
      const cuaEvalCurrent = String(step.fields.env_id ?? '');
      // Auto-selection handled by useEffect above
      return (
        <>
          <Field label="Environment ID">
            {cuaEvalEnvIds.length === 0 ? (
              <p className="text-xs text-[var(--muted-foreground)] italic">
                Add a Deploy Env step first
              </p>
            ) : (
              <div className="flex flex-col gap-1">
                {cuaEvalEnvIds.map(envId => (
                  <label
                    key={envId}
                    className="flex items-center gap-2 cursor-pointer py-0.5"
                  >
                    <input
                      type="radio"
                      name={`cua-eval-env-${step.key}`}
                      checked={cuaEvalCurrent === envId}
                      onChange={() => onUpdateField(index, 'env_id', envId)}
                      className="border-[var(--border)]"
                    />
                    <span className="text-sm font-mono">{envId}</span>
                  </label>
                ))}
              </div>
            )}
          </Field>
          {textarea('evaluator', 'Evaluator (JSON object)', 4)}
        </>
      );
    }
    case 'prompt_usersim':
      return (
        <>
          {text('usersim_image_id', 'UserSim Image ID', { mono: true })}
          {select('usersim_model', 'UserSim Model', MODEL_OPTIONS)}
          {text('solver_image_id', 'Solver Image ID', { mono: true })}
          {text('task_artifact_id', 'Task Artifact ID', { mono: true })}
          {number('timeout_seconds', 'Timeout (seconds)')}
          {number('poll_interval', 'Poll Interval (seconds)')}
        </>
      );
    case 'build_interactive_solver_image':
      return (
        <>
          {text('task_artifact_id', 'Task Artifact ID', { mono: true })}
          {text('output_artifact_id', 'Output Artifact ID', {
            mono: true,
            placeholder: 'Auto-derives from task_artifact_id if blank',
          })}
          {text('language_fallback', 'Language Fallback', {
            placeholder:
              'e.g. python, typescript — used if artifact has no Dockerfile',
          })}
          {text('solver_base_image_id', 'Solver Base Image ID', {
            mono: true,
            placeholder: 'Alternative to language_fallback',
          })}
          {text('build_script_path', 'Build Script Path', {
            mono: true,
            placeholder: 'Override for scripts/build_interactive_solver.sh',
          })}
        </>
      );
    case 'interactive_usersim':
    case 'interactive_usersim_deploy':
      return (
        <>
          {text('task_artifact_id', 'Task Artifact ID', { mono: true })}
          {text('solver_image_id', 'Solver Image ID', {
            mono: true,
            placeholder:
              'Leave blank to chain from prior Build Interactive Solver Image step',
          })}
          {text('userassistant_image_id', 'UserAssistant Image ID', {
            mono: true,
          })}
          {text('proxy_image_id', 'Proxy Image ID', {
            mono: true,
            placeholder:
              'Clear to bypass the standalone proxy sidecar and call the model provider directly.',
          })}
          {select('advisor_model', 'Advisor Model', MODEL_OPTIONS)}
          {text('agent_model', 'Agent Model', { mono: true })}
          {number('timeout_seconds', 'Timeout (seconds)')}
          {text('model_label', 'Model Label')}
          {text('advisor_api_key_secret', 'Advisor API Key Secret', {
            mono: true,
          })}
          {text('advisor_base_url', 'Advisor Base URL', { mono: true })}
        </>
      );
    case 'collect_artifacts': {
      const promptStepIds = allSteps
        .filter(s => s.type === 'prompt_agent')
        .map(s => s.id);
      // Mode is derived from which fields are set (the backend infers it the same way).
      const derivedMode: 'reported' | 'list' | 'everything' = step.fields
        .manifest_step_id
        ? 'reported'
        : step.fields.artifact_paths || step.fields.artifacts_key
        ? 'list'
        : 'everything';
      // `_collectMode` (UI-only; `stepToDict` drops `_`-keys) makes the choice sticky, so clearing
      // both list inputs mid-edit can't flip the mode and unmount its inputs. Else derive.
      const collectMode =
        (step.fields._collectMode as 'reported' | 'list' | 'everything') ??
        derivedMode;
      const setCollectMode = (mode: 'reported' | 'list' | 'everything') => {
        onUpdateField(index, '_collectMode', mode);
        if (mode === 'reported') {
          onUpdateField(index, 'manifest_step_id', promptStepIds[0] ?? '');
          onUpdateField(index, 'artifacts_key', '');
          onUpdateField(index, 'artifact_paths', '');
        } else if (mode === 'list') {
          onUpdateField(index, 'manifest_step_id', '');
          onUpdateField(index, 'exclude_basenames', '');
          if (!step.fields.artifacts_key) {
            onUpdateField(index, 'artifacts_key', 'expected_artifacts');
          }
        } else {
          onUpdateField(index, 'manifest_step_id', '');
          onUpdateField(index, 'exclude_basenames', '');
          onUpdateField(index, 'artifacts_key', '');
          onUpdateField(index, 'artifact_paths', '');
        }
      };
      const modeOption = (
        mode: 'reported' | 'list' | 'everything',
        label: string,
        disabled = false,
      ) => (
        <label
          key={mode}
          className={`flex items-center gap-2 py-0.5 ${
            disabled ? 'opacity-50' : 'cursor-pointer'
          }`}
        >
          <input
            type="radio"
            name={`collect-mode-${step.key}`}
            checked={collectMode === mode}
            disabled={disabled}
            onChange={() => setCollectMode(mode)}
            className="border-[var(--border)]"
          />
          <span className="text-sm">
            {label}
            {disabled ? ' — add a Prompt Agent step first' : ''}
          </span>
        </label>
      );
      return (
        <>
          {agentNamePicker('agent_name')}
          {text('base_path', 'Base Path (on VM)', { mono: true })}
          <Field
            label="Which files should this step collect?"
            key="collect-mode"
          >
            <div className="flex flex-col gap-1">
              {modeOption(
                'reported',
                "Whatever a prior step's agent reported making",
                promptStepIds.length === 0,
              )}
              {modeOption('list', 'A specific list of filenames')}
              {modeOption('everything', 'Everything under the base path')}
            </div>
          </Field>
          {collectMode === 'reported' && (
            <>
              <Field label="Reported by step" key="manifest_step_id">
                <Select.Root
                  value={String(step.fields.manifest_step_id || '')}
                  onValueChange={val =>
                    onUpdateField(index, 'manifest_step_id', val)
                  }
                  size="2"
                >
                  <Select.Trigger
                    variant="surface"
                    className="w-full max-w-xs"
                  />
                  <Select.Content position="popper" sideOffset={4}>
                    {promptStepIds.map(o => (
                      <Select.Item key={o} value={o}>
                        {o}
                      </Select.Item>
                    ))}
                  </Select.Content>
                </Select.Root>
              </Field>
              {text(
                'exclude_basenames',
                'Exclude Files (basenames, comma-separated)',
                { mono: true },
              )}
            </>
          )}
          {collectMode === 'list' && (
            <>
              {text(
                'artifacts_key',
                'Seed Key (per-run filename list, e.g. expected_artifacts)',
              )}
              {text(
                'artifact_paths',
                'Static Filenames (comma-separated, fallback if no seed key)',
                { mono: true },
              )}
            </>
          )}
        </>
      );
    }
    case 'publish': {
      const collectStepIds = allSteps
        .filter(s => s.type === 'collect_artifacts')
        .map(s => s.id);
      return (
        <>
          {text('target', 'Target')}
          {collectStepIds.length === 0 ? (
            <Field label="Artifacts to publish">
              <p className="text-xs text-[var(--muted-foreground)] italic">
                Add a Collect Artifacts step first
              </p>
            </Field>
          ) : collectStepIds.length === 1 ? (
            // Only one possible source — nothing to choose; show it read-only.
            <Field label="Publishing the artifacts collected by">
              <p className="text-sm font-mono text-[var(--muted-foreground)]">
                {collectStepIds[0]}
              </p>
            </Field>
          ) : (
            select(
              'collected_artifacts_step_id',
              'Publish the artifacts collected by',
              collectStepIds,
            )
          )}
          {textarea('options', 'Options (JSON, target-specific)', 4)}
        </>
      );
    }
    case 'human_interaction':
      return <>{number('timeout_seconds', 'Timeout (seconds)')}</>;
  }
}

export function StepEditor({
  index,
  step,
  allSteps,
  isSelected,
  onUpdate,
  onUpdateField,
  onRemove,
  onMove,
  totalSteps,
  locked,
}: {
  index: number;
  step: StepState;
  allSteps: StepState[];
  isSelected?: boolean;
  onUpdate: (index: number, updates: Partial<StepState>) => void;
  onUpdateField: (index: number, key: string, value: unknown) => void;
  onRemove: (index: number) => void;
  onMove: (index: number, direction: -1 | 1) => void;
  totalSteps: number;
  locked?: boolean;
}) {
  const jsonErr = step.showAdvanced ? getJsonError(step.advancedJson) : null;
  const color = STEP_COLORS[step.type] || '#6b7280';

  if (locked) {
    return (
      <div
        className="border rounded-lg p-4 opacity-60"
        style={{ borderColor: 'var(--border)' }}
      >
        <div className="flex items-center gap-3">
          <span className="text-xs font-medium text-[var(--muted-foreground)]">
            {index + 1}.
          </span>
          <span
            className="w-2 h-2 rounded-full flex-shrink-0"
            style={{ backgroundColor: color }}
          />
          <span className="text-sm font-medium text-[var(--foreground)]">
            {STEP_TYPE_LABELS[step.type]}
          </span>
          <span className="text-xs font-mono text-[var(--muted-foreground)]">
            {step.id}
          </span>
          <span className="ml-auto text-xs text-[var(--muted-foreground)] italic">
            Locked
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      className="border rounded-lg p-4 transition-colors"
      style={{
        borderColor: isSelected ? color : 'var(--border)',
        borderWidth: isSelected ? 2 : 1,
      }}
    >
      <div className="flex items-center gap-3 mb-3">
        <span className="flex items-center gap-0.5 text-xs font-medium text-[var(--muted-foreground)] flex-shrink-0 cursor-grab active:cursor-grabbing">
          <GripVertical size={12} className="opacity-40" />
          {index + 1}.
        </span>
        <span className="text-sm font-medium text-[var(--foreground)]">
          {step.type}
        </span>
        {!AUTO_ID_STEP_TYPES.has(step.type) && (
          <input
            type="text"
            value={step.id}
            onChange={e => onUpdate(index, { id: e.target.value })}
            placeholder="step-id"
            className="rounded-md border border-[var(--border)] bg-[var(--background)] px-2 py-1 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-[var(--ring)] w-48"
          />
        )}
        {AUTO_ID_STEP_TYPES.has(step.type) && (
          <span className="text-xs font-mono text-[var(--muted-foreground)] py-1">
            {step.id}
          </span>
        )}
        <div className="ml-auto flex items-center gap-1">
          <button
            onClick={() => onMove(index, -1)}
            disabled={index === 0}
            className="p-0.5 text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors disabled:opacity-25 disabled:pointer-events-none"
            title="Move up"
          >
            <ArrowUp size={14} />
          </button>
          <button
            onClick={() => onMove(index, 1)}
            disabled={index === totalSteps - 1}
            className="p-0.5 text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors disabled:opacity-25 disabled:pointer-events-none"
            title="Move down"
          >
            <ArrowDown size={14} />
          </button>
          <button
            onClick={() => onRemove(index)}
            className="p-0.5 text-[var(--muted-foreground)] hover:text-red-500 transition-colors"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      <div className="ml-9 flex flex-col gap-2">
        <StepFields
          step={step}
          index={index}
          allSteps={allSteps}
          onUpdateField={onUpdateField}
          onUpdateStep={onUpdate}
        />

        <button
          onClick={() => onUpdate(index, { showAdvanced: !step.showAdvanced })}
          className="flex items-center gap-1 text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] transition-colors mt-1"
        >
          {step.showAdvanced ? (
            <ChevronUp size={12} />
          ) : (
            <ChevronDown size={12} />
          )}
          Advanced JSON
        </button>
        {step.showAdvanced && (
          <div className="flex flex-col gap-1">
            <textarea
              value={step.advancedJson}
              onChange={e => onUpdate(index, { advancedJson: e.target.value })}
              placeholder='{"extra_field": "value"}'
              rows={4}
              className={`w-full rounded-md border bg-[var(--background)] px-3 py-2 text-xs font-mono focus:outline-none focus:ring-1 focus:ring-[var(--ring)] resize-y ${
                jsonErr ? 'border-red-500' : 'border-[var(--border)]'
              }`}
            />
            {jsonErr && <span className="text-xs text-red-500">{jsonErr}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
