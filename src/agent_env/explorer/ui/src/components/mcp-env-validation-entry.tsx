import { useState } from 'react';
import {
  CheckCircle,
  ChevronRight,
  Search,
  Wrench,
  XCircle,
} from 'lucide-react';

interface SchemaToolEntry {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

interface CorrectnessResultEntry {
  tool_name: string;
  passed: boolean;
  error: string;
  justification: string;
}

export function MCPEnvValidationEntry({
  validationKey,
  data,
}: {
  validationKey: string;
  data: Record<string, unknown>;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedTool, setExpandedTool] = useState<string | null>(null);

  const passed = 'passed' in data ? data.passed : undefined;
  const tools = (data.tools ?? []) as SchemaToolEntry[];
  const correctnessResults = (data.results ?? []) as CorrectnessResultEntry[];
  const totalTools =
    (data.total_tools as number) ?? (tools.length || correctnessResults.length);
  const missingDesc = (data.tools_missing_description as string[]) ?? [];
  const parameterIssues =
    (data.parameter_issues as Record<string, string[]>) ?? {};
  const hasToolBrowser = tools.length > 0 || correctnessResults.length > 0;

  return (
    <div className="rounded-lg border border-[var(--border)] overflow-hidden">
      {/* Header row */}
      <div className="flex items-center gap-2 px-4 py-2.5 bg-[var(--secondary)]">
        {passed === true && (
          <CheckCircle size={14} className="text-emerald-500 flex-shrink-0" />
        )}
        {passed === false && (
          <XCircle size={14} className="text-red-500 flex-shrink-0" />
        )}
        <span className="text-sm font-medium text-[var(--foreground)]">
          {validationKey}
        </span>
        <span className="text-xs text-[var(--muted-foreground)]">
          {totalTools} tool{totalTools !== 1 ? 's' : ''}
        </span>
        {hasToolBrowser && (
          <button
            onClick={() => {
              setIsOpen(v => !v);
              setSearchQuery('');
              setExpandedTool(null);
            }}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-[var(--border)] text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--accent)] transition-colors"
          >
            <Wrench size={12} />
            Agent tools
          </button>
        )}
        {(missingDesc.length > 0 ||
          Object.keys(parameterIssues).length > 0) && (
          <span className="text-xs text-amber-600">
            {[
              missingDesc.length > 0 &&
                `${missingDesc.length} tool${
                  missingDesc.length !== 1 ? 's' : ''
                } missing description`,
              Object.keys(parameterIssues).length > 0 &&
                `${
                  Object.values(parameterIssues).flat().length
                } parameter issue${
                  Object.values(parameterIssues).flat().length !== 1 ? 's' : ''
                }`,
            ]
              .filter(Boolean)
              .join(', ')}
          </span>
        )}
        {correctnessResults.length > 0 &&
          correctnessResults.some(r => !r.passed) && (
            <span className="text-xs text-red-500">
              {correctnessResults.filter(r => !r.passed).length} tool
              {correctnessResults.filter(r => !r.passed).length !== 1
                ? 's'
                : ''}{' '}
              failed
            </span>
          )}
      </div>

      {/* Schema tool browser */}
      {isOpen && tools.length > 0 && (
        <div className="border-t border-[var(--border)]">
          <div className="px-4 py-2.5">
            <div className="relative">
              <Search
                size={14}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted-foreground)]"
              />
              <input
                type="text"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                placeholder="Search tools by name or description"
                className="w-full rounded-lg border border-[var(--border)] bg-[var(--background)] text-sm pl-9 pr-3 py-2 focus:border-purple-400 focus:outline-none focus:ring-1 focus:ring-purple-200 transition-colors"
              />
            </div>
          </div>
          <div className="max-h-[400px] overflow-y-auto border-t border-[var(--border)]">
            {tools
              .filter(
                t =>
                  !searchQuery.trim() ||
                  t.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                  (t.description ?? '')
                    .toLowerCase()
                    .includes(searchQuery.toLowerCase()),
              )
              .map(tool => {
                const isExpanded = expandedTool === tool.name;
                const hasIssue = missingDesc.includes(tool.name);
                const toolParamIssues = parameterIssues[tool.name] ?? [];
                const hasAnyIssue = hasIssue || toolParamIssues.length > 0;
                const schema = tool.input_schema ?? {};
                const params = (schema.properties ?? {}) as Record<
                  string,
                  {
                    type?: string;
                    description?: string;
                    default?: unknown;
                    anyOf?: Array<{ type?: string }>;
                  }
                >;
                const requiredFields = ((schema as Record<string, unknown>)
                  .required ?? []) as string[];
                const paramEntries = Object.entries(params);
                const firstLine =
                  (tool.description ?? '')
                    .split('\n')
                    .map(l => l.trim())
                    .find(l => l.length > 0) ?? '';

                return (
                  <div
                    key={tool.name}
                    className={`border-b border-[var(--border)] last:border-b-0 ${
                      isExpanded ? 'bg-[var(--accent)]/50' : ''
                    }`}
                  >
                    <button
                      onClick={() =>
                        setExpandedTool(isExpanded ? null : tool.name)
                      }
                      className="w-full text-left px-5 py-3.5 flex items-start gap-2 hover:bg-[var(--accent)]/50 transition-colors"
                    >
                      <ChevronRight
                        size={14}
                        className={`flex-shrink-0 mt-0.5 text-[var(--muted-foreground)] transition-transform ${
                          isExpanded ? 'rotate-90' : ''
                        }`}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 text-[13px] font-medium">
                          {tool.name}
                          {hasIssue && (
                            <span className="text-[9px] font-semibold uppercase px-1.5 py-0.5 rounded bg-amber-100 text-amber-700">
                              missing description
                            </span>
                          )}
                          {!hasIssue && toolParamIssues.length > 0 && (
                            <span className="text-[9px] font-semibold uppercase px-1.5 py-0.5 rounded bg-amber-100 text-amber-700">
                              {toolParamIssues.length} parameter issue
                              {toolParamIssues.length !== 1 ? 's' : ''}
                            </span>
                          )}
                          {!hasAnyIssue && (
                            <CheckCircle
                              size={12}
                              className="text-emerald-500"
                            />
                          )}
                        </div>
                        <div
                          className={`text-xs mt-1 leading-relaxed ${
                            hasIssue
                              ? 'text-amber-500 italic'
                              : 'text-[var(--muted-foreground)]'
                          } ${isExpanded ? '' : 'line-clamp-1'}`}
                        >
                          {firstLine || (
                            <span className="italic text-amber-500">
                              No description
                            </span>
                          )}
                        </div>
                      </div>
                    </button>
                    {isExpanded && paramEntries.length > 0 && (
                      <div className="px-5 pb-3.5 pl-11">
                        <div className="rounded-lg border border-[var(--border)] bg-[var(--background)] overflow-hidden">
                          <div className="px-3 py-2 bg-[var(--secondary)] border-b border-[var(--border)]">
                            <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)]">
                              Parameters
                            </span>
                          </div>
                          {paramEntries.map(([name, param]) => {
                            const isRequired = requiredFields.includes(name);
                            const typeLabel =
                              param.type ??
                              param.anyOf
                                ?.map(t => t.type)
                                .filter(Boolean)
                                .join(' | ') ??
                              'any';
                            const paramHasIssue = toolParamIssues.some(issue =>
                              issue.includes(`'${name}'`),
                            );
                            return (
                              <div
                                key={name}
                                className={`px-3 py-2 border-b border-[var(--border)] last:border-b-0 flex items-baseline gap-2 ${
                                  paramHasIssue ? 'bg-amber-50' : ''
                                }`}
                              >
                                <code
                                  className={`text-xs font-medium flex-shrink-0 ${
                                    paramHasIssue
                                      ? 'text-amber-700'
                                      : 'text-purple-700'
                                  }`}
                                >
                                  {name}
                                </code>
                                <span className="text-[10px] text-[var(--muted-foreground)] font-mono flex-shrink-0">
                                  {typeLabel}
                                </span>
                                {isRequired && (
                                  <span className="text-[9px] font-semibold uppercase text-red-400 flex-shrink-0">
                                    required
                                  </span>
                                )}
                                {param.default !== undefined &&
                                  param.default !== null && (
                                    <span className="text-[10px] text-[var(--muted-foreground)] flex-shrink-0">
                                      = {JSON.stringify(param.default)}
                                    </span>
                                  )}
                                {paramHasIssue && (
                                  <span className="text-[9px] font-semibold uppercase text-amber-600 flex-shrink-0">
                                    missing description
                                  </span>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                    {isExpanded && paramEntries.length === 0 && (
                      <div className="px-5 pb-3.5 pl-11">
                        <span className="text-xs text-[var(--muted-foreground)] italic">
                          No parameters
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* Correctness results browser */}
      {isOpen && correctnessResults.length > 0 && (
        <div className="border-t border-[var(--border)]">
          <div className="px-4 py-2.5">
            <div className="relative">
              <Search
                size={14}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted-foreground)]"
              />
              <input
                type="text"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                placeholder="Search tools by name"
                className="w-full rounded-lg border border-[var(--border)] bg-[var(--background)] text-sm pl-9 pr-3 py-2 focus:border-purple-400 focus:outline-none focus:ring-1 focus:ring-purple-200 transition-colors"
              />
            </div>
          </div>
          <div className="max-h-[400px] overflow-y-auto border-t border-[var(--border)]">
            {correctnessResults
              .filter(
                r =>
                  !searchQuery.trim() ||
                  r.tool_name.toLowerCase().includes(searchQuery.toLowerCase()),
              )
              .map(r => {
                const isExpanded = expandedTool === r.tool_name;
                return (
                  <div
                    key={r.tool_name}
                    className={`border-b border-[var(--border)] last:border-b-0 ${
                      isExpanded ? 'bg-[var(--accent)]/50' : ''
                    }`}
                  >
                    <button
                      onClick={() =>
                        setExpandedTool(isExpanded ? null : r.tool_name)
                      }
                      className="w-full text-left px-5 py-3.5 flex items-start gap-2 hover:bg-[var(--accent)]/50 transition-colors"
                    >
                      <ChevronRight
                        size={14}
                        className={`flex-shrink-0 mt-0.5 text-[var(--muted-foreground)] transition-transform ${
                          isExpanded ? 'rotate-90' : ''
                        }`}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 text-[13px] font-medium">
                          {r.tool_name.replace(/^mcp__.*?__/, '')}
                          {r.passed ? (
                            <CheckCircle
                              size={12}
                              className="text-emerald-500"
                            />
                          ) : (
                            <XCircle size={12} className="text-red-500" />
                          )}
                        </div>
                        {!isExpanded && r.error && (
                          <div className="text-xs text-red-500 mt-1 line-clamp-1">
                            {r.error}
                          </div>
                        )}
                        {!isExpanded && !r.error && r.justification && (
                          <div className="text-xs text-[var(--muted-foreground)] mt-1 line-clamp-1">
                            {r.justification}
                          </div>
                        )}
                      </div>
                    </button>
                    {isExpanded && (
                      <div className="px-5 pb-3.5 pl-11 space-y-2">
                        {r.error && (
                          <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-2.5">
                            <div className="text-[10px] font-semibold uppercase tracking-wider text-red-500 mb-1">
                              Error
                            </div>
                            <div className="text-xs text-red-600">
                              {r.error}
                            </div>
                          </div>
                        )}
                        {r.justification && (
                          <div className="rounded-lg border border-[var(--border)] bg-[var(--background)] p-2.5">
                            <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--muted-foreground)] mb-1">
                              Justification
                            </div>
                            <div className="text-xs text-[var(--foreground)] leading-relaxed">
                              {r.justification}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}
