/** Parse and validate seed CSVs for the Start Runs panel. RFC 4180-style streaming parser: embedded quotes,
 *  commas/newlines in quoted cells, CRLF/LF/CR, UTF-8 BOM strip. Only structural requirement is an `id` column;
 *  everything else is opaque key/value pairs. */

export const MAX_CSV_SIZE = 10 * 1024 * 1024; // 10 MB
// Keep in sync with the backend MAX_PARALLEL_RUNS.
export const MAX_PARALLEL_RUNS = 64;

export interface ParsedSeeds {
  columns: string[];
  seeds: Record<string, string>[];
  /** Per-seed provenance (same length as `seeds`): the 1-based source line, or a unique non-positive sentinel for grid-added rows. */
  sourceLines: number[];
}

function parseDelimited(text: string, delimiter: string): ParsedSeeds {
  const allRows = parseAllRows(text, delimiter);
  if (allRows.length === 0) throw new Error('No data rows');

  const [headerCells] = allRows[0]!;
  const columns = headerCells.map(col => col.replace(/^﻿/, '').trim());

  const seeds: Record<string, string>[] = [];
  const sourceLines: number[] = [];
  for (let i = 1; i < allRows.length; i++) {
    const [cells, lineNum] = allRows[i]!;
    if (cells.every(c => !c.trim())) continue;
    const row: Record<string, string> = {};
    columns.forEach((col, colIdx) => {
      row[col] = cells[colIdx] ?? '';
    });
    seeds.push(row);
    sourceLines.push(lineNum);

    // Fail fast on pathological input without scanning the whole thing.
    if (seeds.length > MAX_PARALLEL_RUNS) {
      throw new Error(
        `More than ${MAX_PARALLEL_RUNS} data rows; the maximum is ${MAX_PARALLEL_RUNS}`,
      );
    }
  }
  if (seeds.length === 0) throw new Error('No data rows');
  return { columns, seeds, sourceLines };
}

export function parseCsvText(text: string): ParsedSeeds {
  return parseDelimited(text, ',');
}

function detectDelimiter(text: string): string {
  const firstBreak = text.search(/\r|\n/);
  const header = text.slice(0, firstBreak === -1 ? text.length : firstBreak);
  return header.includes('\t') ? '\t' : ',';
}

// Delimiter auto-detected (spreadsheets copy cells as TSV); an `id` column is
// synthesized when absent so a bare pasted column of values works as-is.
export function parseSeedsText(text: string): ParsedSeeds {
  const parsed = parseDelimited(text, detectDelimiter(text));
  if (parsed.columns.includes('id')) return parsed;
  return {
    columns: ['id', ...parsed.columns],
    seeds: parsed.seeds.map((row, i) => ({ id: `row-${i + 1}`, ...row })),
    sourceLines: parsed.sourceLines,
  };
}

const rowIsEmpty = (row: Record<string, string>, columns: string[]): boolean =>
  columns.every(c => c === 'id' || !(row[c] ?? '').trim());

/** Non-empty rows in a grid (blank starter/spare rows don't count as runs). */
export function countRealSeeds(parsed: ParsedSeeds | null): number {
  if (!parsed) return 0;
  return parsed.seeds.filter(r => !rowIsEmpty(r, parsed.columns)).length;
}

/** Append `added` onto the grid (paste/upload/add-row accumulate; Clear resets). Drops blank starter rows,
 *  unions columns, re-ids appended rows, caps at MAX_PARALLEL_RUNS. */
export function appendSeeds(prev: ParsedSeeds | null, added: ParsedSeeds): ParsedSeeds {
  const columns = [
    'id',
    ...new Set([
      ...(prev?.columns ?? []).filter(c => c !== 'id'),
      ...added.columns.filter(c => c !== 'id'),
    ]),
  ];
  const seeds: Record<string, string>[] = [];
  const sourceLines: number[] = [];
  const usedIds = new Set<string>();

  const nextId = (): string => {
    let n = usedIds.size + 1;
    while (usedIds.has(`row-${n}`)) n++;
    return `row-${n}`;
  };
  const push = (row: Record<string, string>, line: number, id: string) => {
    seeds.push(Object.fromEntries(columns.map(c => [c, c === 'id' ? id : row[c] ?? ''])));
    sourceLines.push(line);
    usedIds.add(id);
  };

  prev?.seeds.forEach((row, i) => {
    if (rowIsEmpty(row, columns)) return;
    push(row, prev.sourceLines[i] ?? -(i + 1), (row.id ?? '').trim() || nextId());
  });

  let line = Math.min(0, ...sourceLines) - 1;
  for (const row of added.seeds) {
    if (seeds.length >= MAX_PARALLEL_RUNS) break;
    const raw = (row.id ?? '').trim();
    push(row, line, raw && !usedIds.has(raw) ? raw : nextId());
    line -= 1;
  }
  return { columns, seeds, sourceLines };
}

export function validateSeeds(parsed: ParsedSeeds): void {
  const { columns, seeds, sourceLines } = parsed;
  if (!columns.includes('id')) {
    throw new Error("CSV must contain an 'id' column to identify each seed");
  }
  const seen = new Set<string>();
  for (let i = 0; i < seeds.length; i++) {
    const id = (seeds[i]!.id ?? '').trim();
    // Synthetic rows (added in the grid) carry a non-positive sourceLine — no
    // real source line, so refer to them by position instead.
    const line = sourceLines[i] ?? i + 1;
    const where = line > 0 ? `line ${line}` : `row ${i + 1}`;
    if (!id) throw new Error(`Seed at ${where} has an empty 'id' value`);
    if (seen.has(id)) throw new Error(`Duplicate seed id '${id}' at ${where}`);
    seen.add(id);
  }
}

/** RFC 4180 streaming parser. Returns [cells, startLine] per row (1-based source line) so errors reference the user-visible line, not the post-collapse index. */
function parseAllRows(
  text: string,
  delimiter: string,
): Array<[string[], number]> {
  const rows: Array<[string[], number]> = [];
  let cells: string[] = [];
  let cell = '';
  let inQuotes = false;
  let line = 1;
  let rowStartLine = 1;
  let cellHasContent = false;

  const pushCell = () => {
    cells.push(cell);
    cell = '';
  };
  const pushRow = () => {
    pushCell();
    rows.push([cells, rowStartLine]);
    cells = [];
    cellHasContent = false;
    rowStartLine = line;
  };

  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    const next = text[i + 1];

    if (inQuotes) {
      if (ch === '"' && next === '"') {
        cell += '"';
        i++; // skip the doubled quote
      } else if (ch === '"') {
        inQuotes = false;
      } else {
        cell += ch;
        if (ch === '\n') line++;
        else if (ch === '\r' && next !== '\n') line++;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
      cellHasContent = true;
      continue;
    }
    if (ch === delimiter) {
      pushCell();
      continue;
    }
    if (ch === '\r' && next === '\n') {
      pushRow();
      line++;
      i++;
      continue;
    }
    if (ch === '\n' || ch === '\r') {
      pushRow();
      line++;
      continue;
    }
    cell += ch;
    if (ch.trim() !== '') cellHasContent = true;
  }

  if (inQuotes) {
    throw new Error(
      `Unclosed quote in CSV starting on line ${rowStartLine} — likely a missing closing " on a quoted field`,
    );
  }

  if (cell.length > 0 || cells.length > 0 || cellHasContent) {
    pushRow();
  }

  // Drop entirely-empty rows (consecutive/trailing newlines) — the header-driven column lookup assumes every retained row has cells.
  return rows.filter(([row]) => row.some(c => c !== ''));
}
