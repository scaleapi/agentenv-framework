const SSE_BOUNDARY = /\r?\n\r?\n/;

export async function readSSEStream<T = unknown>(
  body: ReadableStream<Uint8Array>,
  onEvent: (eventName: string, data: T) => void,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let lastEnd = 0;
      let searchFrom = 0;

      while (true) {
        const sub = buffer.slice(searchFrom);
        const match = SSE_BOUNDARY.exec(sub);
        if (!match) break;
        const endIdx = searchFrom + match.index + match[0].length;
        const part = buffer.slice(lastEnd, endIdx).trim();
        if (part && !part.startsWith(':')) {
          parseSSEBlock(part, onEvent);
        }
        lastEnd = endIdx;
        searchFrom = endIdx;
      }

      buffer = buffer.slice(lastEnd);
    }
  } finally {
    reader.releaseLock();
  }
}

function parseSSEBlock<T>(
  block: string,
  onEvent: (eventName: string, data: T) => void,
): void {
  let eventName = '';
  const dataLines: string[] = [];
  for (const line of block.split('\n')) {
    const trimmed = line.replace(/\r$/, '');
    if (trimmed.startsWith('event: ')) {
      eventName = trimmed.slice(7).trim();
    } else if (trimmed.startsWith('data: ')) {
      dataLines.push(trimmed.slice(6));
    }
  }
  if (eventName && dataLines.length > 0) {
    try {
      const parsed = JSON.parse(dataLines.join('\n'));
      onEvent(eventName, parsed);
    } catch {
      // ignore malformed JSON
    }
  }
}
