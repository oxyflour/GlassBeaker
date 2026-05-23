export type JsonSseEvent<T = unknown> = {
  event: string;
  data: T;
};

export async function* streamJsonSse<T = unknown>(
  input: RequestInfo | URL,
  init: RequestInit = {},
): AsyncGenerator<JsonSseEvent<T>> {
  const response = await fetch(input, init);
  if (!response.ok) {
    throw new Error(await response.text() || `SSE request failed: ${response.status}`);
  }
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("SSE response body is empty");
  }
  const decoder = new TextDecoder();
  let buffer = "";
  let completed = false;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        completed = true;
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      yield* drainSseBuffer<T>(() => buffer, next => {
        buffer = next;
      });
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      const event = parseSseBlock<T>(buffer);
      if (event) yield event;
    }
  } finally {
    if (!completed) {
      await reader.cancel().catch(() => {});
    }
    reader.releaseLock();
  }
}

function* drainSseBuffer<T>(
  read: () => string,
  write: (value: string) => void,
): Generator<JsonSseEvent<T>> {
  let buffer = read();
  let delimiter = findEventDelimiter(buffer);
  while (delimiter) {
    const block = buffer.slice(0, delimiter.index);
    buffer = buffer.slice(delimiter.index + delimiter.length);
    const event = parseSseBlock<T>(block);
    if (event) yield event;
    delimiter = findEventDelimiter(buffer);
  }
  write(buffer);
}

function findEventDelimiter(buffer: string): { index: number; length: number } | null {
  const lf = buffer.indexOf("\n\n");
  const crlf = buffer.indexOf("\r\n\r\n");
  if (lf < 0) return crlf < 0 ? null : { index: crlf, length: 4 };
  if (crlf < 0 || lf < crlf) return { index: lf, length: 2 };
  return { index: crlf, length: 4 };
}

function parseSseBlock<T>(block: string): JsonSseEvent<T> | null {
  let event = "message";
  const data: string[] = [];
  for (const rawLine of block.split("\n")) {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice(6).trimStart();
    } else if (line.startsWith("data:")) {
      data.push(line.slice(5).trimStart());
    }
  }
  if (data.length === 0) return null;
  return { event, data: JSON.parse(data.join("\n")) as T };
}
