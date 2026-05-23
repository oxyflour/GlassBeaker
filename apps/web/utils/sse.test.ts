import assert from "node:assert/strict";
import test from "node:test";

type SseModule = typeof import("./sse");

test("streamJsonSse yields named JSON events from a fetch stream", async () => {
  const { streamJsonSse } = await import("./sse") as SseModule;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response(streamFrom([
    'event: started\ndata: {"task":"init"}\n\n',
    'event: done\ndata: {"ok":true}\n\n',
  ]), {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  })) as typeof fetch;

  try {
    const events = [];
    for await (const event of streamJsonSse("/python/zapdos/sess-1/tasks/init")) {
      events.push(event);
    }

    assert.deepEqual(events, [
      { event: "started", data: { task: "init" } },
      { event: "done", data: { ok: true } },
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("streamJsonSse accepts CRLF event delimiters", async () => {
  const { streamJsonSse } = await import("./sse") as SseModule;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response(streamFrom([
    'event: done\r\ndata: {"ok":true}\r\n\r\n',
  ]), {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  })) as typeof fetch;

  try {
    const events = [];
    for await (const event of streamJsonSse("/python/zapdos/sess-1/tasks/init")) {
      events.push(event);
    }

    assert.deepEqual(events, [
      { event: "done", data: { ok: true } },
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("streamJsonSse throws route text for non-2xx responses", async () => {
  const { streamJsonSse } = await import("./sse") as SseModule;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => new Response("Session expired", { status: 409 })) as typeof fetch;

  try {
    const events = streamJsonSse("/python/zapdos/sess-1/tasks/init");
    await assert.rejects(async () => {
      for await (const _event of events) {
        // consume
      }
    }, /Session expired/);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("streamJsonSse cancels the response body when iteration stops early", async () => {
  const { streamJsonSse } = await import("./sse") as SseModule;
  const encoder = new TextEncoder();
  const originalFetch = globalThis.fetch;
  let cancelled = false;
  globalThis.fetch = (async () => new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode('event: started\ndata: {"task":"init"}\n\n'));
    },
    cancel() {
      cancelled = true;
    },
  }), { status: 200 })) as typeof fetch;

  try {
    for await (const _event of streamJsonSse("/python/zapdos/sess-1/tasks/init")) {
      break;
    }

    assert.equal(cancelled, true);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("streamJsonSse passes the abort signal to fetch", async () => {
  const { streamJsonSse } = await import("./sse") as SseModule;
  const controller = new AbortController();
  const originalFetch = globalThis.fetch;
  let observed: AbortSignal | null = null;
  globalThis.fetch = (async (_input: unknown, init?: RequestInit) => {
    observed = init?.signal ?? null;
    return new Response(streamFrom(['event: done\ndata: {"ok":true}\n\n']), { status: 200 });
  }) as typeof fetch;

  try {
    for await (const _event of streamJsonSse("/python/zapdos/sess-1/tasks/init", {
      signal: controller.signal,
    })) {
      // consume
    }
    assert.equal(observed, controller.signal);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

function streamFrom(chunks: string[]) {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}
