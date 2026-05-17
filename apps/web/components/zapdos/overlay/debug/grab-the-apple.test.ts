import assert from "node:assert/strict";
import test from "node:test";

type GrabTheAppleModule = typeof import("./grab-the-apple");

test("createGrabTheAppleRequest posts the canned apple pick payload", async () => {
  const { createGrabTheAppleRequest } = await loadModule<GrabTheAppleModule>("./grab-the-apple.ts");

  assert.deepEqual(createGrabTheAppleRequest(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  });
});

test("grabTheApple posts to the zapdos grab_apple route", async () => {
  const { createGrabTheAppleRequest, grabTheApple } = await loadModule<GrabTheAppleModule>("./grab-the-apple.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      ok: true,
      op_id: "op-1",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;
  const stream = new FakeEventSource("/python/zapdos/sess-1/op/op-1");

  try {
    const pending = grabTheApple("sess-1", () => stream);
    await waitFor(() => stream.listenerCount("done") > 0);
    stream.dispatch("done", {
      ok: true,
      arm: "left",
      target_body: "Scene_apple_01",
      scene_revision: "rev-3",
    });
    const payload = await pending;
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/grab_apple");
    assert.deepEqual(calls[0]?.init, createGrabTheAppleRequest());
    assert.equal(stream.url, "/python/zapdos/sess-1/op/op-1");
    assert.equal(stream.closed, true);
    assert.equal(payload.target_body, "Scene_apple_01");
    assert.equal(payload.scene_revision, "rev-3");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

async function loadModule<TModule>(specifier: string): Promise<TModule> {
  const loaded = await import(specifier);
  return (loaded.default ?? loaded["module.exports"] ?? loaded) as TModule;
}

async function waitFor(predicate: () => boolean, timeoutMs = 1000) {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() > deadline) {
      throw new Error("Timed out waiting for test condition");
    }
    await Promise.resolve();
  }
}

class FakeEventSource {
  closed = false;
  onerror: ((event: Event) => void) | null = null;
  private readonly listeners = new Map<string, Array<(event: MessageEvent) => void>>();

  constructor(readonly url: string) {}

  addEventListener(type: string, listener: (event: MessageEvent) => void) {
    const current = this.listeners.get(type) ?? [];
    current.push(listener);
    this.listeners.set(type, current);
  }

  close() {
    this.closed = true;
  }

  dispatch(type: string, payload: unknown) {
    for (const listener of this.listeners.get(type) ?? []) {
      listener(new MessageEvent(type, { data: JSON.stringify(payload) }));
    }
  }

  listenerCount(type: string) {
    return this.listeners.get(type)?.length ?? 0;
  }
}
