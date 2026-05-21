import assert from "node:assert/strict";
import test from "node:test";

type PlaceTheAppleModule = typeof import("./place-the-apple");

test("createPlaceTheAppleRequest posts the canned cube release payload", async () => {
  const { createPlaceTheAppleRequest } = await loadModule<PlaceTheAppleModule>("./place-the-apple.ts");

  assert.deepEqual(createPlaceTheAppleRequest(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([]),
  });
});

test("placeTheApple posts to the zapdos place_apple route", async () => {
  const { createPlaceTheAppleRequest, placeTheApple } = await loadModule<PlaceTheAppleModule>("./place-the-apple.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      ok: true,
      op_id: "op-2",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;
  const stream = new FakeEventSource("/python/zapdos/sess-1/op/op-2");

  try {
    const pending = placeTheApple("sess-1", () => stream);
    await waitFor(() => stream.listenerCount("done") > 0);
    stream.dispatch("done", {
      ok: true,
      arm: "left",
      target_body: "Scene_benchmark_building_blocks_006_01",
      scene_revision: "rev-4",
    });
    const payload = await pending;
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/place_apple");
    assert.deepEqual(calls[0]?.init, createPlaceTheAppleRequest());
    assert.equal(stream.url, "/python/zapdos/sess-1/op/op-2");
    assert.equal(stream.closed, true);
    assert.equal(payload.target_body, "Scene_benchmark_building_blocks_006_01");
    assert.equal(payload.scene_revision, "rev-4");
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
