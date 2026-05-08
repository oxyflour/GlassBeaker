import assert from "node:assert/strict";
import test from "node:test";

type AddBenchmarkTableModule = typeof import("./add-benchmark-table");

test("createAddBenchmarkTableRequest posts the benchmark table shortcut payload", async () => {
  const { createAddBenchmarkTableRequest } = await loadModule<AddBenchmarkTableModule>("./add-benchmark-table.ts");

  assert.deepEqual(createAddBenchmarkTableRequest(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify([
      [{
        asset_id: "benchmark_table_000",
        motion: "static",
        placement: {
          kind: "floor_at_xy",
          xy: [0.5, 0],
          z_offset: 0,
          yaw: 0,
        },
      }],
    ]),
  });
});

test("addBenchmarkTable posts to the zapdos set_scene_assets route", async () => {
  const { addBenchmarkTable, createAddBenchmarkTableRequest } = await loadModule<AddBenchmarkTableModule>("./add-benchmark-table.ts");
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
    const pending = addBenchmarkTable("sess-1", () => stream);
    await waitFor(() => stream.listenerCount("done") > 0);
    stream.dispatch("done", {
      items: [{
        body: "Scene_benchmark_table_000_01",
        instance_id: "benchmark_table_000_01",
        asset_id: "benchmark_table_000",
      }],
      scene_revision: "rev-2",
    });
    const payload = await pending;
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/set_scene_assets");
    assert.deepEqual(calls[0]?.init, createAddBenchmarkTableRequest());
    assert.equal(stream.closed, true);
    assert.equal(payload.instance_id, "benchmark_table_000_01");
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
