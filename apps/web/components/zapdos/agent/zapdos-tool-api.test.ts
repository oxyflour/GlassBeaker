import assert from "node:assert/strict";
import test from "node:test";

type ZapdosToolApiModule = typeof import("./zapdos-tool-api");

test("createSetSceneAssetsRequest posts the full asset batch", async () => {
  const { createSetSceneAssetsRequest } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");

  assert.deepEqual(
    createSetSceneAssetsRequest({
      assets: [{
        asset_id: "table_000",
        motion: "static",
        placement: { kind: "floor_at_xy", xy: [0, 0], z_offset: 0, yaw: 0 },
      }],
    }),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([[{
        asset_id: "table_000",
        motion: "static",
        placement: { kind: "floor_at_xy", xy: [0, 0], z_offset: 0, yaw: 0 },
      }]]),
    }
  );
});

test("listSceneBodies posts to the zapdos route", async () => {
  const { createSceneToolRequest, listSceneBodies } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      items: [],
      robot_bounds: { min: [0, 0, 0], max: [1, 1, 1] },
      scene_revision: "rev-1",
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const payload = await listSceneBodies("sess-1");
    assert.equal(payload.scene_revision, "rev-1");
    assert.deepEqual(payload.robot_bounds, { min: [0, 0, 0], max: [1, 1, 1] });
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/list_scene_bodies");
    assert.deepEqual(calls[0]?.init, createSceneToolRequest([]));
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("setSceneAssets posts to the batch route and returns the started operation payload", async () => {
  const { createSetSceneAssetsRequest, setSceneAssets } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");
  const calls: Array<{ input: unknown; init: unknown }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: unknown, init?: unknown) => {
    calls.push({ input, init });
    return new Response(JSON.stringify({
      ok: true,
      op_id: "op-1",
      status: "started",
      items: [{
        asset_id: "table_000",
        body: "Scene_table_000_01",
        instance_id: "table_000_01",
      }],
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as typeof fetch;

  try {
    const payload = await setSceneAssets("sess-1", {
      assets: [{
        asset_id: "table_000",
        motion: "static",
        placement: { kind: "floor_at_xy", xy: [0, 0], z_offset: 0, yaw: 0 },
      }],
    });
    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/set_scene_assets");
    assert.deepEqual(calls[0]?.init, createSetSceneAssetsRequest({
      assets: [{
        asset_id: "table_000",
        motion: "static",
        placement: { kind: "floor_at_xy", xy: [0, 0], z_offset: 0, yaw: 0 },
      }],
    }));
    assert.equal(payload.op_id, "op-1");
    assert.equal(payload.status, "started");
    assert.equal(payload.items[0]?.instance_id, "table_000_01");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("removeAssetFromScene posts then waits for the operation stream", async () => {
  const { createSceneOpStreamUrl, createSceneToolRequest, removeAssetFromScene } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");
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
  const stream = new FakeEventSource(createSceneOpStreamUrl("sess-1", "op-2"));

  try {
    const pending = removeAssetFromScene("sess-1", "table_000_01", () => stream);
    await waitFor(() => stream.listenerCount("done") > 0);
    stream.dispatch("done", {
      instance_id: "table_000_01",
      scene_revision: "rev-3",
    });
    const payload = await pending;

    assert.equal(calls[0]?.input, "/python/zapdos/sess-1/call/remove_asset_from_scene");
    assert.deepEqual(calls[0]?.init, createSceneToolRequest(["table_000_01"]));
    assert.equal(stream.url, "/python/zapdos/sess-1/op/op-2");
    assert.equal(stream.closed, true);
    assert.equal(payload.instance_id, "table_000_01");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("waitForSceneToolOp rejects failed operation events", async () => {
  const { waitForSceneToolOp } = await loadModule<ZapdosToolApiModule>("./zapdos-tool-api.ts");
  const stream = new FakeEventSource("/python/zapdos/sess-1/op/op-3");

  const pending = waitForSceneToolOp("sess-1", "op-3", () => stream);
  await Promise.resolve();
  stream.dispatch("failed", { detail: "Scene rebuild already in progress" });

  await assert.rejects(pending, /Scene rebuild already in progress/);
  assert.equal(stream.closed, true);
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
